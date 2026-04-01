import json
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "ministral-3:14b"
PROMPT_PATH = Path("app/prompts/ollama_critical_alerts_prompt.txt")


@dataclass
class CriticalAlert:
    finding: str
    severity: str
    status: str
    reason: str


@dataclass
class CriticalAlertResult:
    available: bool
    model: str
    alerts: list[CriticalAlert]
    error: str = ""


def build_system_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise RuntimeError(f"Prompt dosyası bulunamadı: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


def _call_ollama(report_text: str, model: str) -> str:
    payload = {
        "model": model,
        "system": build_system_prompt(),
        "prompt": f"Rapor:\n{report_text}",
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    http_request = request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=180) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    return parsed.get("response", "").strip()


def classify_critical_alerts(report_text: str, model: str = DEFAULT_MODEL) -> CriticalAlertResult:
    if not report_text.strip():
        return CriticalAlertResult(available=False, model=model, alerts=[], error="empty_report")

    try:
        raw_response = _call_ollama(report_text, model)
        parsed = json.loads(raw_response)
        alerts: list[CriticalAlert] = []
        for item in parsed.get("alerts", []):
            finding = str(item.get("finding", "")).strip()
            severity = str(item.get("severity", "")).strip().lower()
            status = str(item.get("status", "")).strip().lower()
            reason = str(item.get("reason", "")).strip()
            if not finding or severity not in {"kritik", "yuksek", "orta"} or status not in {"present", "absent", "uncertain"}:
                continue
            alerts.append(
                CriticalAlert(
                    finding=finding,
                    severity=severity,
                    status=status,
                    reason=reason,
                )
            )
        return CriticalAlertResult(available=True, model=model, alerts=alerts)
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        return CriticalAlertResult(available=False, model=model, alerts=[], error=f"http_{exc.code}: {details}")
    except error.URLError:
        return CriticalAlertResult(available=False, model=model, alerts=[], error="connection_failed")
    except Exception as exc:
        return CriticalAlertResult(available=False, model=model, alerts=[], error=str(exc))
