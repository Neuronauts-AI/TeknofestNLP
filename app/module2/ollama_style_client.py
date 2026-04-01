import json
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "ministral-3:14b"
REFERENCE_STYLE_PATH = Path("docs/reference_report_style.txt")
PROMPT_PATH = Path("app/prompts/ollama_style_prompt.txt")


@dataclass
class OllamaStyleResult:
    label: str
    score: float
    reason: str
    model: str
    available: bool = True
    error: str = ""


def load_reference_style() -> str:
    if not REFERENCE_STYLE_PATH.exists():
        raise RuntimeError(f"Referans metin bulunamadı: {REFERENCE_STYLE_PATH}")
    return REFERENCE_STYLE_PATH.read_text(encoding="utf-8").strip()


def build_system_prompt(reference_style: str) -> str:
    if not PROMPT_PATH.exists():
        raise RuntimeError(f"Prompt dosyası bulunamadı: {PROMPT_PATH}")
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(reference_style=reference_style)


def build_report_text(findings: str, impression: str) -> str:
    sections = []
    if findings.strip():
        sections.append(f"Bulgular:\n{findings.strip()}")
    if impression.strip():
        sections.append(f"Sonuç:\n{impression.strip()}")
    return "\n\n".join(sections).strip()


def _call_ollama(report_text: str, model: str) -> str:
    payload = {
        "model": model,
        "system": build_system_prompt(load_reference_style()),
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


def classify_report_style_with_ollama(
    findings: str,
    impression: str,
    model: str = DEFAULT_MODEL,
) -> OllamaStyleResult:
    report_text = build_report_text(findings, impression)
    if not report_text:
        return OllamaStyleResult(
            label="uygun_degil",
            score=0.0,
            reason="Sınıflandırma için metin yok.",
            model=model,
            available=False,
            error="empty_report",
        )

    try:
        raw_response = _call_ollama(report_text, model)
        parsed = json.loads(raw_response)
        label = str(parsed.get("label", "")).strip()
        if label not in {"uygun", "uygun_degil"}:
            raise RuntimeError(f"Geçersiz label döndü: {label}")
        score = max(0.0, min(1.0, round(float(parsed.get("score", 0.0)), 4)))
        reason = str(parsed.get("reason", "")).strip()
        return OllamaStyleResult(
            label=label,
            score=score,
            reason=reason,
            model=model,
        )
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        return OllamaStyleResult(
            label="uygun_degil",
            score=0.0,
            reason="Ollama çağrısı başarısız oldu.",
            model=model,
            available=False,
            error=f"http_{exc.code}: {details}",
        )
    except error.URLError:
        return OllamaStyleResult(
            label="uygun_degil",
            score=0.0,
            reason="Ollama servisine ulaşılamadı.",
            model=model,
            available=False,
            error="connection_failed",
        )
    except Exception as exc:
        return OllamaStyleResult(
            label="uygun_degil",
            score=0.0,
            reason="Ollama sonucu ayrıştırılamadı.",
            model=model,
            available=False,
            error=str(exc),
        )
