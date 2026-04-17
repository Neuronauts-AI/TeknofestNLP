import json
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "ministral-3:14b"
REFERENCE_STYLE_PATH = Path("docs/reference_report_style.txt")
PROMPT_PATH = Path("app/prompts/quality_control_prompt.txt")


@dataclass
class QualityControlResult:
    available: bool
    model: str
    overall_label: str
    overall_score: float
    subscores: dict
    issues: list[str]
    summary: str
    error: str = ""


def load_reference_style() -> str:
    if not REFERENCE_STYLE_PATH.exists():
        raise RuntimeError(f"Referans metin bulunamadı: {REFERENCE_STYLE_PATH}")
    return REFERENCE_STYLE_PATH.read_text(encoding="utf-8").strip()


def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise RuntimeError(f"Prompt dosyası bulunamadı: {PROMPT_PATH}")
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(reference_style=load_reference_style())


def build_report_text(findings: str, impression: str) -> str:
    sections = []
    if findings.strip():
        sections.append(f"Bulgular:\n{findings.strip()}")
    if impression.strip():
        sections.append(f"Sonuç:\n{impression.strip()}")
    return "\n\n".join(sections).strip()


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, round(float(value), 4)))


def normalize_result(parsed: dict, model: str) -> QualityControlResult:
    label = str(parsed.get("overall_label", "")).strip()
    if label not in {"uygun", "sinirda", "uygun_degil"}:
        raise RuntimeError(f"Geçersiz kalite etiketi: {label}")

    subscores = parsed.get("subscores", {})
    normalized_subscores = {
        "dil_kalitesi": clamp_score(subscores.get("dil_kalitesi", 0.0)),
        "terminoloji_tutarliligi": clamp_score(subscores.get("terminoloji_tutarliligi", 0.0)),
        "yapi_uygunlugu": clamp_score(subscores.get("yapi_uygunlugu", 0.0)),
        "sonuc_yeterliligi": clamp_score(subscores.get("sonuc_yeterliligi", 0.0)),
    }
    issues = [str(item).strip() for item in parsed.get("issues", []) if str(item).strip()]

    return QualityControlResult(
        available=True,
        model=model,
        overall_label=label,
        overall_score=clamp_score(parsed.get("overall_score", 0.0)),
        subscores=normalized_subscores,
        issues=issues,
        summary=str(parsed.get("summary", "")).strip(),
    )


def call_ollama(report_text: str, model: str) -> dict:
    payload = {
        "model": model,
        "system": load_prompt(),
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
    outer = json.loads(body)
    content = str(outer.get("response", "")).strip()
    if not content:
        content = str(outer.get("thinking", "")).strip()
    if not content:
        raise RuntimeError("Ollama boş yanıt döndürdü.")
    return json.loads(content)


def classify_quality_control(
    findings: str,
    impression: str,
    model: str = DEFAULT_MODEL,
) -> QualityControlResult:
    report_text = build_report_text(findings, impression)
    if not report_text:
        return QualityControlResult(
            available=False,
            model=model,
            overall_label="uygun_degil",
            overall_score=0.0,
            subscores={
                "dil_kalitesi": 0.0,
                "terminoloji_tutarliligi": 0.0,
                "yapi_uygunlugu": 0.0,
                "sonuc_yeterliligi": 0.0,
            },
            issues=[],
            summary="Kalite kontrol için rapor metni yok.",
            error="empty_report",
        )

    try:
        return normalize_result(call_ollama(report_text, model), model)
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        return QualityControlResult(
            available=False,
            model=model,
            overall_label="uygun_degil",
            overall_score=0.0,
            subscores={
                "dil_kalitesi": 0.0,
                "terminoloji_tutarliligi": 0.0,
                "yapi_uygunlugu": 0.0,
                "sonuc_yeterliligi": 0.0,
            },
            issues=[],
            summary="Kalite kontrol çağrısı başarısız oldu.",
            error=f"http_{exc.code}: {details}",
        )
    except error.URLError:
        return QualityControlResult(
            available=False,
            model=model,
            overall_label="uygun_degil",
            overall_score=0.0,
            subscores={
                "dil_kalitesi": 0.0,
                "terminoloji_tutarliligi": 0.0,
                "yapi_uygunlugu": 0.0,
                "sonuc_yeterliligi": 0.0,
            },
            issues=[],
            summary="Kalite kontrol servisine ulaşılamadı.",
            error="connection_failed",
        )
    except Exception as exc:
        return QualityControlResult(
            available=False,
            model=model,
            overall_label="uygun_degil",
            overall_score=0.0,
            subscores={
                "dil_kalitesi": 0.0,
                "terminoloji_tutarliligi": 0.0,
                "yapi_uygunlugu": 0.0,
                "sonuc_yeterliligi": 0.0,
            },
            issues=[],
            summary="Kalite kontrol sonucu ayrıştırılamadı.",
            error=str(exc),
        )
