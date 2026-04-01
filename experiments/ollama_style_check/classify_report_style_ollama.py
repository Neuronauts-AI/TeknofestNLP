import argparse
import sys
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
from urllib import error, request


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "lfm2-8B-A1B:latest"
REFERENCE_STYLE_PATH = Path("docs/reference_report_style.txt")
DEFAULT_CSV_PATH = Path("data/processed/mimic_cxr_text_only_tr_gemini.csv")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
PROMPT_PATH = Path("app/prompts/ollama_style_prompt.txt")


@dataclass
class StyleResult:
    label: str
    score: float
    reason: str
    raw_response: str


def load_reference_style() -> str:
    if not REFERENCE_STYLE_PATH.exists():
        raise RuntimeError(f"Referans metin bulunamadı: {REFERENCE_STYLE_PATH}")
    return REFERENCE_STYLE_PATH.read_text(encoding="utf-8").strip()


def build_system_prompt(reference_style: str) -> str:
    if not PROMPT_PATH.exists():
        raise RuntimeError(f"Prompt dosyası bulunamadı: {PROMPT_PATH}")
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(reference_style=reference_style)


def build_report_text(findings: str, impression: str, text: str) -> str:
    if text.strip():
        return text.strip()

    sections = []
    if findings.strip():
        sections.append(f"Bulgular:\n{findings.strip()}")
    if impression.strip():
        sections.append(f"Sonuç:\n{impression.strip()}")
    return "\n\n".join(sections).strip()


def call_ollama(report_text: str, model: str) -> str:
    reference_style = load_reference_style()
    prompt = f"Rapor:\n{report_text}"
    payload = {
        "model": model,
        "system": build_system_prompt(reference_style),
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }

    http_request = request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=180) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP hatası: {exc.code} {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(
            "Ollama'ya bağlanılamadı. `ollama serve` çalışıyor mu kontrol et."
        ) from exc

    parsed = json.loads(body)
    return parsed.get("response", "").strip()


def parse_result(raw_response: str) -> StyleResult:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model geçerli JSON döndürmedi: {raw_response}") from exc

    label = str(parsed.get("label", "")).strip()
    if label not in {"uygun", "uygun_degil"}:
        raise RuntimeError(f"Geçersiz label döndü: {label}")

    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Geçersiz score döndü: {parsed.get('score')}") from exc

    score = max(0.0, min(1.0, round(score, 4)))
    reason = str(parsed.get("reason", "")).strip()
    return StyleResult(label=label, score=score, reason=reason, raw_response=raw_response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ollama ile Türkçe radyoloji raporu üslup/gerçeklik sınıflandırması yapar."
    )
    parser.add_argument("--findings", default="", help="Bulgular metni.")
    parser.add_argument("--impression", default="", help="Sonuç/izlenim metni.")
    parser.add_argument("--text", default="", help="Hazır tam rapor metni.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model adı.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Toplu test için CSV yolu.")
    parser.add_argument("--limit", type=int, default=0, help="CSV testinde kaç satır işleneceği.")
    args = parser.parse_args()

    if args.limit > 0:
        if not args.csv.exists():
            raise SystemExit(f"CSV bulunamadı: {args.csv}")

        results = []
        with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                findings = row.get("findings_tr", "")
                impression = row.get("impression_tr", "")
                report_text = build_report_text(findings, impression, "")
                raw_response = call_ollama(report_text, args.model)
                result = parse_result(raw_response)
                results.append(
                    {
                        "row": index,
                        "label": result.label,
                        "score": result.score,
                        "reason": result.reason,
                        "report_preview": report_text[:220],
                    }
                )
                if len(results) >= args.limit:
                    break

        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    report_text = build_report_text(args.findings, args.impression, args.text)
    if not report_text:
        raise SystemExit("Sınıflandırmak için --text ya da --findings/--impression ver.")

    raw_response = call_ollama(report_text, args.model)
    result = parse_result(raw_response)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
