import argparse
import csv
import json
import sys
from pathlib import Path
from urllib import error, request


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "ministral-3:14b"
REFERENCE_STYLE_PATH = Path("docs/reference_report_style.txt")
PROMPT_PATH = Path("experiments/quality_control/quality_control_prompt.txt")


def load_reference_style() -> str:
    return REFERENCE_STYLE_PATH.read_text(encoding="utf-8").strip()


def load_prompt() -> str:
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
    return json.loads(outer.get("response", "{}"))


def normalize_result(parsed: dict) -> dict:
    label = str(parsed.get("overall_label", "")).strip()
    if label not in {"uygun", "sinirda", "uygun_degil"}:
        raise ValueError(f"Geçersiz label: {label}")

    subscores = parsed.get("subscores", {})
    normalized_subscores = {
        "dil_kalitesi": clamp_score(subscores.get("dil_kalitesi", 0.0)),
        "terminoloji_tutarliligi": clamp_score(subscores.get("terminoloji_tutarliligi", 0.0)),
        "yapi_uygunlugu": clamp_score(subscores.get("yapi_uygunlugu", 0.0)),
        "sonuc_yeterliligi": clamp_score(subscores.get("sonuc_yeterliligi", 0.0)),
    }
    issues = [str(item).strip() for item in parsed.get("issues", []) if str(item).strip()]
    return {
        "overall_label": label,
        "overall_score": clamp_score(parsed.get("overall_score", 0.0)),
        "subscores": normalized_subscores,
        "issues": issues,
        "summary": str(parsed.get("summary", "")).strip(),
    }


def run_single_report(report_text: str, model: str) -> dict:
    return normalize_result(call_ollama(report_text, model))


def pick_text(row: dict, *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def run_csv_mode(csv_path: Path, limit: int, model: str) -> dict:
    rows_output = []
    distribution = {"uygun": 0, "sinirda": 0, "uygun_degil": 0}
    score_total = 0.0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(reader, start=1):
            if limit and index > limit:
                break

            findings = pick_text(row, "findings_tr", "findings")
            impression = pick_text(row, "impression_tr", "impression")
            report_text = build_report_text(findings, impression)
            if not report_text:
                continue

            result = run_single_report(report_text, model)
            distribution[result["overall_label"]] += 1
            score_total += result["overall_score"]
            rows_output.append(
                {
                    "row": index,
                    "overall_label": result["overall_label"],
                    "overall_score": result["overall_score"],
                    "subscores": result["subscores"],
                    "issues": result["issues"],
                    "summary": result["summary"],
                    "report_preview": report_text[:280],
                }
            )

    count = len(rows_output)
    return {
        "csv_path": str(csv_path),
        "model": model,
        "count": count,
        "distribution": distribution,
        "average_score": round(score_total / count, 4) if count else 0.0,
        "rows": rows_output,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Deneysel rapor kalite kontrolü")
    parser.add_argument("--findings", default="", help="Bulgular metni")
    parser.add_argument("--impression", default="", help="Sonuç metni")
    parser.add_argument("--text", default="", help="Hazır tam rapor metni")
    parser.add_argument("--csv", default="", help="Toplu değerlendirme için CSV dosyası")
    parser.add_argument("--limit", type=int, default=0, help="CSV modunda işlenecek satır sayısı")
    parser.add_argument("--output", default="", help="JSON sonucu dosyaya yaz")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model adı")
    args = parser.parse_args()

    try:
        if args.csv.strip():
            result = run_csv_mode(Path(args.csv.strip()), args.limit, args.model)
        else:
            report_text = args.text.strip() or build_report_text(args.findings, args.impression)
            if not report_text:
                raise SystemExit("Değerlendirilecek rapor metni yok.")
            result = run_single_report(report_text, args.model)
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Ollama HTTP hatası: {exc.code} {details}") from exc
    except error.URLError as exc:
        raise SystemExit(f"Ollama bağlantı hatası: {exc}") from exc

    if args.output.strip():
        output_path = Path(args.output.strip())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
