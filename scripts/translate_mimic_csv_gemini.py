import argparse
import csv
import json
import os
import time
from pathlib import Path
from urllib import error, parse, request


DEFAULT_INPUT_CSV = Path("data/raw/mimic_cxr_text_only.csv")
DEFAULT_OUTPUT_CSV = Path("data/processed/mimic_cxr_text_only_tr_gemini.csv")
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_LIMIT = 10
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_RETRIES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini model name.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="How many rows to translate. Use 0 for all rows.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=GEMINI_API_BASE,
        help="Gemini API base URL.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output CSV if it exists.",
    )
    return parser.parse_args()


def build_prompt(text: str) -> str:
    return (
        "Translate the following English radiology report text into Turkish.\n"
        "Use the natural formal style used in radiology reports written in hospitals in Turkey.\n"
        "Preserve the medical meaning accurately.\n"
        "Write in concise, professional report language.\n"
        "Do not summarize.\n"
        "Do not explain.\n"
        "Do not add interpretation.\n"
        "Translate all content into Turkish, including medical terms.\n"
        "Do not leave English medical words in the output.\n"
        "Avoid conversational language.\n"
        "Sound like a real Turkish radiology report, not a word-for-word translation.\n"
        "Output only the Turkish translation.\n\n"
        f"{text}"
    )


def extract_text(response_body: dict) -> str:
    candidates = response_body.get("candidates", [])
    if not candidates:
        prompt_feedback = response_body.get("promptFeedback")
        raise RuntimeError(f"Gemini yaniti bos veya engellendi: {prompt_feedback or response_body}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_chunks = []
    for part in parts:
        text = part.get("text")
        if text:
            text_chunks.append(text)

    content = "\n".join(text_chunks).strip()
    if not content:
        raise RuntimeError(f"Gemini bos metin dondurdu: {response_body}")
    return content


def parse_retry_delay(details: list[dict]) -> int | None:
    for detail in details:
        retry_delay = detail.get("retryDelay")
        if isinstance(retry_delay, str) and retry_delay.endswith("s"):
            try:
                return max(1, int(float(retry_delay[:-1])))
            except ValueError:
                return None
    return None


def parse_translation_pair(text: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result = {"findings_tr": "", "impression_tr": ""}

    current_key = None
    for line in lines:
        upper_line = line.upper()
        if upper_line.startswith("FINDINGS_TR:"):
            current_key = "findings_tr"
            result[current_key] = line.split(":", 1)[1].strip()
            continue
        if upper_line.startswith("IMPRESSION_TR:"):
            current_key = "impression_tr"
            result[current_key] = line.split(":", 1)[1].strip()
            continue

        if current_key:
            result[current_key] = f"{result[current_key]} {line}".strip()

    if not result["findings_tr"] and not result["impression_tr"]:
        parts = text.split("\n\n", 1)
        result["findings_tr"] = parts[0].strip()
        result["impression_tr"] = parts[1].strip() if len(parts) > 1 else ""

    return result


def translate_row(
    findings: str,
    impression: str,
    model: str,
    api_base: str,
    api_key: str,
) -> dict[str, str]:
    clean_findings = (findings or "").strip()
    clean_impression = (impression or "").strip()
    if not clean_findings and not clean_impression:
        return {"findings_tr": "", "impression_tr": ""}

    endpoint = f"{api_base}/{model}:generateContent?key={parse.quote(api_key)}"
    row_prompt = (
        f"{build_prompt('')}"
        "Translate the following two sections and return them in exactly this format:\n"
        "FINDINGS_TR: <Turkish translation>\n"
        "IMPRESSION_TR: <Turkish translation>\n\n"
        f"FINDINGS: {clean_findings}\n"
        f"IMPRESSION: {clean_impression}"
    )
    payload = {
        "system_instruction": {
            "parts": [{"text": "Return only the requested Turkish report fields."}]
        },
        "contents": [
            {
                "parts": [{"text": row_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.8,
            "maxOutputTokens": 2048,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with request.urlopen(http_request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            return parse_translation_pair(extract_text(body))
        except error.HTTPError as exc:
            details_text = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                try:
                    error_body = json.loads(details_text)
                except json.JSONDecodeError:
                    error_body = {}
                retry_delay = parse_retry_delay(error_body.get("error", {}).get("details", [])) or 30
                if attempt < MAX_RETRIES:
                    print(f"Rate limit alindi, {retry_delay} saniye beklenecek. Deneme {attempt}/{MAX_RETRIES}")
                    time.sleep(retry_delay)
                    continue
            raise RuntimeError(f"Gemini HTTP hatasi: {exc.code} {details_text}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Gemini istegi basarisiz oldu: {exc}") from exc

    raise RuntimeError("Gemini istegi tekrar denemelerden sonra da basarisiz oldu.")


def load_existing_rows(output_path: Path) -> list[dict[str, str]]:
    if not output_path.exists():
        return []

    with output_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def main() -> None:
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        print("GEMINI_API_KEY ortam degiskeni bulunamadi.")
        return

    if not args.input.exists():
        print(f"Girdi CSV bulunamadi: {args.input}")
        return

    with args.input.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        print("Girdi CSV bos.")
        return

    row_limit = len(rows) if args.limit == 0 else min(args.limit, len(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = load_existing_rows(args.output) if args.resume else []
    output_rows = list(existing_rows)
    start_index = min(len(existing_rows), row_limit)

    print(f"Model: {args.model}")
    print(f"Toplam satir: {len(rows)}")
    print(f"Cevirilecek satir: {row_limit}")
    if args.resume:
        print(f"Mevcut ceviri sayisi: {len(existing_rows)}")

    for index, row in enumerate(rows[start_index:row_limit], start=start_index + 1):
        translated_row = translate_row(
            row.get("findings", ""),
            row.get("impression", ""),
            args.model,
            args.api_base,
            api_key,
        )
        output_rows.append(translated_row)

        with args.output.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["findings_tr", "impression_tr"])
            writer.writeheader()
            writer.writerows(output_rows)

        print(f"Satir {index} cevrildi")

    output_fieldnames = ["findings_tr", "impression_tr"]

    with args.output.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Cikti kaydedildi: {args.output.resolve()}")


if __name__ == "__main__":
    main()
