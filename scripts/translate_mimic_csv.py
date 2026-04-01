import argparse
import csv
import json
import os
from pathlib import Path
from urllib import error, request


DEFAULT_INPUT_CSV = Path("data/raw/mimic_cxr_text_only.csv")
DEFAULT_OUTPUT_CSV = Path("data/processed/mimic_cxr_text_only_tr.csv")
DEFAULT_MODEL = "minimax/minimax-m2.5:free"
DEFAULT_LIMIT = 10
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_DATA_COLLECTION = "allow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenRouter model name.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="How many rows to translate. Use 0 for all rows.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=OPENROUTER_URL,
        help="OpenRouter chat completions API URL.",
    )
    parser.add_argument(
        "--data-collection",
        type=str,
        default=DEFAULT_DATA_COLLECTION,
        choices=["allow", "deny"],
        help="OpenRouter provider data policy filter.",
    )
    return parser.parse_args()


def build_prompt(text: str) -> str:
    return (
        "You are a professional English (en) to Turkish (tr) translator for radiology reports.\n"
        "Translate the text accurately.\n"
        "Preserve medical terminology and clinical meaning.\n"
        "Do not summarize.\n"
        "Do not explain.\n"
        "Produce only the Turkish translation.\n\n"
        f"{text}"
    )


def translate_text(
    text: str,
    model: str,
    api_url: str,
    api_key: str,
    data_collection: str,
) -> str:
    clean_text = (text or "").strip()
    if not clean_text:
        return ""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_prompt(clean_text),
            }
        ],
        "temperature": 0.1,
        "provider": {"data_collection": data_collection},
    }
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        api_url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP hatasi: {exc.code} {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouter istegi basarisiz oldu: {exc}") from exc

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"Model yaniti gecersiz: {body}")

    message = choices[0].get("message", {})
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"Model bos yanit verdi: {body}")
    return content


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        print("OPENROUTER_API_KEY ortam degiskeni bulunamadi.")
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
    output_rows = []

    print(f"Model: {args.model}")
    print(f"Toplam satir: {len(rows)}")
    print(f"Cevirilecek satir: {row_limit}")
    print(f"Data policy: {args.data_collection}")

    for index, row in enumerate(rows[:row_limit], start=1):
        translated_row = dict(row)
        translated_row["findings_tr"] = translate_text(
            row.get("findings", ""),
            args.model,
            args.api_url,
            api_key,
            args.data_collection,
        )
        translated_row["impression_tr"] = translate_text(
            row.get("impression", ""),
            args.model,
            args.api_url,
            api_key,
            args.data_collection,
        )
        output_rows.append(translated_row)
        print(f"Satir {index} cevrildi")

    output_fieldnames = list(fieldnames)
    if "findings_tr" not in output_fieldnames:
        output_fieldnames.append("findings_tr")
    if "impression_tr" not in output_fieldnames:
        output_fieldnames.append("impression_tr")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Cikti kaydedildi: {args.output.resolve()}")


if __name__ == "__main__":
    main()
