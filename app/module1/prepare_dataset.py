import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_INPUT = Path("data/processed/mimic_cxr_text_only_tr_gemini.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/module1")
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input Turkish CSV path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for cleaned train/val outputs.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio.")
    return parser.parse_args()


def clean_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def build_report(findings: str, impression: str) -> str:
    sections = []
    if findings:
        sections.append(f"Bulgular:\n{findings}")
    if impression:
        sections.append(f"Sonuç:\n{impression}")
    return "\n\n".join(sections)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        print(f"Girdi dosyasi bulunamadi: {args.input}")
        return

    with args.input.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    cleaned_rows = []
    for row in rows:
        findings = clean_text(row.get("findings_tr", ""))
        impression = clean_text(row.get("impression_tr", ""))
        if not findings and not impression:
            continue
        cleaned_rows.append(
            {
                "findings_tr": findings,
                "impression_tr": impression,
                "report_tr": build_report(findings, impression),
            }
        )

    random.Random(args.seed).shuffle(cleaned_rows)
    split_index = max(1, int(len(cleaned_rows) * (1 - args.val_ratio))) if cleaned_rows else 0
    train_rows = cleaned_rows[:split_index]
    val_rows = cleaned_rows[split_index:]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, subset in (("train", train_rows), ("val", val_rows), ("all", cleaned_rows)):
        csv_path = args.output_dir / f"{name}.csv"
        jsonl_path = args.output_dir / f"{name}.jsonl"

        with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["findings_tr", "impression_tr", "report_tr"])
            writer.writeheader()
            writer.writerows(subset)

        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for item in subset:
                jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Toplam temiz kayit: {len(cleaned_rows)}")
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)}")
    print(f"Cikti klasoru: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
