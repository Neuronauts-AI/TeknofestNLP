import argparse
import csv
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "itsanmolgupta/mimic-cxr-dataset"
OUTPUT_DIR = Path("data/raw")
IMAGE_DIR = OUTPUT_DIR / "images"
DEFAULT_LIMIT = 10
CSV_NAME = "mimic_cxr_text_only.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="How many samples to fetch from the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading dataset: {DATASET_NAME}")
    print(f"Sample limit: {args.limit}")

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / CSV_NAME

    rows = []
    text_columns = None

    for index, sample in enumerate(dataset.take(args.limit), start=1):
        image_path = IMAGE_DIR / f"mimic_cxr_sample_{index:02d}.png"
        sample["image"].save(image_path)

        if text_columns is None:
            text_columns = [key for key in sample.keys() if key != "image"]

        row = {column: sample.get(column, "") for column in text_columns}
        rows.append(row)
        print(f"Sample {index} hazirlandi - Gorsel: {image_path}")

    if not rows or text_columns is None:
        print("No rows were loaded from the dataset.")
        return

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=text_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV kaydedildi: {output_path.resolve()}")
    print(f"Kolonlar: {', '.join(text_columns)}")


if __name__ == "__main__":
    main()
