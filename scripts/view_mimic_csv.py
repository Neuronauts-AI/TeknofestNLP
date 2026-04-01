import argparse
import csv
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DEFAULT_CSV_PATH = Path("data/raw/mimic_cxr_text_only.csv")
DEFAULT_LIMIT = 5
console = Console(safe_box=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to the CSV file to display.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="How many rows to display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        console.print(f"[bold red]CSV bulunamadi:[/bold red] {args.csv}")
        return

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    with args.csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        console.print("[bold yellow]CSV bos.[/bold yellow]")
        return

    summary = Table(title="MIMIC CXR CSV Ozeti", show_header=True, header_style="bold cyan")
    summary.add_column("Alan")
    summary.add_column("Deger", overflow="fold")
    summary.add_row("Dosya", str(args.csv.resolve()))
    summary.add_row("Toplam satir", str(len(rows)))
    summary.add_row("Gosterilen satir", str(min(args.limit, len(rows))))
    summary.add_row("Kolonlar", ", ".join(fieldnames))
    console.print(summary)

    for index, row in enumerate(rows[: args.limit], start=1):
        body = Text()
        for column, value in row.items():
            clean_value = (value or "").strip()
            body.append(f"{column.upper()}\n", style="bold green")
            body.append((clean_value if clean_value else "-") + "\n\n")

        console.print(
            Panel(
                body,
                title=f"Satir {index}",
                border_style="blue",
                expand=True,
            )
        )


if __name__ == "__main__":
    main()
