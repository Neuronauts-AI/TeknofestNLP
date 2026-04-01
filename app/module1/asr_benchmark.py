from __future__ import annotations

import argparse
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.module1.asr_whisper import DEFAULT_ASR_MODEL, transcribe_audio_file
from app.module1.transcript_sections import parse_transcript_sections


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def similarity(a: str, b: str) -> float:
    return round(SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio(), 4)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def benchmark_row(row: dict[str, str], model_id: str, base_dir: Path) -> dict[str, object]:
    audio_path = Path(row["audio_path"])
    if not audio_path.is_absolute():
        audio_path = (base_dir / audio_path).resolve()

    asr_result = transcribe_audio_file(audio_path, model_id=model_id)
    parsed = parse_transcript_sections(asr_result.text)

    reference_transcript = row.get("reference_transcript", "")
    reference_findings = row.get("reference_findings", "")
    reference_impression = row.get("reference_impression", "")

    transcript_score = similarity(asr_result.text, reference_transcript) if reference_transcript else None
    findings_score = similarity(parsed.findings, reference_findings) if reference_findings else None
    impression_score = similarity(parsed.impression, reference_impression) if reference_impression else None

    return {
        "audio_path": str(audio_path),
        "quality": row.get("quality", ""),
        "model_id": model_id,
        "transcript": asr_result.text,
        "parsed_findings": parsed.findings,
        "parsed_impression": parsed.impression,
        "reference_transcript": reference_transcript,
        "reference_findings": reference_findings,
        "reference_impression": reference_impression,
        "transcript_score": transcript_score,
        "findings_score": findings_score,
        "impression_score": impression_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Whisper ASR on radiology dictation samples.")
    parser.add_argument("--manifest", required=True, help="CSV manifest path.")
    parser.add_argument("--model", default=DEFAULT_ASR_MODEL, help="Whisper model id.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    rows = load_manifest(manifest_path)
    base_dir = manifest_path.parent

    results = [benchmark_row(row, args.model, base_dir) for row in rows]

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Çıktı kaydedildi: {output_path}")

    for item in results:
        print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
