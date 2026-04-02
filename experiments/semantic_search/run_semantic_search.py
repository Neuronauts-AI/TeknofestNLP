import argparse
import csv
import json
import math
import sys
from pathlib import Path
from urllib import error, request


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_CSV = Path("data/processed/mimic_cxr_text_only_tr_gemini.csv")
DEFAULT_INDEX = Path("data/processed/semantic_search_index.json")


def pick_text(row: dict, *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def build_report_text(findings: str, impression: str) -> str:
    sections = []
    if findings:
        sections.append(f"Bulgular:\n{findings}")
    if impression:
        sections.append(f"Sonuç:\n{impression}")
    return "\n\n".join(sections).strip()


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    payload = {"model": model, "input": texts}
    http_request = request.Request(
        OLLAMA_EMBED_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["embeddings"]


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = vector_norm(left)
    right_norm = vector_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot / (left_norm * right_norm)


def load_rows(csv_path: Path, limit: int) -> list[dict]:
    rows = []
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
            rows.append(
                {
                    "row": index,
                    "findings": findings,
                    "impression": impression,
                    "report_text": report_text,
                }
            )
    return rows


def build_index(csv_path: Path, output_path: Path, model: str, limit: int, batch_size: int) -> dict:
    rows = load_rows(csv_path, limit)
    embeddings: list[list[float]] = []

    for batch in batched([row["report_text"] for row in rows], batch_size):
        embeddings.extend(embed_texts(batch, model))

    index_payload = {
        "model": model,
        "source_csv": str(csv_path),
        "count": len(rows),
        "items": [
            {
                "row": row["row"],
                "findings": row["findings"],
                "impression": row["impression"],
                "report_text": row["report_text"],
                "embedding": embedding,
            }
            for row, embedding in zip(rows, embeddings)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index_payload, ensure_ascii=False), encoding="utf-8")
    return {
        "index_path": str(output_path),
        "model": model,
        "count": len(rows),
        "source_csv": str(csv_path),
    }


def query_index(index_path: Path, text: str, model: str, top_k: int) -> dict:
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    query_embedding = embed_texts([text], model)[0]
    scored = []

    for item in index_payload["items"]:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append(
            {
                "row": item["row"],
                "score": round(score, 4),
                "findings": item["findings"],
                "impression": item["impression"],
                "report_preview": item["report_text"][:320],
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return {
        "index_path": str(index_path),
        "query": text,
        "model": model,
        "top_k": top_k,
        "results": scored[:top_k],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Deneysel semantik arama")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Embedding indeksi oluştur")
    build_parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Kaynak CSV")
    build_parser.add_argument("--output", default=str(DEFAULT_INDEX), help="İndeks JSON yolu")
    build_parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding modeli")
    build_parser.add_argument("--limit", type=int, default=0, help="İsteğe bağlı satır limiti")
    build_parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch boyutu")

    query_parser = subparsers.add_parser("query", help="İndekste benzer vaka ara")
    query_parser.add_argument("--index", default=str(DEFAULT_INDEX), help="İndeks JSON yolu")
    query_parser.add_argument("--text", required=True, help="Sorgu metni")
    query_parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding modeli")
    query_parser.add_argument("--top-k", type=int, default=5, help="Döndürülecek sonuç sayısı")

    args = parser.parse_args()

    try:
        if args.command == "build":
            result = build_index(
                csv_path=Path(args.csv),
                output_path=Path(args.output),
                model=args.model,
                limit=args.limit,
                batch_size=args.batch_size,
            )
        else:
            result = query_index(
                index_path=Path(args.index),
                text=args.text,
                model=args.model,
                top_k=args.top_k,
            )
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Ollama HTTP hatası: {exc.code} {details}") from exc
    except error.URLError as exc:
        raise SystemExit(f"Ollama bağlantı hatası: {exc}") from exc

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
