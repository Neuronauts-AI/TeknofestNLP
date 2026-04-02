import csv
import json
import math
from pathlib import Path
from urllib import error, request


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
DEFAULT_SOURCE_CSV = Path("data/processed/mimic_cxr_text_only_tr_gemini.csv")
DEFAULT_INDEX_PATH = Path("data/processed/semantic_search_index.json")


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


def build_index(
    csv_path: Path = DEFAULT_SOURCE_CSV,
    output_path: Path = DEFAULT_INDEX_PATH,
    model: str = DEFAULT_EMBED_MODEL,
    limit: int = 0,
    batch_size: int = 32,
) -> dict:
    rows = load_rows(csv_path, limit)
    embeddings: list[list[float]] = []
    for batch in batched([row["report_text"] for row in rows], batch_size):
        embeddings.extend(embed_texts(batch, model))

    payload = {
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
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"index_path": str(output_path), "count": len(rows), "model": model}


def ensure_index(index_path: Path = DEFAULT_INDEX_PATH, source_csv: Path = DEFAULT_SOURCE_CSV) -> Path:
    if index_path.exists():
        return index_path
    build_index(csv_path=source_csv, output_path=index_path)
    return index_path


def query_index(index_path: Path, text: str, model: str = DEFAULT_EMBED_MODEL, top_k: int = 5) -> dict:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    query_embedding = embed_texts([text], model)[0]
    scored = []
    for item in payload["items"]:
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
        "available": True,
        "model": model,
        "query": text,
        "top_k": top_k,
        "results": scored[:top_k],
        "error": "",
    }


def semantic_search(text: str, top_k: int = 5, model: str = DEFAULT_EMBED_MODEL) -> dict:
    if not text.strip():
        return {"available": False, "model": model, "query": text, "top_k": top_k, "results": [], "error": "empty_query"}
    try:
        index_path = ensure_index()
        return query_index(index_path=index_path, text=text, model=model, top_k=top_k)
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        return {"available": False, "model": model, "query": text, "top_k": top_k, "results": [], "error": f"http_{exc.code}: {details}"}
    except error.URLError:
        return {"available": False, "model": model, "query": text, "top_k": top_k, "results": [], "error": "connection_failed"}
    except Exception as exc:
        return {"available": False, "model": model, "query": text, "top_k": top_k, "results": [], "error": str(exc)}
