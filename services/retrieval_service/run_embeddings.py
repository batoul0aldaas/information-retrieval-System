from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

SAMPLE_SIZE = 10000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


DATASETS = {
    "dataset1": {
        "documents_path": DATA_DIR / "dataset1" / "documents.jsonl",
        "output_dir": DATA_DIR / "dataset1",
    },
    "dataset2": {
        "documents_path": DATA_DIR / "dataset2" / "documents.jsonl",
        "output_dir": DATA_DIR / "dataset2",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def get_doc_id(doc: dict[str, Any], index: int) -> str:
    for key in ["doc_id", "id", "_id", "document_id"]:
        value = doc.get(key)
        if value is not None:
            return str(value)

    return f"DOC_{index}"


def get_doc_text(doc: dict[str, Any]) -> str:
    title = doc.get("title", "")
    text = ""

    for key in ["text", "contents", "content", "body", "passage"]:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break

    if isinstance(title, str) and title.strip():
        return f"{title.strip()} {text}".strip()

    return text


def build_dataset_embeddings(
    dataset_name: str,
    documents_path: Path,
    output_dir: Path,
    model: SentenceTransformer,
) -> None:
    print("=" * 80)
    print(f"Building embeddings for {dataset_name}")
    print(f"Input file: {documents_path}")

    documents = read_jsonl(documents_path)
    documents = documents[:SAMPLE_SIZE]
    
    doc_ids: list[str] = []
    texts: list[str] = []

    for index, doc in enumerate(documents):
        text = get_doc_text(doc)

        if not text:
            continue

        doc_ids.append(get_doc_id(doc, index))
        texts.append(text)

    print(f"Valid documents: {len(texts):,}")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.pkl"
    metadata_path = output_dir / "embedding_metadata.pkl"
    faiss_path = output_dir / "faiss.index"

    with embeddings_path.open("wb") as file:
        pickle.dump(
            {
                "model_name": MODEL_NAME,
                "doc_ids": doc_ids,
                "texts": texts,
                "embeddings": embeddings,
            },
            file,
        )

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, str(faiss_path))

    with metadata_path.open("wb") as file:
        pickle.dump(
            {
                "model_name": MODEL_NAME,
                "doc_ids": doc_ids,
                "texts": texts,
                "embedding_dim": dimension,
                "documents_count": len(doc_ids),
            },
            file,
        )

    print(f"Saved: {embeddings_path}")
    print(f"Saved: {metadata_path}")
    print(f"Saved: {faiss_path}")
    print(f"Embedding shape: {embeddings.shape}")


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    for dataset_name, config in DATASETS.items():
        build_dataset_embeddings(
            dataset_name=dataset_name,
            documents_path=config["documents_path"],
            output_dir=config["output_dir"],
            model=model,
        )

    print("=" * 80)
    print("Done. Embeddings and FAISS indexes were generated successfully.")


if __name__ == "__main__":
    main()
