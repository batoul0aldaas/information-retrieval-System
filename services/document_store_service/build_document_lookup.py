from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DATASETS = ["dataset1", "dataset2"]


def get_doc_id(doc: dict[str, Any], index: int) -> str:
    for key in ["doc_id", "id", "_id", "document_id"]:
        value = doc.get(key)
        if value is not None:
            return str(value)

    return str(index)


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


def build_lookup(dataset: str) -> None:
    documents_path = DATA_DIR / dataset / "documents.jsonl"
    output_path = DATA_DIR / dataset / "document_lookup.pkl"

    if not documents_path.exists():
        raise FileNotFoundError(f"File not found: {documents_path}")

    lookup: dict[str, str] = {}

    with documents_path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if not line.strip():
                continue

            doc = json.loads(line)
            doc_id = get_doc_id(doc, index)
            text = get_doc_text(doc)

            if text:
                lookup[doc_id] = text

    with output_path.open("wb") as file:
        pickle.dump(lookup, file)

    print(f"Saved {dataset}: {output_path}")
    print(f"Documents: {len(lookup):,}")


def main() -> None:
    for dataset in DATASETS:
        build_lookup(dataset)


if __name__ == "__main__":
    main()