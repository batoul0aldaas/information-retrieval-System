import json
from pathlib import Path

from services.retrieval_service.embedding_retrieval import (
    build_embeddings,
    save_embeddings,
)

DATASETS = ["dataset1", "dataset2"]


def load_documents(path: Path):
    documents = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            documents[item["doc_id"]] = item["text"]

    return documents


def process_dataset(dataset_name: str):
    print(f"\n{'=' * 50}")
    print(f"Processing {dataset_name}")
    print(f"{'=' * 50}")

    dataset_dir = Path("data") / dataset_name
    documents_path = dataset_dir / "documents.jsonl"
    embeddings_path = dataset_dir / "embeddings.pkl"

    if embeddings_path.exists():
        print(f"Embeddings already exist: {embeddings_path}")
        return

    print("Loading documents...")
    documents = load_documents(documents_path)

    print(f"Loaded {len(documents):,} documents")

    embeddings, doc_ids = build_embeddings(
        documents,
        model_name="all-MiniLM-L6-v2",
        batch_size=64
    )

    save_embeddings(
        embeddings,
        doc_ids,
        str(embeddings_path)
    )


if __name__ == "__main__":
    for dataset in DATASETS:
        process_dataset(dataset)

    print("\nDone.")