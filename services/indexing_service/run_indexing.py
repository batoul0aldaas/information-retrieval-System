import json
from pathlib import Path

from services.indexing_service.inverted_index import InvertedIndex


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_empty_doc_ids(path: Path) -> set:
    """
    Reads empty_docs_datasetX.txt if it exists.
    Returns a set of doc_ids that should be skipped.
    """
    if not path.exists():
        print(f"[INFO] Empty docs file not found: {path}")
        return set()

    empty_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            doc_id = line.strip()
            if doc_id:
                empty_ids.add(doc_id)

    print(f"[INFO] Loaded {len(empty_ids):,} empty doc ids from {path.name}")
    return empty_ids


def load_processed_tokens(processed_path: Path, skip_ids: set) -> dict:
    """
    Reads processed_datasetX.jsonl.

    Returns:
    {
        "doc_id": ["token1", "token2", ...]
    }

    This matches InvertedIndex.build_from_tokens().
    """
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    token_data = {}
    total = 0
    skipped_empty = 0
    skipped_no_tokens = 0

    with processed_path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)
            doc_id = str(item.get("doc_id", "")).strip()
            tokens = item.get("tokens", [])

            if not doc_id:
                skipped_no_tokens += 1
                continue

            if doc_id in skip_ids:
                skipped_empty += 1
                continue

            if not tokens:
                skipped_no_tokens += 1
                continue

            token_data[doc_id] = tokens

    print(f"[INFO] Read {total:,} lines from {processed_path.name}")
    print(f"[INFO] Kept {len(token_data):,} documents")
    print(f"[INFO] Skipped empty docs: {skipped_empty:,}")
    print(f"[INFO] Skipped docs without tokens: {skipped_no_tokens:,}")

    return token_data


def build_index_for_dataset(dataset_name: str):
    """
    Builds and saves inverted index for one dataset.
    dataset_name should be: dataset1 or dataset2
    """
    print("\n" + "=" * 70)
    print(f"Building index for {dataset_name}")
    print("=" * 70)

    if dataset_name == "dataset1":
        processed_path = DATA_DIR / "processed_dataset1.jsonl"
        empty_docs_path = DATA_DIR / "empty_docs_dataset1.txt"
        output_path = DATA_DIR / "dataset1" / "index.pkl"
    elif dataset_name == "dataset2":
        processed_path = DATA_DIR / "processed_dataset2.jsonl"
        empty_docs_path = DATA_DIR / "empty_docs_dataset2.txt"
        output_path = DATA_DIR / "dataset2" / "index.pkl"
    else:
        raise ValueError("dataset_name must be 'dataset1' or 'dataset2'")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    skip_ids = load_empty_doc_ids(empty_docs_path)
    token_data = load_processed_tokens(processed_path, skip_ids)

    print("[INFO] Creating InvertedIndex...")
    index = InvertedIndex()

    print("[INFO] Building index from tokens...")
    index.build_from_tokens(token_data)

    print("[INFO] Saving index...")
    index.save(output_path)

    print(f"[DONE] Saved index to: {output_path}")

    if hasattr(index, "doc_count"):
        print(f"[INFO] doc_count: {index.doc_count:,}")

    if hasattr(index, "avg_doc_length"):
        print(f"[INFO] avg_doc_length: {index.avg_doc_length:.2f}")

    if hasattr(index, "index"):
        print(f"[INFO] vocabulary size: {len(index.index):,}")


def main():
    build_index_for_dataset("dataset1")
    build_index_for_dataset("dataset2")

    print("\n" + "=" * 70)
    print("All indexes were built successfully.")
    print("=" * 70)
    print("Expected outputs:")
    print("data/dataset1/index.pkl")
    print("data/dataset2/index.pkl")


if __name__ == "__main__":
    main()