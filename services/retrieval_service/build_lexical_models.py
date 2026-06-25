"""
Build library-based lexical models (TF-IDF + BM25) for each dataset.

Reads pre-tokenized documents from processed_datasetX.jsonl and saves:
  data/datasetX/tfidf_model.pkl  — sklearn TfidfVectorizer + matrix
  data/datasetX/bm25_model.pkl     — rank_bm25 BM25Okapi + corpus_tokens

Usage:
    python -m services.retrieval_service.build_lexical_models
    python -m services.retrieval_service.build_lexical_models --dataset dataset1
    python -m services.retrieval_service.build_lexical_models --force
    python -m services.retrieval_service.build_lexical_models --sample 5000
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

DATASET_CONFIG = {
    "dataset1": {
        "processed": DATA_DIR / "processed_dataset1.jsonl",
        "out_dir": DATA_DIR / "dataset1",
    },
    "dataset2": {
        "processed": DATA_DIR / "processed_dataset2.jsonl",
        "out_dir": DATA_DIR / "dataset2",
    },
}


def load_corpus(processed_path: Path, sample: int = 0):
    """Read processed JSONL → doc_ids, token lists, space-joined texts."""
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {processed_path}\n"
            "Run: python -m services.preprocessing_service.run_preprocessing"
        )

    doc_ids = []
    corpus_tokens = []
    corpus_texts = []

    with processed_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(tqdm(f, desc="Loading tokens", unit="doc"), 1):
            if sample and line_no > sample:
                break

            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            doc_id = str(item.get("doc_id", "")).strip()
            tokens = item.get("tokens") or []

            if not doc_id or not tokens:
                continue

            doc_ids.append(doc_id)
            corpus_tokens.append(tokens)
            corpus_texts.append(" ".join(tokens))

    return doc_ids, corpus_tokens, corpus_texts


def build_tfidf_model(doc_ids, corpus_texts, output_path: Path) -> None:
    """Fit sklearn TfidfVectorizer and persist vectorizer + sparse matrix."""
    print(f"  Building TF-IDF (sklearn TfidfVectorizer) for {len(doc_ids):,} docs...")
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus_texts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "matrix": matrix,
                "doc_ids": doc_ids,
            },
            f,
        )
    print(f"  Saved TF-IDF model -> {output_path}")


def build_bm25_model(
    doc_ids,
    corpus_tokens,
    output_path: Path,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> None:
    """Fit rank_bm25 BM25Okapi and persist model + corpus for k1/b overrides."""
    print(f"  Building BM25 (rank_bm25 BM25Okapi) for {len(doc_ids):,} docs...")
    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "doc_ids": doc_ids,
                "corpus_tokens": corpus_tokens,
                "k1": k1,
                "b": b,
            },
            f,
        )
    print(f"  Saved BM25 model -> {output_path}")


def build_for_dataset(dataset_name: str, force: bool = False, sample: int = 0) -> None:
    cfg = DATASET_CONFIG[dataset_name]
    processed_path = cfg["processed"]
    out_dir = cfg["out_dir"]
    tfidf_path = out_dir / "tfidf_model.pkl"
    bm25_path = out_dir / "bm25_model.pkl"

    print("\n" + "=" * 70)
    print(f"Building lexical models for {dataset_name}")
    print("=" * 70)

    if not force and tfidf_path.exists() and bm25_path.exists():
        print(f"  Models already exist (use --force to rebuild):")
        print(f"    {tfidf_path}")
        print(f"    {bm25_path}")
        return

    t0 = time.time()
    doc_ids, corpus_tokens, corpus_texts = load_corpus(processed_path, sample=sample)
    print(f"  Loaded {len(doc_ids):,} documents")

    if not doc_ids:
        raise ValueError(f"No documents loaded for {dataset_name}")

    build_tfidf_model(doc_ids, corpus_texts, tfidf_path)
    build_bm25_model(doc_ids, corpus_tokens, bm25_path)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Build sklearn TF-IDF and rank_bm25 BM25 models per dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=["dataset1", "dataset2", "all"],
        default="all",
        help="Which dataset to build (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if model files already exist",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Use only first N documents (0 = all). Useful for quick testing.",
    )
    args = parser.parse_args()

    keys = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    print("IR System — Build Lexical Models (sklearn + rank_bm25)")
    print(f"  Datasets : {', '.join(keys)}")
    print(f"  Force    : {args.force}")
    if args.sample:
        print(f"  Sample   : first {args.sample:,} documents")

    for key in keys:
        build_for_dataset(key, force=args.force, sample=args.sample)

    print("\n" + "=" * 70)
    print("SUMMARY — expected outputs:")
    for key in keys:
        out = DATA_DIR / key
        for name in ("tfidf_model.pkl", "bm25_model.pkl"):
            path = out / name
            status = "OK" if path.exists() else "MISSING"
            print(f"  [{status}] {path}")
    print("\nNext: start API/UI or run retrieval tests.")
    print("=" * 70)


if __name__ == "__main__":
    main()
