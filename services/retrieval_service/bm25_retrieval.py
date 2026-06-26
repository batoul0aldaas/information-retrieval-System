"""
BM25 Retrieval

Primary path: rank_bm25 BM25Okapi (library-based).
Legacy path: manual BM25 over InvertedIndex (kept for reference/fallback).
"""

import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from services.indexing_service.inverted_index import InvertedIndex
from services.preprocessing_service.preprocessor import preprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BM25_MODEL_PATHS = {
    "dataset1": PROJECT_ROOT / "data" / "dataset1" / "bm25_model.pkl",
    "dataset2": PROJECT_ROOT / "data" / "dataset2" / "bm25_model.pkl",
}

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
BUILD_CMD = "python -m services.retrieval_service.build_lexical_models"

_BM25_CACHE: Dict[str, dict] = {}


# ─── Model loading (cached) ───────────────────────────────────────────────────

def load_bm25_model(dataset: str) -> dict:
    """
    Load and cache BM25Okapi artifacts for a dataset.

    Returns dict with keys: bm25, doc_ids, corpus_tokens, k1, b
    """
    if dataset in _BM25_CACHE:
        return _BM25_CACHE[dataset]

    path = BM25_MODEL_PATHS.get(dataset)
    if not path or not path.exists():
        raise FileNotFoundError(
            f"BM25 model not found for '{dataset}' at {path}.\n"
            f"Run: {BUILD_CMD}"
        )

    with path.open("rb") as f:
        data = pickle.load(f)

    for key in ("bm25", "doc_ids", "corpus_tokens"):
        if key not in data:
            raise ValueError(f"Invalid BM25 model file (missing '{key}'): {path}")

    _BM25_CACHE[dataset] = data
    return data


def clear_bm25_cache(dataset: Optional[str] = None) -> None:
    """Clear in-memory BM25 cache (useful after rebuilding models)."""
    if dataset is None:
        _BM25_CACHE.clear()
    else:
        _BM25_CACHE.pop(dataset, None)


def _get_bm25_scorer(model_data: dict, k1: float, b: float) -> BM25Okapi:
    """
    Return BM25Okapi scorer for given k1/b.

    Uses cached default model when k1/b match saved defaults; otherwise
    rebuilds BM25Okapi in memory from corpus_tokens (no disk reload).
    """
    saved_k1 = model_data.get("k1", DEFAULT_K1)
    saved_b = model_data.get("b", DEFAULT_B)

    if abs(k1 - saved_k1) < 1e-9 and abs(b - saved_b) < 1e-9:
        return model_data["bm25"]

    return BM25Okapi(model_data["corpus_tokens"], k1=k1, b=b)


# ─── Library-based retrieval (primary) ───────────────────────────────────────

def retrieve_bm25_library(
    query: str,
    dataset: str,
    top_k: int = 10,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k documents using rank_bm25 BM25Okapi.get_scores().
    """
    model_data = load_bm25_model(dataset)
    doc_ids = model_data["doc_ids"]

    query_tokens = preprocess(query)
    if not query_tokens:
        return []

    bm25 = _get_bm25_scorer(model_data, k1=k1, b=b)
    scores = bm25.get_scores(query_tokens)

    if len(scores) == 0:
        return []

    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [
        (doc_ids[i], float(scores[i]))
        for i in top_indices
        if scores[i] > 0
    ]


# ─── Legacy manual implementation ──────────────────────────────────────────────

def bm25_score(
    term_freq: int,
    doc_freq: int,
    doc_length: int,
    avg_doc_length: float,
    doc_count: int,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> float:
    """Legacy manual BM25 score for a single term in a document."""
    idf = math.log((doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    tf_norm = (term_freq * (k1 + 1)) / (
        term_freq + k1 * (1 - b + b * (doc_length / avg_doc_length))
    )
    return idf * tf_norm


def retrieve_bm25_manual(
    query: str,
    index: InvertedIndex,
    top_k: int = 10,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> List[Tuple[str, float]]:
    """
    Legacy: manual BM25 scoring over inverted index postings.
    """
    query_tokens = preprocess(query)
    scores: Dict[str, float] = {}

    for term in set(query_tokens):
        postings = index.get_postings(term)
        doc_freq = len(postings)
        if doc_freq == 0:
            continue

        for doc_id, term_freq in postings.items():
            doc_length = index.get_doc_length(doc_id)
            score = bm25_score(
                term_freq, doc_freq, doc_length,
                index.avg_doc_length, index.doc_count,
                k1=k1, b=b,
            )
            scores[doc_id] = scores.get(doc_id, 0) + score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ─── Public API (backward compatible) ─────────────────────────────────────────

def retrieve_bm25(
    query: str,
    index: InvertedIndex = None,
    top_k: int = 10,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    dataset: Optional[str] = None,
    use_library: bool = True,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k documents using BM25.

    Primary: rank_bm25 BM25Okapi (requires `dataset` + pre-built model).
    Legacy:  manual implementation over `index` when use_library=False.

    Returns list of (doc_id, score) sorted descending.
    """
    if use_library:
        if not dataset:
            raise ValueError(
                "dataset is required for library-based BM25. "
                f"Build models first: {BUILD_CMD}"
            )
        return retrieve_bm25_library(query, dataset, top_k=top_k, k1=k1, b=b)

    if index is None:
        raise ValueError("index is required for manual BM25 retrieval")
    return retrieve_bm25_manual(query, index, top_k=top_k, k1=k1, b=b)
