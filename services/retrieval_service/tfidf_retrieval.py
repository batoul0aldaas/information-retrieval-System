"""
TF-IDF Retrieval (Vector Space Model)

Primary path: sklearn TfidfVectorizer + cosine_similarity (library-based).
Legacy path: manual TF-IDF over InvertedIndex (kept for reference/fallback).
"""

import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from services.indexing_service.inverted_index import InvertedIndex
from services.preprocessing_service.preprocessor import preprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TFIDF_MODEL_PATHS = {
    "dataset1": PROJECT_ROOT / "data" / "dataset1" / "tfidf_model.pkl",
    "dataset2": PROJECT_ROOT / "data" / "dataset2" / "tfidf_model.pkl",
}

BUILD_CMD = "python -m services.retrieval_service.build_lexical_models"

_TFIDF_CACHE: Dict[str, dict] = {}


# ─── Model loading (cached) ───────────────────────────────────────────────────

def load_tfidf_model(dataset: str) -> dict:
    """
    Load and cache sklearn TF-IDF artifacts for a dataset.

    Returns dict with keys: vectorizer, matrix, doc_ids
    """
    if dataset in _TFIDF_CACHE:
        return _TFIDF_CACHE[dataset]

    path = TFIDF_MODEL_PATHS.get(dataset)
    if not path or not path.exists():
        raise FileNotFoundError(
            f"TF-IDF model not found for '{dataset}' at {path}.\n"
            f"Run: {BUILD_CMD}"
        )

    with path.open("rb") as f:
        data = pickle.load(f)

    for key in ("vectorizer", "matrix", "doc_ids"):
        if key not in data:
            raise ValueError(f"Invalid TF-IDF model file (missing '{key}'): {path}")

    _TFIDF_CACHE[dataset] = data
    return data


def clear_tfidf_cache(dataset: Optional[str] = None) -> None:
    """Clear in-memory TF-IDF cache (useful after rebuilding models)."""
    if dataset is None:
        _TFIDF_CACHE.clear()
    else:
        _TFIDF_CACHE.pop(dataset, None)


# ─── Library-based retrieval (primary) ───────────────────────────────────────

def retrieve_tfidf_library(
    query: str,
    dataset: str,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k documents using sklearn TfidfVectorizer + cosine similarity.
    """
    model = load_tfidf_model(dataset)
    vectorizer = model["vectorizer"]
    matrix = model["matrix"]
    doc_ids = model["doc_ids"]

    query_tokens = preprocess(query)
    if not query_tokens:
        return []

    query_text = " ".join(query_tokens)
    query_vec = vectorizer.transform([query_text])
    scores = cosine_similarity(query_vec, matrix).flatten()

    if not np.any(scores):
        return []

    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [
        (doc_ids[i], float(scores[i]))
        for i in top_indices
        if scores[i] > 0
    ]


# ─── Legacy manual implementation ──────────────────────────────────────────────

def compute_tfidf(term_freq: int, doc_freq: int, doc_count: int) -> float:
    """Legacy manual TF-IDF weight (log TF × log IDF)."""
    tf = 1 + math.log(term_freq) if term_freq > 0 else 0
    idf = math.log(doc_count / (1 + doc_freq)) if doc_freq > 0 else 0
    return tf * idf


def retrieve_tfidf_manual(
    query: str,
    index: InvertedIndex,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Legacy: manual TF-IDF dot product over inverted index postings.
    """
    query_tokens = preprocess(query)
    scores: Dict[str, float] = {}

    query_tf: Dict[str, int] = {}
    for token in query_tokens:
        query_tf[token] = query_tf.get(token, 0) + 1

    for term, q_freq in query_tf.items():
        postings = index.get_postings(term)
        doc_freq = len(postings)
        if doc_freq == 0:
            continue

        q_tfidf = compute_tfidf(q_freq, doc_freq, index.doc_count)

        for doc_id, d_freq in postings.items():
            d_tfidf = compute_tfidf(d_freq, doc_freq, index.doc_count)
            scores[doc_id] = scores.get(doc_id, 0) + q_tfidf * d_tfidf

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ─── Public API (backward compatible) ─────────────────────────────────────────

def retrieve_tfidf(
    query: str,
    index: InvertedIndex = None,
    top_k: int = 10,
    dataset: Optional[str] = None,
    use_library: bool = True,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k documents using TF-IDF.

    Primary: sklearn TfidfVectorizer (requires `dataset` + pre-built model).
    Legacy:  manual implementation over `index` when use_library=False.

    Returns list of (doc_id, score) sorted descending.
    """
    if use_library:
        if not dataset:
            raise ValueError(
                "dataset is required for library-based TF-IDF. "
                f"Build models first: {BUILD_CMD}"
            )
        return retrieve_tfidf_library(query, dataset, top_k=top_k)

    if index is None:
        raise ValueError("index is required for manual TF-IDF retrieval")
    return retrieve_tfidf_manual(query, index, top_k=top_k)
