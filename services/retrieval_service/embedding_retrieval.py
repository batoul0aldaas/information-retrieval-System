"""
Embedding-based Retrieval
Uses sentence-transformers + FAISS vector index for semantic search.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Tuple

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_MODEL_CACHE: SentenceTransformer | None = None
_FAISS_CACHE: dict[str, Any] = {}
_METADATA_CACHE: dict[str, dict[str, Any]] = {}


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    global _MODEL_CACHE

    if _MODEL_CACHE is None:
        print(f"Loading embedding model: {model_name}")
        _MODEL_CACHE = SentenceTransformer(model_name)

    return _MODEL_CACHE


def _load_faiss_dataset(dataset_id: str) -> tuple[Any, dict[str, Any]]:
    dataset_dir = DATA_DIR / dataset_id

    faiss_path = dataset_dir / "faiss.index"
    metadata_path = dataset_dir / "embedding_metadata.pkl"

    if not faiss_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {faiss_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing embedding metadata: {metadata_path}")

    if dataset_id not in _FAISS_CACHE:
        _FAISS_CACHE[dataset_id] = faiss.read_index(str(faiss_path))

    if dataset_id not in _METADATA_CACHE:
        with metadata_path.open("rb") as file:
            _METADATA_CACHE[dataset_id] = pickle.load(file)

    return _FAISS_CACHE[dataset_id], _METADATA_CACHE[dataset_id]


def retrieve_embedding_faiss(
    query: str,
    dataset_id: str,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    index, metadata = _load_faiss_dataset(dataset_id)

    doc_ids = metadata["doc_ids"]

    model = load_model()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results: List[Tuple[str, float]] = []

    for score, doc_index in zip(scores[0], indices[0]):
        if doc_index < 0:
            continue

        results.append((str(doc_ids[doc_index]), float(score)))

    return results


def load_embeddings(path: str):
    with open(path, "rb") as file:
        data = pickle.load(file)

    return data["embeddings"], data["doc_ids"]


def retrieve_embedding(
    query: str,
    embeddings=None,
    doc_ids=None,
    model_name: str = MODEL_NAME,
    top_k: int = 10,
    dataset_id: str = "dataset1",
):
    return retrieve_embedding_faiss(
        query=query,
        dataset_id=dataset_id,
        top_k=top_k,
    )