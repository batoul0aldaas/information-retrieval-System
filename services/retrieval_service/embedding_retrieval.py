"""
Embedding-based Retrieval

Uses sentence-transformers (BERT-based) to encode documents and queries,
then retrieves via cosine similarity.
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

_MODEL_CACHE = {}


def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load and cache the sentence-transformer model.
    """
    from sentence_transformers import SentenceTransformer

    if model_name not in _MODEL_CACHE:
        print(f"Loading embedding model: {model_name}")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)

    return _MODEL_CACHE[model_name]


def build_embeddings(
    documents: Dict[str, str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode all documents into normalized embedding vectors.

    Returns:
        (embeddings_matrix, doc_ids_list)
    """
    model = load_model(model_name)

    doc_ids = list(documents.keys())
    texts = list(documents.values())

    print(f"Encoding {len(texts)} documents...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return np.array(embeddings), doc_ids


def retrieve_embedding(
    query: str,
    embeddings: np.ndarray,
    doc_ids: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k documents using cosine similarity.

    Assumes document embeddings are already normalized.
    """
    model = load_model(model_name)

    query_vec = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(embeddings, query_vec)

    top_indices = np.argsort(scores)[-top_k:][::-1]

    return [
        (doc_ids[i], float(scores[i]))
        for i in top_indices
    ]


def save_embeddings(
    embeddings: np.ndarray,
    doc_ids: List[str],
    path: str
) -> None:
    """
    Save embeddings to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,
                "doc_ids": doc_ids
            },
            f
        )

    print(f"Embeddings saved to {path}")


def load_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings from disk.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["embeddings"], data["doc_ids"]