"""
Embedding-based Retrieval
Uses sentence-transformers (BERT-based) to encode documents and queries,
then retrieves via cosine similarity.
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple
from tqdm import tqdm


def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load a sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def build_embeddings(
    documents: Dict[str, str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode all documents into embedding vectors.
    Returns (embeddings_matrix, doc_ids_list).
    """
    model = load_model(model_name)
    doc_ids = list(documents.keys())
    texts = list(documents.values())

    print(f"Encoding {len(texts)} documents...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings), doc_ids


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def retrieve_embedding(
    query: str,
    embeddings: np.ndarray,
    doc_ids: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k documents using embedding cosine similarity.
    Returns list of (doc_id, score) sorted descending.
    """
    model = load_model(model_name)
    query_vec = model.encode([query])[0]

    scores = []
    for i, doc_id in enumerate(doc_ids):
        score = cosine_similarity(query_vec, embeddings[i])
        scores.append((doc_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def save_embeddings(embeddings: np.ndarray, doc_ids: List[str], path: str) -> None:
    """Save embeddings to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "doc_ids": doc_ids}, f)
    print(f"Embeddings saved to {path}")


def load_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings from disk."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["doc_ids"]
