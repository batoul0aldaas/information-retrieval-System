"""
Index Registry
Loads and caches indexes and embeddings per dataset (lazy loading).
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Set
from services.indexing_service.inverted_index import InvertedIndex

_indexes = {}
_embeddings_cache = {}

INDEX_PATHS = {
    "dataset1": "data/dataset1/index.pkl",
    "dataset2": "data/dataset2/index.pkl",
}

EMBEDDING_PATHS = {
    "dataset1": "data/dataset1/embeddings.pkl",
    "dataset2": "data/dataset2/embeddings.pkl",
}

EMPTY_DOCS_PATHS = {
    "dataset1": "data/empty_docs_dataset1.txt",
    "dataset2": "data/empty_docs_dataset2.txt",
}


def get_index(dataset: str) -> InvertedIndex:
    """Load and cache inverted index for a dataset."""
    if dataset not in _indexes:
        path = INDEX_PATHS.get(dataset)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Index not found for dataset '{dataset}'. Run indexing first.")
        idx = InvertedIndex()
        idx.load(path)
        _indexes[dataset] = idx
    return _indexes[dataset]


def get_empty_doc_ids(dataset: str) -> Set[str]:
    """Load the set of doc_ids that had zero tokens after preprocessing."""
    path = EMPTY_DOCS_PATHS.get(dataset, "")
    if not path or not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def get_embeddings(dataset: str) -> Tuple[np.ndarray, List[str]]:
    """Load and cache embeddings for a dataset."""
    if dataset not in _embeddings_cache:
        path = EMBEDDING_PATHS.get(dataset)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Embeddings not found for dataset '{dataset}'. Run embedding first.")
        from services.retrieval_service.embedding_retrieval import load_embeddings
        embeddings, doc_ids = load_embeddings(path)
        _embeddings_cache[dataset] = (embeddings, doc_ids)
    return _embeddings_cache[dataset]
