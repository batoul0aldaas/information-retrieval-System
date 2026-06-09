"""
BM25 Retrieval
Probabilistic ranking function with tunable parameters k1 and b.
"""

import math
from typing import Dict, List, Tuple
from services.indexing_service.inverted_index import InvertedIndex
from services.preprocessing_service.preprocessor import preprocess


def bm25_score(
    term_freq: int,
    doc_freq: int,
    doc_length: int,
    avg_doc_length: float,
    doc_count: int,
    k1: float = 1.5,
    b: float = 0.75
) -> float:
    """
    Compute BM25 score for a single term in a document.

    Parameters:
        k1: Controls term frequency saturation (typical: 1.2 - 2.0)
        b:  Controls document length normalization (typical: 0.75)
    """
    idf = math.log((doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    tf_norm = (term_freq * (k1 + 1)) / (
        term_freq + k1 * (1 - b + b * (doc_length / avg_doc_length))
    )
    return idf * tf_norm


def retrieve_bm25(
    query: str,
    index: InvertedIndex,
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k documents using BM25.
    Returns list of (doc_id, score) sorted descending.
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
                k1=k1, b=b
            )
            scores[doc_id] = scores.get(doc_id, 0) + score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
