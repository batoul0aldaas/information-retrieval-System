"""
TF-IDF Retrieval (Vector Space Model)
Uses cosine similarity between query and document TF-IDF vectors.
"""

import math
from typing import Dict, List, Tuple
from services.indexing_service.inverted_index import InvertedIndex
from services.preprocessing_service.preprocessor import preprocess


def compute_tfidf(term_freq: int, doc_freq: int, doc_count: int) -> float:
    """Compute TF-IDF score for a term in a document."""
    tf = 1 + math.log(term_freq) if term_freq > 0 else 0
    idf = math.log(doc_count / (1 + doc_freq)) if doc_freq > 0 else 0
    return tf * idf


def retrieve_tfidf(
    query: str,
    index: InvertedIndex,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k documents for a query using TF-IDF + Cosine Similarity.
    Returns list of (doc_id, score) sorted descending.
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
