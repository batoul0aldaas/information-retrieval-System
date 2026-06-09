"""
Hybrid Retrieval
Combines multiple retrieval models in two modes:
  - Serial:   Use first model to filter candidates, re-rank with second model.
  - Parallel: Run all models simultaneously, fuse scores (RRF or Linear).
"""

from typing import Dict, List, Tuple
import numpy as np

from services.indexing_service.inverted_index import InvertedIndex
from services.retrieval_service.bm25_retrieval import retrieve_bm25
from services.retrieval_service.tfidf_retrieval import retrieve_tfidf
from services.retrieval_service.embedding_retrieval import retrieve_embedding


# ─── Fusion Methods ───────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    results_list: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) — combines multiple ranked lists.
    Score = sum of 1 / (k + rank) across all lists.
    """
    rrf_scores: Dict[str, float] = {}
    for results in results_list:
        for rank, (doc_id, _) in enumerate(results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def linear_combination_fusion(
    results_list: List[List[Tuple[str, float]]],
    weights: List[float] = None
) -> List[Tuple[str, float]]:
    """
    Linear weighted combination of scores from multiple models.
    Scores are normalized to [0, 1] before combining.
    """
    if weights is None:
        weights = [1.0 / len(results_list)] * len(results_list)

    combined: Dict[str, float] = {}
    for results, weight in zip(results_list, weights):
        if not results:
            continue
        max_score = max(s for _, s in results) or 1
        for doc_id, score in results:
            normalized = score / max_score
            combined[doc_id] = combined.get(doc_id, 0) + weight * normalized

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


# ─── Serial Hybrid ────────────────────────────────────────────────────────────

def retrieve_hybrid_serial(
    query: str,
    index: InvertedIndex,
    embeddings: np.ndarray,
    doc_ids: List[str],
    first_stage_top_k: int = 100,
    final_top_k: int = 10,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> List[Tuple[str, float]]:
    """
    Serial Hybrid:
    Step 1 — BM25 retrieves top candidates.
    Step 2 — Embedding model re-ranks those candidates.
    """
    candidates = retrieve_bm25(query, index, top_k=first_stage_top_k, k1=bm25_k1, b=bm25_b)
    candidate_ids = {doc_id for doc_id, _ in candidates}

    filtered_ids = [d for d in doc_ids if d in candidate_ids]
    filtered_idx = [doc_ids.index(d) for d in filtered_ids]
    filtered_embeddings = embeddings[filtered_idx]

    reranked = retrieve_embedding(query, filtered_embeddings, filtered_ids, embedding_model, top_k=final_top_k)
    return reranked


# ─── Parallel Hybrid ─────────────────────────────────────────────────────────

def retrieve_hybrid_parallel(
    query: str,
    index: InvertedIndex,
    embeddings: np.ndarray,
    doc_ids: List[str],
    top_k: int = 10,
    fusion_method: str = "rrf",
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    embedding_model: str = "all-MiniLM-L6-v2",
    weights: List[float] = None
) -> List[Tuple[str, float]]:
    """
    Parallel Hybrid:
    Runs BM25 + TF-IDF + Embedding simultaneously,
    then fuses results using RRF or Linear Combination.

    fusion_method: 'rrf' or 'linear'
    """
    bm25_results = retrieve_bm25(query, index, top_k=top_k * 2, k1=bm25_k1, b=bm25_b)
    tfidf_results = retrieve_tfidf(query, index, top_k=top_k * 2)
    embed_results = retrieve_embedding(query, embeddings, doc_ids, embedding_model, top_k=top_k * 2)

    all_results = [bm25_results, tfidf_results, embed_results]

    if fusion_method == "rrf":
        fused = reciprocal_rank_fusion(all_results)
    else:
        fused = linear_combination_fusion(all_results, weights=weights)

    return fused[:top_k]
