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
from services.fusion_service.fusion_service import (
    reciprocal_rank_fusion,
    linear_combination_fusion
)

# ✅ FIX: استخدام FAISS version بدل غير موجود
from services.retrieval_service.embedding_retrieval import retrieve_embedding_faiss


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
    embedding_model: str = "all-MiniLM-L6-v2",
    dataset: str = None,
) -> List[Tuple[str, float]]:

    """
    Serial Hybrid:
    Step 1 — BM25 retrieves top candidates.
    Step 2 — Embedding model re-ranks those candidates.
    """
    candidates = retrieve_bm25(
        query, index, top_k=first_stage_top_k,
        k1=bm25_k1, b=bm25_b, dataset=dataset,
    )

    candidate_ids = {doc_id for doc_id, _ in candidates}

    # Step 2: Embedding retrieval (FAISS-based)
   
    embed_results = retrieve_embedding_faiss(
        query,
        dataset_id=dataset if dataset else "dataset1",
        top_k=first_stage_top_k
    )

    # filter by BM25 candidates
    reranked = [
        (doc_id, score)
        for doc_id, score in embed_results
        if doc_id in candidate_ids
    ]

    return reranked[:final_top_k]


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
    weights: List[float] = None,
    dataset: str = None,
) -> List[Tuple[str, float]]:

    bm25_results = retrieve_bm25(
        query, index, top_k=top_k * 2,
        k1=bm25_k1, b=bm25_b ,  dataset=dataset,
    )

    tfidf_results = retrieve_tfidf(
        query, index, top_k=top_k * 2
        ,  dataset=dataset,
    )

    # FIX: FAISS embedding retrieval
    embed_results = retrieve_embedding_faiss(
        query,
        dataset_id=dataset if dataset else "dataset1",
        top_k=top_k * 2
    )


    all_results = [bm25_results, tfidf_results, embed_results]

    if fusion_method == "rrf":
        fused = reciprocal_rank_fusion(all_results)
    else:
        fused = linear_combination_fusion(all_results, weights=weights)

    return fused[:top_k]