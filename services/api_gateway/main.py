"""
API Gateway — FastAPI
Central entry point that routes requests to all IR services.
"""

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from services.document_store_service.document_store import get_original_text

app = FastAPI(
    title="Information Retrieval System",
    description="Search engine with TF-IDF, BM25, Embeddings, and Hybrid models.",
    version="1.0.0"
)


# ─── Request / Response Models ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    dataset: str = "dataset1"
    model: str = "bm25"           # tfidf | bm25 | embedding | hybrid_serial | hybrid_parallel
    top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_mode: str = "serial"   # serial | parallel
    fusion_method: str = "rrf"    # rrf | linear
    use_spell_correction: bool = True
    use_synonyms: bool = False


class SearchResult(BaseModel):
    doc_id: str
    score: float
    rank: int
    snippet: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    corrected_query: Optional[str]
    model_used: str
    results: List[SearchResult]
    total_results: int


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "IR System is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Main search endpoint.
    Accepts query + model parameters, returns ranked results.
    """
    from services.query_service.query_processor import refine_query

    refined = refine_query(
        request.query,
        use_spell_correction=request.use_spell_correction,
        use_synonyms=request.use_synonyms
    )

    results = _run_retrieval(request, refined["expanded_tokens"])

    return SearchResponse(
        query=request.query,
        corrected_query=refined.get("corrected"),
        model_used=request.model,
        results=results,
        total_results=len(results)
    )


@app.get("/suggest")
def suggest(query: str = Query(...), history: List[str] = Query(default=[])):
    """Return query suggestions based on history."""
    from services.query_service.query_processor import suggest_queries
    return {"suggestions": suggest_queries(query, history)}


@app.get("/datasets")
def list_datasets():
    """List available datasets."""
    return {"datasets": ["dataset1", "dataset2"]}


# ─── Internal Routing ─────────────────────────────────────────────────────────

def _run_retrieval(request: SearchRequest, tokens: List[str]) -> List[SearchResult]:
    """Route to the correct retrieval service based on model parameter."""
    query_text = " ".join(tokens)
    raw_results = []

    if request.model == "tfidf":
        from services.retrieval_service.tfidf_retrieval import retrieve_tfidf
        from services.api_gateway.index_registry import get_index
        index = get_index(request.dataset)
        raw_results = retrieve_tfidf(
            query_text, index, top_k=request.top_k, dataset=request.dataset
        )

    elif request.model == "bm25":
        from services.retrieval_service.bm25_retrieval import retrieve_bm25
        from services.api_gateway.index_registry import get_index
        index = get_index(request.dataset)
        raw_results = retrieve_bm25(
            query_text, index, top_k=request.top_k,
            k1=request.bm25_k1, b=request.bm25_b,
            dataset=request.dataset,
        )

    elif request.model == "embedding":
        from services.retrieval_service.embedding_retrieval import retrieve_embedding_faiss
        raw_results = retrieve_embedding_faiss(
            query_text,
            request.dataset,
            top_k=request.top_k,
        )

    elif request.model in ("hybrid_serial", "hybrid_parallel"):
        from services.api_gateway.index_registry import get_index, get_embeddings
        index = get_index(request.dataset)
        embeddings, doc_ids = get_embeddings(request.dataset)

        if request.model == "hybrid_serial":
            from services.retrieval_service.hybrid_retrieval import retrieve_hybrid_serial
            raw_results = retrieve_hybrid_serial(
                query_text, index, embeddings, doc_ids,
                final_top_k=request.top_k,
                bm25_k1=request.bm25_k1, bm25_b=request.bm25_b,
                dataset=request.dataset,
            )
        else:
            from services.retrieval_service.hybrid_retrieval import retrieve_hybrid_parallel
            raw_results = retrieve_hybrid_parallel(
                query_text, index, embeddings, doc_ids,
                top_k=request.top_k,
                fusion_method=request.fusion_method,
                bm25_k1=request.bm25_k1, bm25_b=request.bm25_b,
                dataset=request.dataset,
            )

    results: List[SearchResult] = []

    for rank, (doc_id, score) in enumerate(raw_results):
        original_text = get_original_text(request.dataset, str(doc_id))

        results.append(
            SearchResult(
                doc_id=str(doc_id),
                score=round(float(score), 4),
                rank=rank + 1,
                snippet=original_text[:500] if original_text else None,
            )
        )

    return results

    