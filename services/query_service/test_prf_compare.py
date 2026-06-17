from services.query_service.query_processor import pseudo_relevance_feedback
from services.api_gateway.index_registry import get_index
from services.retrieval_service.bm25_retrieval import retrieve_bm25

# تحميل index
index = get_index("dataset1")

query = "climate change"

# ─── بدون PRF ───
base_results = retrieve_bm25(query, index, top_k=5)

print("\n=== WITHOUT PRF ===")
for doc_id, score in base_results:
    print(doc_id, score)

# ─── مع PRF ───
expanded_query = pseudo_relevance_feedback(
    query=query,
    index=index,
    top_k=5,
    expansion_terms=5
)

print("\nExpanded Query:", expanded_query)

prf_results = retrieve_bm25(expanded_query, index, top_k=5)

print("\n=== WITH PRF ===")
for doc_id, score in prf_results:
    print(doc_id, score)