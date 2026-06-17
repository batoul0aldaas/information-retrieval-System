from services.query_service.query_processor import pseudo_relevance_feedback
from services.api_gateway.index_registry import get_index

index = get_index("dataset1")

query = "climate change"

expanded_query = pseudo_relevance_feedback(
    query=query,
    index=index,
    top_k=5,
    expansion_terms=5
)

print("Original:", query)
print("Expanded:", expanded_query)