from services.api_gateway.index_registry import get_index, get_embeddings
from services.retrieval_service.hybrid_retrieval import retrieve_hybrid_parallel

index = get_index("dataset1")
embeddings, doc_ids = get_embeddings("dataset1")

results = retrieve_hybrid_parallel(
    query="climate change",
    index=index,
    dataset="dataset1",
    embeddings=embeddings,
    doc_ids=doc_ids,
    top_k=10,
    fusion_method="rrf"   # أو "linear"
)

for rank, (doc_id, score) in enumerate(results, start=1):
    print(rank, doc_id, score)