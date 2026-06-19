from services.retrieval_service.bm25_retrieval import retrieve_bm25
from services.retrieval_service.embedding_retrieval import retrieve_embeddings
from services.retrieval_service.hybrid_retrieval import retrieve_hybrid

from services.indexing_service.inverted_index import InvertedIndex
from services.ranking_evaluation_service.evaluator import (
    precision_at_k,
    recall_at_k
)

# =========================
# Mock dataset (لاحقاً نربطه الحقيقي)
# =========================
queries = {
    "q1": "climate change",
    "q2": "machine learning"
}

relevant_docs = {
    "q1": ["D1", "D5"],
    "q2": ["D2", "D8"]
}

# =========================
# Load index
# =========================
index = InvertedIndex.load()

results = []

for qid, query in queries.items():

    bm25_res = retrieve_bm25(query, index, top_k=10)
    emb_res = retrieve_embeddings(query, index, top_k=10)
    hybrid_res = retrieve_hybrid(query, index, top_k=10)

    bm25_ids = [doc for doc, score in bm25_res]
    emb_ids = [doc for doc, score in emb_res]
    hybrid_ids = [doc for doc, score in hybrid_res]

    results.append({
        "query": qid,

        "bm25_p@10": precision_at_k(bm25_ids, relevant_docs[qid], 10),
        "bm25_recall": recall_at_k(bm25_ids, relevant_docs[qid], 10),

        "emb_p@10": precision_at_k(emb_ids, relevant_docs[qid], 10),
        "emb_recall": recall_at_k(emb_ids, relevant_docs[qid], 10),

        "hybrid_p@10": precision_at_k(hybrid_ids, relevant_docs[qid], 10),
        "hybrid_recall": recall_at_k(hybrid_ids, relevant_docs[qid], 10),
    })

print(results)