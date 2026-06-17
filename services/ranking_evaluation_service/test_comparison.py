import pandas as pd
from services.ranking_evaluation_service.comparison_service import ComparisonService

# -------------------------
# fake runs (doc_id, score)
# -------------------------
runs = {
    "bm25": pd.DataFrame([
        ("doc1", 10),
        ("doc2", 8),
        ("doc3", 2),
    ], columns=["doc_id", "score"]),

    "tfidf": pd.DataFrame([
        ("doc2", 9),
        ("doc3", 7),
        ("doc1", 1),
    ], columns=["doc_id", "score"]),
}

# -------------------------
# fake qrels (doc_id, relevance)
# -------------------------
qrels = pd.DataFrame([
    ("doc1", 1),
    ("doc2", 1),
    ("doc3", 0),
], columns=["doc_id", "relevance"])

# -------------------------
# run comparison
# -------------------------
result = ComparisonService.compare_models(runs, qrels)

print("\n=== MODEL COMPARISON ===\n")
print(result)