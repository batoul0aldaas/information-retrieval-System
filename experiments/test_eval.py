from services.ranking_evaluation_service.evaluator import precision_at_k

retrieved = [
    {"doc_id": "D1"},
    {"doc_id": "D2"},
    {"doc_id": "D3"}
]

relevant = ["D1", "D3"]

print("Precision@3 =", precision_at_k(retrieved, relevant, 3))