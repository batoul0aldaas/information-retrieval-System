from services.ranking_evaluation_service.ranking_service import RankingService

# fake results (تجريب)
results = [
    ("doc1", 10.0),
    ("doc2", 5.0),
    ("doc3", 2.0),
]

# اختبار normalization
normalized = RankingService.normalize_scores(results)

print("=== NORMALIZED ===")
print(normalized)

# اختبار formatting
df = RankingService.format_results("q1", results)

print("\n=== DATAFRAME ===")
print(df)