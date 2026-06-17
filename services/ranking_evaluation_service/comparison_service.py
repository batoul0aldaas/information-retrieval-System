import pandas as pd
from typing import Dict


class ComparisonService:

    @staticmethod
    def evaluate(run: pd.DataFrame, qrels: pd.DataFrame):

        # تحويل إلى dict سريع
        relevance = {
            row["doc_id"]: row["relevance"]
            for _, row in qrels.iterrows()
        }

        # ترتيب النتائج
        run_sorted = run.sort_values("score", ascending=False)

        retrieved = list(run_sorted["doc_id"])

        # Precision@10
        top10 = retrieved[:10]
        relevant_top10 = [doc for doc in top10 if relevance.get(doc, 0) > 0]

        p10 = len(relevant_top10) / 10

        # Recall
        total_relevant = sum(1 for v in relevance.values() if v > 0)
        recall = len(set(relevant_top10)) / total_relevant if total_relevant else 0

        # MAP (simplified)
        ap_sum = 0
        hit = 0

        for i, doc in enumerate(retrieved, start=1):
            if relevance.get(doc, 0) > 0:
                hit += 1
                ap_sum += hit / i

        map_score = ap_sum / total_relevant if total_relevant else 0

        return {
            "MAP": map_score,
            "P@10": p10,
            "Recall": recall
        }

    @staticmethod
    def compare_models(runs: Dict[str, pd.DataFrame], qrels: pd.DataFrame):

        results = []

        for name, run in runs.items():
            scores = ComparisonService.evaluate(run, qrels)
            scores["model"] = name
            results.append(scores)

        return pd.DataFrame(results).set_index("model")