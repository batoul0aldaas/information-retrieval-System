from typing import List, Tuple, Dict
import pandas as pd


class RankingService:
    """
    Unified ranking layer for all retrieval models.
    Standardizes output format for BM25, TF-IDF, Embedding, Hybrid.
    """

    @staticmethod
    def format_results(
        query_id: str,
        results: List[Tuple[str, float]]
    ) -> pd.DataFrame:
        """
        Convert raw results into a unified ranked DataFrame.
        """
        rows = []

        for rank, (doc_id, score) in enumerate(results, start=1):
            rows.append({
                "query_id": query_id,
                "doc_id": str(doc_id),
                "score": float(score),
                "rank": rank
            })

        return pd.DataFrame(rows)

    @staticmethod
    def normalize_scores(results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Normalize scores to [0,1] for fair comparison.
        """
        if not results:
            return results

        max_score = max(score for _, score in results)
        if max_score == 0:
            return results

        return [
            (doc_id, score / max_score)
            for doc_id, score in results
        ]

    @staticmethod
    def merge_runs(runs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple model runs into one comparison table.
        """
        all_data = []

        for model_name, df in runs.items():
            df_copy = df.copy()
            df_copy["model"] = model_name
            all_data.append(df_copy)

        return pd.concat(all_data, ignore_index=True)