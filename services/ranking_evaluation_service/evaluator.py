"""
Ranking & Evaluation Service
Computes IR metrics: MAP, Recall, Precision@10, nDCG
using ir-measures library with standard qrels.
"""

import ir_measures
from ir_measures import MAP, Recall, nDCG, P
import pandas as pd
from typing import Dict, List, Tuple


def results_to_dataframe(
    query_id: str,
    ranked_results: List[Tuple[str, float]]
) -> pd.DataFrame:
    """Convert ranked results list to ir-measures Run DataFrame format."""
    rows = []
    for rank, (doc_id, score) in enumerate(ranked_results, start=1):
        rows.append({
            "query_id": query_id,
            "doc_id": doc_id,
            "score": score,
            "rank": rank,
        })
    return pd.DataFrame(rows)


def evaluate(
    run: pd.DataFrame,
    qrels: pd.DataFrame,
    metrics: List = None
) -> Dict[str, float]:
    """
    Evaluate a retrieval run against qrels.

    Parameters:
        run:     DataFrame with columns [query_id, doc_id, score]
        qrels:   DataFrame with columns [query_id, doc_id, relevance]
        metrics: List of ir_measures metrics (default: MAP, Recall, P@10, nDCG@10)

    Returns:
        Dict of {metric_name: average_score}
    """
    if metrics is None:
        metrics = [MAP, Recall, P@10, nDCG@10]

    run_obj = ir_measures.ScoredDocs.from_dataframe(run)
    qrels_obj = ir_measures.Qrels.from_dataframe(qrels)

    results = ir_measures.calc_aggregate(metrics, qrels_obj, run_obj)
    return {str(m): float(v) for m, v in results.items()}


def compare_models(
    runs: Dict[str, pd.DataFrame],
    qrels: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare multiple retrieval models side by side.

    Parameters:
        runs:  Dict of {model_name: run_dataframe}
        qrels: Ground truth relevance judgments

    Returns:
        DataFrame comparing MAP, Recall, P@10, nDCG@10 across models
    """
    comparison = []
    for model_name, run in runs.items():
        scores = evaluate(run, qrels)
        scores["model"] = model_name
        comparison.append(scores)

    return pd.DataFrame(comparison).set_index("model")
