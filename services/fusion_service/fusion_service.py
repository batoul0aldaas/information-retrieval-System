from typing import Dict, List, Tuple

# ─── Fusion Methods ───────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    results_list: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF)
    """
    rrf_scores: Dict[str, float] = {}

    for results in results_list:
        for rank, (doc_id, _) in enumerate(results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def linear_combination_fusion(
    results_list: List[List[Tuple[str, float]]],
    weights: List[float] = None
) -> List[Tuple[str, float]]:
    """
    Linear weighted fusion
    """
    if weights is None:
        weights = [1.0 / len(results_list)] * len(results_list)

    combined: Dict[str, float] = {}

    for results, weight in zip(results_list, weights):
        if not results:
            continue

        max_score = max(s for _, s in results) or 1

        for doc_id, score in results:
            normalized = score / max_score
            combined[doc_id] = combined.get(doc_id, 0) + weight * normalized

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)

