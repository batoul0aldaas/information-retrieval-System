import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from services.api_gateway.index_registry import get_index, get_embeddings
from services.retrieval_service.tfidf_retrieval import retrieve_tfidf
from services.retrieval_service.bm25_retrieval import retrieve_bm25
from services.retrieval_service.embedding_retrieval import retrieve_embedding
from services.retrieval_service.hybrid_retrieval import (
    retrieve_hybrid_serial,
    retrieve_hybrid_parallel,
)


def precision_at_k(retrieved, relevant, k=10):
    retrieved = retrieved[:k]
    hits = sum(1 for doc_id in retrieved if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved, relevant, k=10):
    retrieved = retrieved[:k]

    if not relevant:
        return 0.0

    hits = sum(1 for doc_id in retrieved if doc_id in relevant)
    return hits / len(relevant)


def average_precision(retrieved, relevant, k=10):
    retrieved = retrieved[:k]

    score = 0.0
    hits = 0

    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / i

    if not relevant:
        return 0.0

    return score / len(relevant)


def ndcg_at_k(retrieved, relevant, k=10):
    import math

    retrieved = retrieved[:k]

    dcg = 0.0

    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            dcg += 1 / math.log2(i + 1)

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1 / math.log2(i + 1)
        for i in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def load_queries(path):
    queries = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            queries.append({
                "query_id": str(item["query_id"]),
                "text": item["text"]
            })

    return queries


def load_qrels(path):
    qrels = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            query_id, _, doc_id, rel = line.strip().split()

            if int(rel) > 0:
                qrels[str(query_id)].add(str(doc_id))

    return qrels


def search(model, query, dataset):
    index = get_index(dataset)

    if model == "tfidf":
        return retrieve_tfidf(query, index, top_k=10, dataset=dataset)

    if model == "bm25":
        return retrieve_bm25(query, index, top_k=10, dataset=dataset)

    embeddings, doc_ids = get_embeddings(dataset)

    if model == "embedding":
        return retrieve_embedding(
            query,
            embeddings,
            doc_ids,
            top_k=10
        )

    if model == "hybrid_serial":
        return retrieve_hybrid_serial(
            query,
            index,
            embeddings,
            doc_ids,
            final_top_k=10,
            dataset=dataset,
        )

    return retrieve_hybrid_parallel(
        query,
        index,
        embeddings,
        doc_ids,
        top_k=10,
        dataset=dataset,
    )


def evaluate(dataset="dataset1"):
    queries = load_queries(f"data/queries_{dataset}.jsonl")
    qrels = load_qrels(f"data/qrels_{dataset}.txt")

    print(f"Loaded queries: {len(queries)}")
    print(f"Loaded qrels: {len(qrels)}")

    models = [
        "tfidf",
        "bm25",
        "embedding",
        "hybrid_serial",
        "hybrid_parallel"
    ]

    rows = []

    for model in models:
        print(f"Evaluating {model}...")

        map_scores = []
        p10_scores = []
        recall_scores = []
        ndcg_scores = []

        matched_queries = 0

        for item in queries[:5000]:
            query_id = str(item["query_id"])

            if query_id not in qrels:
                continue

            matched_queries += 1

            relevant = qrels[query_id]

            results = search(
                model,
                item["text"],
                dataset
            )

            retrieved = [
                str(doc_id)
                for doc_id, _ in results
            ]

            map_scores.append(
                average_precision(retrieved, relevant)
            )

            p10_scores.append(
                precision_at_k(retrieved, relevant)
            )

            recall_scores.append(
                recall_at_k(retrieved, relevant)
            )

            ndcg_scores.append(
                ndcg_at_k(retrieved, relevant)
            )

        print(f"Matched queries: {matched_queries}")

        if matched_queries == 0:
            print(
                "WARNING: No matching query IDs found "
                "between queries and qrels."
            )

            rows.append({
                "Model": model,
                "MAP": 0.0,
                "Recall@10": 0.0,
                "P@10": 0.0,
                "nDCG@10": 0.0
            })

            continue

        rows.append({
            "Model": model,
            "MAP": sum(map_scores) / len(map_scores),
            "Recall@10": sum(recall_scores) / len(recall_scores),
            "P@10": sum(p10_scores) / len(p10_scores),
            "nDCG@10": sum(ndcg_scores) / len(ndcg_scores)
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = evaluate("dataset1")

    output = (
        "services/retrieval_service/reports/"
        "tables/metrics_report.csv"
    )

    Path(output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output, index=False)

    print(df)