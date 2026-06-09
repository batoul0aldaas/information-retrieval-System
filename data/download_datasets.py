"""
Dataset Downloader & Saver
Downloads two IR datasets to D: drive (C: is full).

Dataset 1: msmarco-passage/trec-dl-2019
  - 8.8M passages, 43 queries, graded qrels (0-3)

Dataset 2: beir/nq  (Natural Questions)
  - 2.6M Wikipedia passages, 3452 queries, qrels
"""

import os

# ── Redirect ir-datasets cache to D: drive (C: is full) ──────────────────────
os.environ["IR_DATASETS_HOME"] = r"D:\ir_datasets_cache"

import ir_datasets
import json
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────

DATASETS = {
    "dataset1": {
        "name": "msmarco-passage/trec-dl-2019",
        "save_dir": r"D:\IR_data\dataset1",
        "doc_limit": 500_000,
    },
    "dataset2": {
        "name": "beir/nq",
        "save_dir": r"D:\IR_data\dataset2",
        "doc_limit": 500_000,
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_jsonl(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} records -> {path}")


def download_dataset(key: str, config: dict):
    name     = config["name"]
    save_dir = config["save_dir"]
    limit    = config["doc_limit"]

    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Save dir   : {save_dir}")
    print(f"Doc limit  : {limit:,}")
    print(f"{'='*60}")

    dataset = ir_datasets.load(name)

    # ── Documents ─────────────────────────────────────────────
    print(f"\n[1/3] Loading documents (first {limit:,})...")
    docs = []
    for doc in tqdm(dataset.docs_iter()):
        text = doc.text if hasattr(doc, "text") else getattr(doc, "body", "")
        docs.append({"doc_id": doc.doc_id, "text": text})
        if len(docs) >= limit:
            break
    save_jsonl(docs, os.path.join(save_dir, "documents.jsonl"))

    # ── Queries ───────────────────────────────────────────────
    print(f"\n[2/3] Loading queries...")
    queries = []
    for q in dataset.queries_iter():
        queries.append({"query_id": q.query_id, "text": q.text})
    save_jsonl(queries, os.path.join(save_dir, "queries.jsonl"))

    # ── Qrels ─────────────────────────────────────────────────
    print(f"\n[3/3] Loading qrels...")
    qrels = []
    for qrel in dataset.qrels_iter():
        qrels.append({
            "query_id":  qrel.query_id,
            "doc_id":    qrel.doc_id,
            "relevance": qrel.relevance
        })
    save_jsonl(qrels, os.path.join(save_dir, "qrels.jsonl"))

    print(f"\nSummary [{name}]:")
    print(f"  Documents : {len(docs):,}")
    print(f"  Queries   : {len(queries):,}")
    print(f"  Qrels     : {len(qrels):,}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("IR Dataset Downloader")
    print(f"Cache dir  : {os.environ['IR_DATASETS_HOME']}")
    print(f"Data saved to D:\\IR_data\\\n")

    for key, config in DATASETS.items():
        download_dataset(key, config)

    print("\nAll datasets downloaded successfully!")
    print("Next: run the indexing script to build Inverted Indexes.")
