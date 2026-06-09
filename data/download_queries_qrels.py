"""
Resume Download: Queries + Qrels only
(Documents already saved, only queries/qrels are missing)
"""

import os
os.environ["IR_DATASETS_HOME"] = r"D:\ir_datasets_cache"

import ir_datasets
import json
from tqdm import tqdm

DATASETS = {
    "dataset1": {
        "name": "msmarco-passage/trec-dl-2019",
        "save_dir": r"D:\IR_data\dataset1",
    },
    "dataset2": {
        "name": "beir/nq",
        "save_dir": r"D:\IR_data\dataset2",
        "doc_limit": 500_000,
    },
}


def save_jsonl(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} -> {path}")


def load_queries_qrels(key: str, config: dict):
    name     = config["name"]
    save_dir = config["save_dir"]

    print(f"\n{'='*55}")
    print(f"Loading queries + qrels: {name}")
    print(f"{'='*55}")

    dataset = ir_datasets.load(name)

    # Documents for dataset2 (not yet downloaded)
    if key == "dataset2":
        doc_path = os.path.join(save_dir, "documents.jsonl")
        if not os.path.exists(doc_path):
            limit = config.get("doc_limit", 500_000)
            print(f"\n[0/2] Loading documents (first {limit:,})...")
            docs = []
            for doc in tqdm(dataset.docs_iter()):
                text = doc.text if hasattr(doc, "text") else getattr(doc, "body", "")
                docs.append({"doc_id": doc.doc_id, "text": text})
                if len(docs) >= limit:
                    break
            save_jsonl(docs, doc_path)
        else:
            print(f"\n[0/2] Documents already saved: {doc_path}")

    # Queries
    q_path = os.path.join(save_dir, "queries.jsonl")
    if not os.path.exists(q_path):
        print(f"\n[1/2] Loading queries...")
        queries = []
        for q in dataset.queries_iter():
            queries.append({"query_id": q.query_id, "text": q.text})
        save_jsonl(queries, q_path)
    else:
        print(f"\n[1/2] Queries already saved.")

    # Qrels
    qr_path = os.path.join(save_dir, "qrels.jsonl")
    if not os.path.exists(qr_path):
        print(f"\n[2/2] Loading qrels...")
        qrels = []
        for qrel in dataset.qrels_iter():
            qrels.append({
                "query_id":  qrel.query_id,
                "doc_id":    qrel.doc_id,
                "relevance": qrel.relevance
            })
        save_jsonl(qrels, qr_path)
    else:
        print(f"\n[2/2] Qrels already saved.")

    # Summary
    files = {
        "documents": os.path.join(save_dir, "documents.jsonl"),
        "queries":   q_path,
        "qrels":     qr_path,
    }
    print(f"\nFiles saved for [{name}]:")
    for fname, fpath in files.items():
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1_048_576
            print(f"  {fname:<12}: {fpath}  ({size_mb:.1f} MB)")
        else:
            print(f"  {fname:<12}: MISSING")


if __name__ == "__main__":
    for key, config in DATASETS.items():
        load_queries_qrels(key, config)

    print("\nDone! All queries and qrels saved.")
