"""
Direct Download of Queries & Qrels
Bypasses ir-datasets file locking issue on Windows.
Downloads files directly from official sources.
"""

import os
import json
import gzip
import urllib.request
import tarfile
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "ir_datasets_cache"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["IR_DATASETS_HOME"] = str(CACHE_DIR)


# ─── Dataset 1: msmarco-passage/dev ──────────────────────────────────────────
# Uses the same 8.8M corpus (already downloaded), no NIST registration needed.
# 6980 dev queries with qrels.

DATASET1_DIR = DATA_DIR / "dataset1"
DATASET1_NAME = "msmarco-passage/dev"


class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading: {url}")
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(dest)) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
    print(f"  Saved: {dest}")


def save_jsonl(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} records -> {path}")


# ─── Dataset 1 ────────────────────────────────────────────────────────────────

def build_dataset1():
    print("\n" + "="*55)
    print(f"Dataset 1: {DATASET1_NAME}")
    print("="*55)

    import ir_datasets
    dataset = ir_datasets.load(DATASET1_NAME)
    os.makedirs(DATASET1_DIR, exist_ok=True)

    # Documents
    doc_path = DATASET1_DIR / "documents.jsonl"
    if not os.path.exists(doc_path):
        print("  Loading documents (first 500,000)...")
        docs = []
        for doc in tqdm(dataset.docs_iter()):
            text = doc.text if hasattr(doc, "text") else getattr(doc, "body", "")
            docs.append({"doc_id": doc.doc_id, "text": text})
            if len(docs) >= 500_000:
                break
        save_jsonl(docs, doc_path)
    else:
        print("  Documents already saved.")

    # Queries (extracted from already-downloaded collectionandqueries.tar.gz)
    q_path = DATASET1_DIR / "queries.jsonl"
    if not os.path.exists(q_path):
        print("  Loading queries...")
        queries = [{"query_id": q.query_id, "text": q.text}
                   for q in tqdm(dataset.queries_iter())]
        save_jsonl(queries, q_path)
    else:
        print("  Queries already saved.")

    # Qrels
    qr_path = DATASET1_DIR / "qrels.jsonl"
    if not os.path.exists(qr_path):
        print("  Loading qrels...")
        qrels = [{"query_id": qr.query_id, "doc_id": qr.doc_id, "relevance": qr.relevance}
                 for qr in tqdm(dataset.qrels_iter())]
        save_jsonl(qrels, qr_path)
    else:
        print("  Qrels already saved.")


# ─── Dataset 2: beir/nq ───────────────────────────────────────────────────────

def build_dataset2():
    print("\n" + "="*55)
    print("Dataset 2: beir/nq (Natural Questions)")
    print("="*55)

    import ir_datasets
    dataset = ir_datasets.load("beir/nq")
    save_dir = DATA_DIR / "dataset2"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Documents
    doc_path = save_dir / "documents.jsonl"
    if not os.path.exists(doc_path):
        print("  Loading documents (first 500,000)...")
        docs = []
        for doc in tqdm(dataset.docs_iter()):
            text = doc.text if hasattr(doc, "text") else getattr(doc, "body", "")
            docs.append({"doc_id": doc.doc_id, "text": text})
            if len(docs) >= 500_000:
                break
        save_jsonl(docs, doc_path)
    else:
        print("  Documents already saved.")

    # Queries
    q_path = save_dir / "queries.jsonl"
    if not os.path.exists(q_path):
        print("  Loading queries...")
        queries = [{"query_id": q.query_id, "text": q.text} for q in dataset.queries_iter()]
        save_jsonl(queries, q_path)
    else:
        print("  Queries already saved.")

    # Qrels
    qr_path = save_dir / "qrels.jsonl"
    if not os.path.exists(qr_path):
        print("  Loading qrels...")
        qrels = [{"query_id": qr.query_id, "doc_id": qr.doc_id, "relevance": qr.relevance}
                 for qr in dataset.qrels_iter()]
        save_jsonl(qrels, qr_path)
    else:
        print("  Qrels already saved.")


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "="*55)
    print("FINAL SUMMARY")
    print("="*55)
    for ds, folder in [("Dataset 1", DATA_DIR / "dataset1"), ("Dataset 2", DATA_DIR / "dataset2")]:
        print(f"\n{ds}: {folder}")
        for fname in ["documents.jsonl", "queries.jsonl", "qrels.jsonl"]:
            fpath = folder / fname
            if os.path.exists(fpath):
                size_mb = os.path.getsize(fpath) / 1_048_576
                print(f"  {fname:<20} {size_mb:>8.1f} MB  OK")
            else:
                print(f"  {fname:<20}   MISSING")


if __name__ == "__main__":
    # Delete any leftover temp files
    for tmp in [
        CACHE_DIR / "msmarco-passage" / "trec-dl-2019" / "queries.tsv.tmp0",
        CACHE_DIR / "msmarco-passage" / "dev" / "queries.tsv.tmp0",
    ]:
        if tmp.exists():
            tmp.unlink()
            print(f"Removed temp file: {tmp}")

    build_dataset1()
    build_dataset2()
    print_summary()
    print("\nAll done! Ready for preprocessing and indexing.")
