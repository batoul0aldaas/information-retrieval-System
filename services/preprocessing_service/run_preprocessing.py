"""
Preprocessing Runner
====================
Reads raw dataset files and produces 6 clean output files:

  data/processed_dataset1.jsonl   ← documents with tokens (for indexing)
  data/processed_dataset2.jsonl
  data/queries_dataset1.jsonl     ← queries with tokens (for retrieval)
  data/queries_dataset2.jsonl
  data/qrels_dataset1.txt         ← relevance judgments (TREC format)
  data/qrels_dataset2.txt

Output formats
--------------
processed_datasetX.jsonl  →  {"doc_id": "0", "tokens": ["word1", "word2", ...]}
queries_datasetX.jsonl    →  {"query_id": "q1", "text": "...", "tokens": [...]}
qrels_datasetX.txt        →  query_id  0  doc_id  relevance   (TREC format)

Usage
-----
    # From project root (d:\\IR):
    python -m services.preprocessing_service.run_preprocessing

    # Stemming instead of lemmatization:
    python -m services.preprocessing_service.run_preprocessing --stemming

    # Process only one dataset:
    python -m services.preprocessing_service.run_preprocessing --dataset dataset1

    # Re-run and overwrite existing output:
    python -m services.preprocessing_service.run_preprocessing --force
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, List, Tuple

from tqdm import tqdm

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.preprocessing_service.preprocessor import preprocess_batch, preprocess  # noqa: E402

# ── Dataset configuration ─────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data"

DATASETS = {
    "dataset1": {
        "label":      "MS MARCO / TREC-DL 2019",
        "docs_in":    DATA_DIR / "dataset1" / "documents.jsonl",
        "queries_in": DATA_DIR / "dataset1" / "queries.jsonl",
        "qrels_in":   DATA_DIR / "dataset1" / "qrels.jsonl",
        "docs_out":   DATA_DIR / "processed_dataset1.jsonl",
        "queries_out": DATA_DIR / "queries_dataset1.jsonl",
        "qrels_out":  DATA_DIR / "qrels_dataset1.txt",
        "empty_out":  DATA_DIR / "empty_docs_dataset1.txt",
        "checkpoint": DATA_DIR / "dataset1" / ".checkpoint_docs.txt",
    },
    "dataset2": {
        "label":      "BEIR / Natural Questions",
        "docs_in":    DATA_DIR / "dataset2" / "documents.jsonl",
        "queries_in": DATA_DIR / "dataset2" / "queries.jsonl",
        "qrels_in":   DATA_DIR / "dataset2" / "qrels.jsonl",
        "docs_out":   DATA_DIR / "processed_dataset2.jsonl",
        "queries_out": DATA_DIR / "queries_dataset2.jsonl",
        "qrels_out":  DATA_DIR / "qrels_dataset2.txt",
        "empty_out":  DATA_DIR / "empty_docs_dataset2.txt",
        "checkpoint": DATA_DIR / "dataset2" / ".checkpoint_docs.txt",
    },
}

BATCH_SIZE = 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def load_checkpoint(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {lineno} skipped: {e}")


# ── Step 1: Process Documents ─────────────────────────────────────────────────

def process_documents(cfg: dict, use_stemming: bool, force: bool, sample: int = 0) -> dict:
    """
    Read documents.jsonl → tokenize → write processed_datasetX.jsonl
    Also writes empty_docs_datasetX.txt listing doc_ids with zero tokens
    so the Indexing Service can skip them automatically.

    Output: {"doc_id": "...", "tokens": ["word1", "word2", ...]}
    """
    docs_in    = cfg["docs_in"]
    out_path   = cfg["docs_out"]
    empty_path = cfg["empty_out"]
    chk_path   = cfg["checkpoint"]

    if not docs_in.exists():
        print(f"  [ERROR] Missing: {docs_in}")
        return {"status": "missing"}

    if force and out_path.exists():
        out_path.unlink()
        chk_path.unlink(missing_ok=True)
        empty_path.unlink(missing_ok=True)

    done_ids   = load_checkpoint(chk_path)
    total_file = count_lines(docs_in)
    total      = min(total_file, sample) if sample else total_file

    print(f"\n  [1/3] Documents  ({total:,} / {total_file:,}, {len(done_ids):,} already done)")
    print(f"        Input  : {docs_in.name}")
    print(f"        Output : {out_path.name}")
    print(f"        Empty  : {empty_path.name}  (doc_ids with 0 tokens)")

    processed = skipped = empty = 0
    batch_ids:   List[str] = []
    batch_texts: List[str] = []
    t0 = time.time()

    out_fh   = open(out_path,   "a", encoding="utf-8")
    chk_fh   = open(chk_path,   "a", encoding="utf-8")
    empty_fh = open(empty_path, "a", encoding="utf-8")

    def flush():
        nonlocal processed, empty
        if not batch_ids:
            return
        token_lists = preprocess_batch(batch_texts, use_stemming=use_stemming,
                                       batch_size=BATCH_SIZE)
        for doc_id, tokens in zip(batch_ids, token_lists):
            if not tokens:
                # Log empty doc — Indexing will skip it
                empty_fh.write(doc_id + "\n")
                empty += 1
            out_fh.write(json.dumps({"doc_id": doc_id, "tokens": tokens},
                                    ensure_ascii=False) + "\n")
            chk_fh.write(doc_id + "\n")
            processed += 1
        out_fh.flush()
        chk_fh.flush()
        empty_fh.flush()
        batch_ids.clear()
        batch_texts.clear()

    docs_seen = 0
    with tqdm(total=total, unit="doc", dynamic_ncols=True, leave=True) as bar:
        for obj in iter_jsonl(docs_in):
            if sample and docs_seen >= sample:
                break
            docs_seen += 1
            bar.update(1)
            doc_id = obj["doc_id"]
            if doc_id in done_ids:
                skipped += 1
                continue
            batch_ids.append(doc_id)
            batch_texts.append(obj.get("text") or "")
            if len(batch_ids) >= BATCH_SIZE:
                flush()
        flush()

    out_fh.close()
    chk_fh.close()
    empty_fh.close()

    elapsed   = time.time() - t0
    speed     = processed / elapsed if elapsed > 0 else 0
    pct_empty = empty / processed * 100 if processed else 0
    print(f"        Done : {processed:,} processed | {skipped:,} skipped")
    print(f"        Empty: {empty:,} ({pct_empty:.2f}%) — saved to {empty_path.name}")
    print(f"        Speed: {elapsed:.0f}s  ({speed:.0f} docs/sec)")
    return {"processed": processed, "empty": empty, "elapsed": elapsed}


# ── Step 2: Process Queries ───────────────────────────────────────────────────

def process_queries(cfg: dict, use_stemming: bool, force: bool) -> dict:
    """
    Read queries.jsonl → tokenize → write queries_datasetX.jsonl
    Output: {"query_id": "...", "text": "original query", "tokens": [...]}
    """
    queries_in = cfg["queries_in"]
    out_path   = cfg["queries_out"]

    if not queries_in.exists():
        print(f"  [ERROR] Missing: {queries_in}")
        return {"status": "missing"}

    if force and out_path.exists():
        out_path.unlink()

    if out_path.exists():
        count = count_lines(out_path)
        print(f"\n  [2/3] Queries   — already done ({count:,} queries)")
        return {"processed": count}

    total = count_lines(queries_in)
    print(f"\n  [2/3] Queries   ({total:,} total)")
    print(f"        Input  : {queries_in.name}")
    print(f"        Output : {out_path.name}")

    processed = 0
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as out_fh:
        for obj in tqdm(iter_jsonl(queries_in), total=total, unit="query",
                        dynamic_ncols=True, leave=True):
            query_id = obj["query_id"]
            text     = obj.get("text") or ""
            tokens   = preprocess(text, use_stemming=use_stemming)
            out_fh.write(json.dumps({
                "query_id": query_id,
                "text":     text,
                "tokens":   tokens,
            }, ensure_ascii=False) + "\n")
            processed += 1

    elapsed = time.time() - t0
    print(f"        Done: {processed:,} queries in {elapsed:.1f}s")
    return {"processed": processed}


# ── Step 3: Convert Qrels ─────────────────────────────────────────────────────

def convert_qrels(cfg: dict, force: bool) -> dict:
    """
    Read qrels.jsonl → write qrels_datasetX.txt in TREC format.
    TREC format: query_id  0  doc_id  relevance
    (The '0' column is the iteration number, always 0 by convention)
    """
    qrels_in = cfg["qrels_in"]
    out_path = cfg["qrels_out"]

    if not qrels_in.exists():
        print(f"  [ERROR] Missing: {qrels_in}")
        return {"status": "missing"}

    if force and out_path.exists():
        out_path.unlink()

    if out_path.exists():
        count = count_lines(out_path)
        print(f"\n  [3/3] Qrels     — already done ({count:,} lines)")
        return {"converted": count}

    total = count_lines(qrels_in)
    print(f"\n  [3/3] Qrels     ({total:,} judgments)")
    print(f"        Input  : {qrels_in.name}")
    print(f"        Output : {out_path.name}  (TREC format)")

    converted = 0
    with open(out_path, "w", encoding="utf-8") as out_fh:
        for obj in tqdm(iter_jsonl(qrels_in), total=total, unit="qrel",
                        dynamic_ncols=True, leave=True):
            # TREC format: qid  iter  docno  rel
            out_fh.write(
                f"{obj['query_id']}\t0\t{obj['doc_id']}\t{obj['relevance']}\n"
            )
            converted += 1

    print(f"        Done: {converted:,} qrels written")
    return {"converted": converted}


# ── Dataset runner ────────────────────────────────────────────────────────────

def process_dataset(key: str, cfg: dict, use_stemming: bool, force: bool, sample: int = 0):
    label = cfg["label"]
    print(f"\n{'='*60}")
    print(f"  Dataset : {label}  [{key}]")
    print(f"  Mode    : {'Stemming' if use_stemming else 'Lemmatization'}")
    if sample:
        print(f"  Sample  : first {sample:,} documents only")
    print(f"{'='*60}")

    process_documents(cfg, use_stemming, force, sample)
    process_queries(cfg, use_stemming, force)
    convert_qrels(cfg, force)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess IR datasets — produces 6 output files."
    )
    parser.add_argument("--dataset", choices=["dataset1", "dataset2", "all"],
                        default="all")
    parser.add_argument("--stemming", action="store_true",
                        help="Use Stemming instead of Lemmatization")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files")
    parser.add_argument("--sample", type=int, default=0,
                        help="Process only first N documents (0 = all). "
                             "Use for quick testing, e.g. --sample 5000")
    args = parser.parse_args()

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    print("\n" + "="*60)
    print("  IR System — Preprocessing Runner")
    print(f"  Output directory : {DATA_DIR}")
    print(f"  Mode             : {'Stemming' if args.stemming else 'Lemmatization'}")
    print(f"  Datasets         : {', '.join(keys)}")
    print("="*60)

    total_start = time.time()
    for key in keys:
        process_dataset(key, DATASETS[key], args.stemming, args.force, args.sample)

    # ── Final file summary ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  OUTPUT FILES")
    print("="*60)
    expected = []
    for key in keys:
        cfg = DATASETS[key]
        expected += [cfg["docs_out"], cfg["queries_out"], cfg["qrels_out"]]

    for path in expected:
        if path.exists():
            size_mb = path.stat().st_size / 1_048_576
            lines   = count_lines(path)
            print(f"  {'OK':>3}  {path.name:<35}  {lines:>8,} lines  {size_mb:>7.1f} MB")
        else:
            print(f"  {'--':>3}  {path.name:<35}  NOT CREATED")

    print(f"\n  Total time : {time.time() - total_start:.1f}s")
    print("  Next step  : run Indexing Service")
    print("="*60)


if __name__ == "__main__":
    main()
