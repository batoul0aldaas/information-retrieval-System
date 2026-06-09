"""
Indexing Service
Builds and manages an Inverted Index for fast document retrieval.
"""

import json
import pickle
import os
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from services.preprocessing_service.preprocessor import preprocess


class InvertedIndex:
    """
    Inverted Index: maps each term to the list of documents containing it,
    along with term frequency and document positions.
    """

    def __init__(self):
        self.index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.doc_lengths: Dict[str, int] = {}
        self.doc_count: int = 0
        self.avg_doc_length: float = 0.0

    def build(self, documents: Dict[str, str], use_stemming: bool = False,
              skip_ids: set = None) -> None:
        """
        Build the index from a dictionary of {doc_id: text}.

        Args:
            documents:  {doc_id: text} mapping.
            use_stemming: use Porter Stemmer instead of lemmatization.
            skip_ids:   set of doc_ids to ignore (e.g. empty docs from preprocessing).
        """
        skip_ids = skip_ids or set()
        filtered = {k: v for k, v in documents.items() if k not in skip_ids}
        skipped  = len(documents) - len(filtered)
        if skipped:
            print(f"  Skipping {skipped:,} empty/invalid documents.")

        print(f"Building index for {len(filtered):,} documents...")
        total_length = 0

        for doc_id, text in tqdm(filtered.items()):
            tokens = preprocess(text, use_stemming=use_stemming)
            if not tokens:
                continue
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            for term, freq in term_freq.items():
                self.index[term][doc_id] = freq

        self.doc_count = len(self.doc_lengths)
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0
        print(f"Index built: {len(self.index):,} unique terms, {self.doc_count:,} documents")

    def build_from_tokens(self, token_data: Dict[str, List[str]],
                          skip_ids: set = None) -> None:
        """
        Build the index directly from pre-tokenized data {doc_id: [tokens]}.
        Faster than build() — skips preprocessing (already done).

        Args:
            token_data: {doc_id: tokens_list} from processed_datasetX.jsonl
            skip_ids:   doc_ids to ignore (empty docs list)
        """
        skip_ids = skip_ids or set()
        skipped  = sum(1 for k in token_data if k in skip_ids)
        if skipped:
            print(f"  Skipping {skipped:,} empty/invalid documents.")

        print(f"Building index from {len(token_data) - skipped:,} pre-tokenized documents...")
        total_length = 0

        for doc_id, tokens in tqdm(token_data.items()):
            if doc_id in skip_ids or not tokens:
                continue
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            for term, freq in term_freq.items():
                self.index[term][doc_id] = freq

        self.doc_count = len(self.doc_lengths)
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0
        print(f"Index built: {len(self.index):,} unique terms, {self.doc_count:,} documents")

    def get_postings(self, term: str) -> Dict[str, int]:
        """Return {doc_id: term_freq} for a given term."""
        return self.index.get(term, {})

    def get_doc_length(self, doc_id: str) -> int:
        """Return number of tokens in a document."""
        return self.doc_lengths.get(doc_id, 0)

    def save(self, path: str) -> None:
        """Save index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "index": dict(self.index),
                "doc_lengths": self.doc_lengths,
                "doc_count": self.doc_count,
                "avg_doc_length": self.avg_doc_length,
            }, f)
        print(f"Index saved to {path}")

    def load(self, path: str) -> None:
        """Load index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = defaultdict(dict, data["index"])
        self.doc_lengths = data["doc_lengths"]
        self.doc_count = data["doc_count"]
        self.avg_doc_length = data["avg_doc_length"]
        print(f"Index loaded: {len(self.index)} terms, {self.doc_count} documents")
