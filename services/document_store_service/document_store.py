from __future__ import annotations

import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

_LOOKUP_CACHE: dict[str, dict[str, str]] = {}


def _get_lookup_path(dataset: str) -> Path:
    if dataset not in {"dataset1", "dataset2"}:
        raise ValueError("dataset must be dataset1 or dataset2")

    return DATA_DIR / dataset / "document_lookup.pkl"


def _load_lookup(dataset: str) -> dict[str, str]:
    if dataset in _LOOKUP_CACHE:
        return _LOOKUP_CACHE[dataset]

    path = _get_lookup_path(dataset)

    if not path.exists():
        raise FileNotFoundError(
            f"Document lookup not found: {path}. "
            "Run: python -m services.document_store_service.build_document_lookup"
        )

    with path.open("rb") as file:
        lookup = pickle.load(file)

    _LOOKUP_CACHE[dataset] = lookup
    return lookup


def get_original_text(dataset: str, doc_id: str) -> str:
    lookup = _load_lookup(dataset)
    return lookup.get(str(doc_id), "")