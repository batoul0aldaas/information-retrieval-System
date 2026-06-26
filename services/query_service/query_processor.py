"""
Query Processing & Refinement Service
Handles: query normalization, spell correction, synonym expansion,
         query suggestion, and pseudo relevance feedback.
"""
from typing import List, Dict, Optional
from spellchecker import SpellChecker
from collections import Counter

from services.preprocessing_service.preprocessor import preprocess
from services.retrieval_service.bm25_retrieval import retrieve_bm25
from services.document_store_service.document_store import get_original_text

try:
    from nltk.corpus import wordnet
    import nltk
    WORDNET_AVAILABLE = True
except Exception:
    WORDNET_AVAILABLE = False

spell = SpellChecker()


def process_query(query: str, use_stemming: bool = False) -> List[str]:
    """Preprocess query using the same pipeline as documents."""
    return preprocess(query, use_stemming=use_stemming)


def correct_spelling(query: str) -> str:
    """Correct misspelled words in the query."""
    words = query.split()
    corrected = []
    for word in words:
        correction = spell.correction(word)
        corrected.append(correction if correction else word)
    return " ".join(corrected)


def expand_with_synonyms(tokens: List[str], max_synonyms: int = 2) -> List[str]:
    """
    Add synonyms for query terms using WordNet.
    Expands each token with up to max_synonyms synonyms.
    """
    if not WORDNET_AVAILABLE:
        return tokens

    expanded = list(tokens)
    for token in tokens:
        synonyms = set()
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym != token:
                    synonyms.add(synonym)
                if len(synonyms) >= max_synonyms:
                    break
            if len(synonyms) >= max_synonyms:
                break
        expanded.extend(list(synonyms)[:max_synonyms])
    return expanded


def suggest_queries(query: str, history: List[str], top_k: int = 5) -> List[str]:
    """
    Suggest queries based on user search history.
    Returns history entries that share keywords with the current query.
    """
    query_tokens = set(query.lower().split())
    suggestions = []
    for past_query in reversed(history):
        past_tokens = set(past_query.lower().split())
        if query_tokens & past_tokens and past_query != query:
            suggestions.append(past_query)
        if len(suggestions) >= top_k:
            break
    return suggestions

def refine_query(
    query: str,
    history: Optional[List[str]] = None,
    use_spell_correction: bool = True,
    use_synonyms: bool = True,
    use_stemming: bool = False,
    use_prf: bool = False,
    index=None,
    dataset: str = "dataset1",
) -> Dict:

    result = {"original": query}

    # 1. spell correction
    if use_spell_correction:
        corrected = correct_spelling(query)
    else:
        corrected = query

    result["corrected"] = corrected

    # 2. tokenization
    tokens = process_query(corrected, use_stemming=use_stemming)
    result["tokens"] = tokens

    # 3. synonyms (baseline)
    expanded = expand_with_synonyms(tokens) if use_synonyms else tokens
    result["expanded_tokens"] = expanded

    # 4. PRF (IMPORTANT FIXED)
    if use_prf and index is not None:
        prf_query = pseudo_relevance_feedback(
            corrected,
            index,
             dataset=dataset,
            top_k=5,
            expansion_terms=5
        )

        result["prf_query"] = prf_query

        prf_tokens = process_query(prf_query, use_stemming=use_stemming)

        if use_synonyms:
            result["expanded_tokens"] = expand_with_synonyms(prf_tokens)
        else:
            result["expanded_tokens"] = prf_tokens

    # 5. suggestions
    if history:
        result["suggestions"] = suggest_queries(query, history)
    else:
        result["suggestions"] = []

    return result
from collections import Counter
import re

STOP_WORDS = {
    "the", "is", "in", "of", "and", "to", "a", "for", "on", "with",
    "time", "current", "people", "thing", "things", "use", "used",
    "based", "data", "information", "system"
}


def clean_tokens(tokens):
    """Remove noisy / stop words and short tokens"""
    return [
        t for t in tokens
        if t not in STOP_WORDS and len(t) > 2 and t.isalpha()
    ]


def pseudo_relevance_feedback(
    query: str,
    index,
    dataset: str,
    top_k: int = 5,
    expansion_terms: int = 5,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75
) -> str:

    # 1. retrieve initial docs
    initial_results = retrieve_bm25(
        query,
        index,
        top_k=top_k,
        k1=bm25_k1,
        b=bm25_b,
        dataset=dataset,
    )

    # 2. collect words from top docs
    tokens_pool = []

    for doc_id, _ in initial_results:
        text = get_original_text(dataset, str(doc_id))

        if not text:
            continue

        # تنظيف بسيط
        words = re.findall(r"[a-zA-Z]+", text.lower())
        words = clean_tokens(words)

        tokens_pool.extend(words)

    if not tokens_pool:
        return query

    # 3. pick most frequent meaningful terms
    most_common = Counter(tokens_pool).most_common(expansion_terms)

    expansion_words = [w for w, _ in most_common]

    # 4. build expanded query (clean + controlled expansion)
    expanded_query = query + " " + " ".join(expansion_words)

    return expanded_query