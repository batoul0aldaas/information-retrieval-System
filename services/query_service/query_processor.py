"""
Query Processing & Refinement Service
Handles: query normalization, spell correction, synonym expansion,
         query suggestion, and pseudo relevance feedback.
"""

from typing import List, Dict, Optional
from spellchecker import SpellChecker
from services.preprocessing_service.preprocessor import preprocess

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
    use_stemming: bool = False
) -> Dict:
    """
    Full query refinement pipeline.
    Returns dict with original, corrected, tokens, expanded tokens, suggestions.
    """
    result = {"original": query}

    if use_spell_correction:
        corrected = correct_spelling(query)
        result["corrected"] = corrected
    else:
        corrected = query
        result["corrected"] = query

    tokens = process_query(corrected, use_stemming=use_stemming)
    result["tokens"] = tokens

    if use_synonyms:
        expanded = expand_with_synonyms(tokens)
        result["expanded_tokens"] = expanded
    else:
        result["expanded_tokens"] = tokens

    if history:
        result["suggestions"] = suggest_queries(query, history)
    else:
        result["suggestions"] = []

    return result
