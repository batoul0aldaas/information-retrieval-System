"""
Preprocessing Service
=====================
Covers every technique taught in the IR lectures:

  Text_processing.ipynb:
    - Lowercase / Normalization
    - Tokenization          →  word_tokenize() from NLTK  (lecture approach)
    - Stopword Removal      →  NLTK stopwords corpus
    - Punctuation Removal
    - POS Tagging           →  nltk.pos_tag()
    - Stemming              →  PorterStemmer  (NLTK)
    - Lemmatization         →  WordNetLemmatizer + POS tags  (NLTK lecture approach)
                            →  spaCy token.lemma_            (modern / batch-efficient)
    - Spell Checking        →  pyspellchecker  (in query_processor.py)

  NER_and_Sentiment_analysis.ipynb:
    - Named Entity Recognition  →  nltk.ne_chunk()  (lecture approach)
                                →  spaCy doc.ents   (modern approach)
    - Sentiment Analysis        →  VADER   (lightweight)
                                →  Transformers pipeline  (high-accuracy)

Pipelines
---------
Default (spaCy lemmatization — fast, recommended for indexing):
    normalize → nlp.pipe() → filter stopwords/punct/short tokens

NLTK lemmatization (exact lecture approach):
    normalize → word_tokenize → pos_tag → WordNetLemmatizer

Stemming:
    normalize → word_tokenize → remove_stopwords → PorterStemmer
"""

import re
import string
import unicodedata
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import nltk
import spacy

# ── NLTK downloads (silent, only if missing) ──────────────────────────────────
# Package names verified against NLTK 3.9.4

_NLTK_PACKAGES = [
    "punkt",                            # word_tokenize
    "stopwords",                        # stopwords corpus
    "wordnet",                          # WordNetLemmatizer
    "averaged_perceptron_tagger",       # pos_tag (NLTK < 3.9)
    "averaged_perceptron_tagger_eng",   # pos_tag (NLTK 3.9+)
    "maxent_ne_chunker",                # ne_chunk (NLTK < 3.9)
    "maxent_ne_chunker_tab",            # ne_chunk (NLTK 3.9+)
    "words",                            # required by ne_chunk
    "vader_lexicon",                    # VADER sentiment
]

for _pkg in _NLTK_PACKAGES:
    try:
        nltk.download(_pkg, quiet=True, raise_on_error=True)
    except Exception:
        pass   # already present or network unavailable — will fail gracefully later

from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

# ── spaCy models ──────────────────────────────────────────────────────────────

try:
    # Fast model: parser & NER disabled — used for batch lemmatization
    _nlp_fast = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found.\n"
        "Run: python -m spacy download en_core_web_sm"
    )

# Full model (lazy-loaded): used only for NER — avoids loading cost unless needed
_nlp_full: Optional[spacy.language.Language] = None


def _get_nlp_full() -> spacy.language.Language:
    """Lazy-load the full spaCy model (with NER enabled)."""
    global _nlp_full
    if _nlp_full is None:
        _nlp_full = spacy.load("en_core_web_sm")
    return _nlp_full


# ── Constants ─────────────────────────────────────────────────────────────────

_STOPWORDS    = set(stopwords.words("english"))
_stemmer      = PorterStemmer()
_lemmatizer   = WordNetLemmatizer()
MIN_TOKEN_LEN = 2
SPACY_BATCH   = 500


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Basic Text Operations  (from Text_processing.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    """
    Clean raw text before tokenization.

    Steps (in order):
      1. Guard against None / non-string input
      2. Lowercase
      3. Remove URLs
      4. Remove HTML tags and entities
      5. Normalize unicode → ASCII  (café → cafe)
      6. Remove remaining non-ASCII characters
      7. Remove all ASCII punctuation
      8. Remove standalone numeric tokens
      9. Collapse multiple whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text using NLTK word_tokenize (lecture approach).

    Handles contractions and punctuation more accurately than split():
        "don't"  →  ["do", "n't"]
        "U.S.A"  →  ["U.S.A"]
    """
    if not text or not text.strip():
        return []
    return word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove English stopwords (NLTK corpus)."""
    return [t for t in tokens if t.lower() not in _STOPWORDS]


def remove_punctuation_tokens(tokens: List[str]) -> List[str]:
    """Drop tokens that are purely punctuation (e.g. '.' ',' '--')."""
    return [t for t in tokens if not all(ch in string.punctuation for ch in t)]


# ── Stemming ──────────────────────────────────────────────────────────────────

def stem(tokens: List[str]) -> List[str]:
    """
    Apply Porter Stemmer to each token (NLTK — lecture approach).

    Example:  ["running", "leaves", "falling"]
           →  ["run",     "leav",   "fall"]
    """
    return [_stemmer.stem(t) for t in tokens]


# ── POS Tagging ───────────────────────────────────────────────────────────────

def get_pos_tags(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    POS-tag a token list using NLTK pos_tag (lecture approach).

    Returns list of (token, POS_tag) pairs.
    Example:  [("running", "VBG"), ("leaves", "NNS")]
    """
    return pos_tag(tokens)


def _wordnet_pos(nltk_tag: str) -> str:
    """
    Map NLTK POS tag to WordNet POS constant.
    Matches the get_wordnet_pos() helper shown in the lecture exactly.
    """
    tag = nltk_tag[0].upper()
    mapping = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return mapping.get(tag, wordnet.NOUN)


# ── Lemmatization — NLTK (lecture approach) ───────────────────────────────────

def lemmatize_nltk(tokens: List[str]) -> List[str]:
    """
    Lemmatize tokens using NLTK WordNetLemmatizer with POS tags (lecture approach).

    This is the method shown in Text_processing.ipynb:
        word_tokenize → pos_tag → WordNetLemmatizer.lemmatize(word, pos=...)

    Example:  ["boys", "are", "running", "leaves", "falling"]
           →  ["boy",  "be",  "run",     "leaf",   "fall"]
    """
    tagged = pos_tag(tokens)
    return [
        _lemmatizer.lemmatize(word, pos=_wordnet_pos(tag))
        for word, tag in tagged
    ]


# ── Lemmatization — spaCy (modern / batch-efficient) ─────────────────────────

def _spacy_tokens(doc) -> List[str]:
    """
    Extract clean lemmas from a spaCy Doc using token attributes.

    Fallback rule: if the strict filter removes ALL tokens (very short doc),
    relax the stopword constraint so at least content words survive.
    Example: "Side A" → strict gives [] → fallback gives ["side"]
    """
    strict = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and not token.like_num
        and token.is_alpha
        and len(token.lemma_) >= MIN_TOKEN_LEN
    ]
    if strict:
        return strict

    # Fallback: keep any alphabetic token, ignore stopword constraint
    fallback = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct
        and not token.is_space
        and not token.like_num
        and token.is_alpha
        and len(token.lemma_) >= MIN_TOKEN_LEN
    ]
    return fallback


def lemmatize_spacy(text: str) -> List[str]:
    """
    Lemmatize a single text string using spaCy (modern approach shown in lecture).
    For large-scale use, prefer preprocess_batch() which uses nlp.pipe().
    """
    return _spacy_tokens(_nlp_fast(text))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Named Entity Recognition  (from NER_and_Sentiment_analysis.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_entities_nltk(text: str) -> List[Tuple[str, str]]:
    """
    NER using NLTK ne_chunk (lecture approach).

    Pipeline: word_tokenize → pos_tag → ne_chunk → extract labelled subtrees.

    Returns list of (entity_text, entity_type) pairs.
    Example:  "Barack Obama was born in Hawaii"
           →  [("Barack Obama", "PERSON"), ("Hawaii", "GPE")]

    Note: Requires 'maxent_ne_chunker_tab' NLTK package.
    Falls back to spaCy NER if the NLTK chunker is unavailable.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    try:
        chunked = ne_chunk(tagged)
    except Exception:
        # chunker model not downloaded or corrupted — fall back to spaCy
        return extract_entities_spacy(text)

    entities = []
    for subtree in chunked:
        if hasattr(subtree, "label"):
            entity_text = " ".join(word for word, _ in subtree.leaves())
            entities.append((entity_text, subtree.label()))
    return entities


def extract_entities_spacy(text: str) -> List[Tuple[str, str]]:
    """
    NER using spaCy doc.ents (modern / industry approach shown in lecture).

    Returns list of (entity_text, entity_label) pairs.
    Example:  "Barack Obama was born in Hawaii"
           →  [("Barack Obama", "PERSON"), ("Hawaii", "GPE")]

    Common labels: PERSON, ORG, GPE (country/city), DATE, MONEY, LOC, PRODUCT
    """
    if not isinstance(text, str) or not text.strip():
        return []

    doc = _get_nlp_full()(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def extract_entities(text: str, method: str = "spacy") -> List[Tuple[str, str]]:
    """
    Named Entity Recognition — unified interface.

    Args:
        text:   Raw input text.
        method: "spacy" (default, more accurate) or "nltk" (lecture approach).

    Returns:
        List of (entity_text, entity_label) tuples.
    """
    if method == "nltk":
        return extract_entities_nltk(text)
    return extract_entities_spacy(text)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Sentiment Analysis  (from NER_and_Sentiment_analysis.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_sentiment_vader(text: str) -> Dict:
    """
    Sentiment analysis using VADER (lightweight — lecture approach).

    Returns dict with keys: neg, neu, pos, compound.
    compound score: -1 (most negative) → +1 (most positive)

    Example:  "I love sunny days at the beach!"
           →  {'neg': 0.0, 'neu': 0.407, 'pos': 0.593, 'compound': 0.807}
    """
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    return vader.polarity_scores(text)


def analyze_sentiment_transformers(text: str) -> Dict:
    """
    Sentiment analysis using HuggingFace Transformers (high-accuracy — lecture approach).

    Uses distilbert-base-uncased-finetuned-sst-2-english by default.
    Returns dict with keys: label (POSITIVE/NEGATIVE), score (confidence).

    Example:  "I love sunny days at the beach!"
           →  {'label': 'POSITIVE', 'score': 0.9998}
    """
    if not isinstance(text, str) or not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    try:
        from transformers import pipeline as hf_pipeline
        classifier = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        result = classifier(text[:512])[0]   # truncate to model max length
        return {"label": result["label"], "score": round(result["score"], 4)}
    except Exception as exc:
        return {"label": "ERROR", "score": 0.0, "error": str(exc)}


def analyze_sentiment(text: str, method: str = "vader") -> Dict:
    """
    Sentiment Analysis — unified interface.

    Args:
        text:   Raw input text.
        method: "vader" (default, fast) or "transformers" (more accurate).

    Returns:
        For vader:        {'neg': …, 'neu': …, 'pos': …, 'compound': …}
        For transformers: {'label': 'POSITIVE'/'NEGATIVE', 'score': …}
    """
    if method == "transformers":
        return analyze_sentiment_transformers(text)
    return analyze_sentiment_vader(text)


def get_sentiment_label(compound: float) -> str:
    """
    Convert VADER compound score to human-readable label.
    Thresholds follow the standard VADER convention.
    """
    if compound >= 0.05:
        return "POSITIVE"
    if compound <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Full Preprocessing Pipelines  (public API)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(
    text: str,
    use_stemming: bool = False,
    lemmatizer: str = "spacy",
) -> List[str]:
    """
    Preprocess a single document or query.

    Args:
        text:         Raw input string (may be None / empty).
        use_stemming: Use Porter Stemmer (overrides lemmatizer setting).
        lemmatizer:   "spacy" (default, fast) or "nltk" (WordNetLemmatizer+POS).

    Returns:
        List of clean tokens, [] for empty input.

    Pipelines:
        stemming  → normalize → word_tokenize → remove_stopwords → stem
        nltk      → normalize → word_tokenize → pos_tag → WordNetLemmatizer
        spacy     → normalize → nlp (lemma + stopword + punct filter in one pass)
    """
    normalized = normalize(text)
    if not normalized:
        return []

    if use_stemming:
        tokens = tokenize(normalized)
        tokens = remove_stopwords(tokens)
        tokens = remove_punctuation_tokens(tokens)
        tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LEN]
        return stem(tokens)

    if lemmatizer == "nltk":
        tokens = tokenize(normalized)
        tokens = remove_stopwords(tokens)
        tokens = remove_punctuation_tokens(tokens)
        tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LEN]
        return lemmatize_nltk(tokens)

    # Default: spaCy (fast, handles stopwords/punct internally)
    return lemmatize_spacy(normalized)


def preprocess_batch(
    texts: List[str],
    use_stemming: bool = False,
    lemmatizer: str = "spacy",
    batch_size: int = SPACY_BATCH,
) -> List[List[str]]:
    """
    Preprocess a list of texts efficiently.

    spaCy mode uses nlp.pipe() for batch efficiency on large corpora.
    NLTK mode processes texts one by one (no batch API available).

    Args:
        texts:        List of raw strings (None / empty entries handled).
        use_stemming: Porter Stemmer instead of lemmatization.
        lemmatizer:   "spacy" (default) or "nltk".
        batch_size:   spaCy pipe batch size.

    Returns:
        List[List[str]] — one token list per input text.
    """
    normalized = [normalize(t) for t in texts]

    if use_stemming:
        result = []
        for text in normalized:
            tokens = tokenize(text)
            tokens = remove_stopwords(tokens)
            tokens = remove_punctuation_tokens(tokens)
            tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LEN]
            result.append(stem(tokens))
        return result

    if lemmatizer == "nltk":
        result = []
        for text in normalized:
            tokens = tokenize(text)
            tokens = remove_stopwords(tokens)
            tokens = remove_punctuation_tokens(tokens)
            tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LEN]
            result.append(lemmatize_nltk(tokens))
        return result

    # spaCy batch (fastest for large corpora)
    placeholders = [t if t else " " for t in normalized]
    result = []
    for doc in _nlp_fast.pipe(placeholders, batch_size=batch_size):
        result.append(_spacy_tokens(doc))
    return result


def preprocess_stream(
    texts: Iterable[str],
    use_stemming: bool = False,
    lemmatizer: str = "spacy",
    batch_size: int = SPACY_BATCH,
) -> Iterator[List[str]]:
    """
    Generator version of preprocess_batch — memory-efficient for very large corpora.
    Yields one token list per input text without loading all into memory.
    """
    buffer: List[str] = []

    def _flush(buf: List[str]) -> Iterator[List[str]]:
        yield from preprocess_batch(
            buf,
            use_stemming=use_stemming,
            lemmatizer=lemmatizer,
            batch_size=batch_size,
        )

    for text in texts:
        buffer.append(text)
        if len(buffer) >= batch_size:
            yield from _flush(buffer)
            buffer = []

    if buffer:
        yield from _flush(buffer)
