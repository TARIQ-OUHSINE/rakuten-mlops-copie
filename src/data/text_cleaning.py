"""Nettoyage texte partage entre le pipeline de donnees et l'inference.

Logique extraite de notebooks/main_notebooks/01_preprocessing_eda.ipynb :
clean HTML -> minuscules -> suppression non-alpha -> stop words -> stemming FR.
"""

import re
from functools import lru_cache

import pandas as pd

_HTML_TAG = re.compile(r"<[^>]+>", re.IGNORECASE)
_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_MULTIPLE_SPC = re.compile(r"\s+")
_NON_ALPHA = re.compile(r"[^a-z\s]")

# Stopwords specifiques marketplace, en complement de NLTK
_EXTRA_STOPWORDS = {
    "cm", "mm", "ml", "kg", "g", "l", "n", "x", "ref", "ref.", "lot",
    "pack", "set", "new", "neuf", "article", "produit", "product",
}


@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset:
    from nltk.corpus import stopwords

    try:
        stopwords.words("french")
    except LookupError:
        import nltk

        nltk.download("stopwords", quiet=True)

    all_stopwords = set(_EXTRA_STOPWORDS)
    for lang in ("french", "english", "german", "spanish"):
        all_stopwords.update(stopwords.words(lang))
    return frozenset(all_stopwords)


@lru_cache(maxsize=1)
def _get_stemmer():
    from nltk.stem.snowball import SnowballStemmer

    return SnowballStemmer("french")


def clean_html(text) -> str:
    """Supprime les balises HTML et entites, normalise les espaces."""
    if text is None or pd.isna(text):
        return ""
    text = _HTML_TAG.sub(" ", str(text))
    text = _HTML_ENTITY.sub(" ", text)
    return _MULTIPLE_SPC.sub(" ", text).strip()


def preprocess_text(text, stem: bool = True) -> str:
    """Pipeline baseline : minuscules -> non-alpha -> stop words -> stemming FR."""
    if text is None or pd.isna(text) or not str(text).strip():
        return ""
    stopwords_set = _get_stopwords()
    text = str(text).lower()
    text = _NON_ALPHA.sub(" ", text)
    tokens = [t for t in text.split() if t not in stopwords_set and len(t) > 2]
    if stem:
        stemmer = _get_stemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


def build_text_combined(designation, description) -> str:
    """Texte final pour TF-IDF : designation + description nettoyees."""
    parts = [
        preprocess_text(clean_html(designation)),
        preprocess_text(clean_html(description)),
    ]
    return " ".join(p for p in parts if p).strip()
