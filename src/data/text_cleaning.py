"""Nettoyage texte partage entre le pipeline de donnees et l'inference.

Logique extraite de notebooks/main_notebooks/01_preprocessing_eda.ipynb :
clean HTML -> minuscules -> suppression non-alpha -> stop words -> stemming FR.
"""

import html
import re
import unicodedata
from functools import lru_cache

import pandas as pd

_HTML_TAG = re.compile(r"<[^>]+>", re.IGNORECASE)
_MULTIPLE_SPC = re.compile(r"\s+")
_NON_ALPHA = re.compile(r"[^a-z\s]")

# Stopwords specifiques marketplace, en complement de NLTK
_EXTRA_STOPWORDS = {
    "cm", "mm", "ml", "kg", "g", "l", "n", "x", "ref", "ref.", "lot",
    "pack", "set", "new", "neuf", "article", "produit", "product",
}


def _strip_accents(text: str) -> str:
    """Normalise les accents vers leur base ASCII (e -> e, c -> c...).

    NFKD decompose une lettre accentuee en base + diacritique combinant ; on
    supprime les diacritiques. Doit passer AVANT le filtre [^a-z\\s], sinon les
    lettres accentuees sont supprimees et tronquent les mots francais
    (« magnetisme » perdu). Modele v2 : necessite un re-entrainement complet.
    """
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset:
    from nltk.corpus import stopwords

    try:
        stopwords.words("french")
    except LookupError:
        raise RuntimeError(
            "Stopwords NLTK introuvables — lancer d'abord `make setup` "
            "(python -m nltk.downloader stopwords)"
        ) from None

    # de-accentuer le set : les tokens sont de-accentues avant comparaison, donc
    # « même » -> « meme » doit matcher un stopword « meme » (sinon il fuit dans
    # le vocabulaire). Coherence indispensable avec _strip_accents cote texte.
    all_stopwords = {_strip_accents(w) for w in _EXTRA_STOPWORDS}
    for lang in ("french", "english", "german", "spanish"):
        all_stopwords.update(_strip_accents(w) for w in stopwords.words(lang))
    return frozenset(all_stopwords)


@lru_cache(maxsize=1)
def _get_stemmer():
    from nltk.stem.snowball import SnowballStemmer

    return SnowballStemmer("french")


def clean_html(text) -> str:
    """Supprime les balises HTML, decode les entites, normalise les espaces.

    Les entites sont DECODEES (html.unescape) et non supprimees : « int&eacute;gr&eacute; »
    -> « intégré » (dé-accentué ensuite en « integre »). Les blanchir tronquait les
    mots dont les accents sont encodes en HTML.
    """
    if text is None or pd.isna(text):
        return ""
    text = _HTML_TAG.sub(" ", str(text))
    text = html.unescape(text)  # &eacute; -> é, &#233; -> é, &amp; -> &, &nbsp; -> \xa0
    text = _HTML_TAG.sub(" ", text)  # 2e passe : balises encodees (&lt;b&gt;) devenues reelles
    return _MULTIPLE_SPC.sub(" ", text).strip()


def preprocess_text(text, stem: bool = True) -> str:
    """Pipeline v2 : minuscules -> de-accentuation (NFKD) -> non-alpha ->
    stop words (de-accentues) -> stemming FR."""
    if text is None or pd.isna(text) or not str(text).strip():
        return ""
    stopwords_set = _get_stopwords()
    text = str(text).lower()
    text = _strip_accents(text)  # v2 : de-accentue avant le filtre non-alpha
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
