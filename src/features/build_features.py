"""Vectorisation TF-IDF du texte nettoye (params dans params.yaml)."""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import load_params


def make_vectorizer(params: dict | None = None) -> TfidfVectorizer:
    cfg = params or load_params()["tfidf"]
    return TfidfVectorizer(
        max_features=cfg["max_features"],
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=cfg["min_df"],
        sublinear_tf=cfg["sublinear_tf"],
        strip_accents=cfg["strip_accents"],
    )


def fit_transform_texts(train_texts, val_texts, params: dict | None = None):
    """Fit sur train uniquement (pas de fuite), transform du val."""
    vectorizer = make_vectorizer(params)
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    return vectorizer, x_train, x_val


def save_vectorizer(vectorizer: TfidfVectorizer, path) -> None:
    joblib.dump(vectorizer, path)


def load_vectorizer(path) -> TfidfVectorizer:
    return joblib.load(path)
