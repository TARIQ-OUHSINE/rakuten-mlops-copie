"""Tests de la vectorisation TF-IDF (src/features/build_features.py)."""

import numpy as np

from src.features.build_features import (
    fit_transform_texts,
    load_vectorizer,
    make_vectorizer,
    save_vectorizer,
)
from tests.conftest import TINY_TFIDF


def test_fit_transform_shapes(sample_df):
    texts = sample_df["text_combined"]
    vectorizer, x_train, x_val = fit_transform_texts(texts[:80], texts[80:], TINY_TFIDF)
    assert x_train.shape[0] == 80
    assert x_val.shape[0] == len(texts) - 80
    assert x_train.shape[1] == x_val.shape[1] <= TINY_TFIDF["max_features"]


def test_vectorizer_persistence_roundtrip(sample_df, tmp_path):
    vectorizer = make_vectorizer(TINY_TFIDF)
    x_before = vectorizer.fit_transform(sample_df["text_combined"])

    path = tmp_path / "vectorizer.joblib"
    save_vectorizer(vectorizer, path)
    reloaded = load_vectorizer(path)

    x_after = reloaded.transform(sample_df["text_combined"])
    assert vectorizer.vocabulary_ == reloaded.vocabulary_
    assert np.allclose(x_before.toarray(), x_after.toarray())
