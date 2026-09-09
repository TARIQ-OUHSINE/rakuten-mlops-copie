"""Fixtures partagees : mini-dataset committe + modele jouet entraine en session.

Les tests n'ont pas besoin des 52 MB de donnees brutes : tests/fixtures/sample.csv
contient ~100 lignes echantillonnees (text brut + prdtypecode).
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.data.text_cleaning import build_text_combined
from src.features.build_features import make_vectorizer

FIXTURES_DIR = Path(__file__).parent / "fixtures"

TINY_TFIDF = {
    "max_features": 500,
    "ngram_range": [1, 1],
    "min_df": 1,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES_DIR / "sample.csv")
    df["text_combined"] = [
        build_text_combined(d, desc)
        for d, desc in zip(df["designation"], df["description"])
    ]
    return df


@pytest.fixture(scope="session")
def model_dir(sample_df, tmp_path_factory) -> Path:
    """Entraine un modele jouet (memes artefacts que train_model) dans un tmp dir."""
    out = tmp_path_factory.mktemp("baseline")

    vectorizer = make_vectorizer(TINY_TFIDF)
    x = vectorizer.fit_transform(sample_df["text_combined"])
    le = LabelEncoder()
    y = le.fit_transform(sample_df["prdtypecode"])

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        n_estimators=10,
        num_leaves=7,
        min_child_samples=1,
        random_state=42,
        verbose=-1,
    )
    clf.fit(x, y)

    joblib.dump(clf, out / "model.joblib")
    joblib.dump(vectorizer, out / "vectorizer.joblib")
    joblib.dump(le, out / "label_encoder.joblib")
    return out
