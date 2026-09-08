"""Tests de l'inference (src/models/predict_model.py)."""

import joblib
import numpy as np
import pytest

from src.models.predict_model import BaselinePredictor


@pytest.fixture(scope="module")
def predictor(model_dir) -> BaselinePredictor:
    return BaselinePredictor(model_dir=model_dir)


def test_predict_output_structure(predictor):
    preds = predictor.predict("Console Nintendo Switch OLED", top_k=3)
    assert len(preds) == 3
    for p in preds:
        assert set(p) == {"prdtypecode", "label_idx", "proba"}
        assert 0.0 <= p["proba"] <= 1.0


def test_predictions_sorted_by_proba(predictor):
    preds = predictor.predict("Coussin decoratif salon", top_k=5)
    probas = [p["proba"] for p in preds]
    assert probas == sorted(probas, reverse=True)


def test_full_probas_sum_to_one(predictor, model_dir):
    le = joblib.load(model_dir / "label_encoder.joblib")
    preds = predictor.predict("Livre roman poche", top_k=len(le.classes_))
    assert np.isclose(sum(p["proba"] for p in preds), 1.0, atol=1e-2)


def test_prdtypecode_in_known_classes(predictor, model_dir):
    le = joblib.load(model_dir / "label_encoder.joblib")
    preds = predictor.predict("Piscine gonflable enfant", top_k=3)
    known = {int(c) for c in le.classes_}
    assert all(p["prdtypecode"] in known for p in preds)


def test_description_changes_input(predictor):
    without = predictor.predict("Console", top_k=1)
    with_desc = predictor.predict("Console", "jeu video portable ecran tactile", top_k=1)
    assert without and with_desc  # les deux chemins fonctionnent
