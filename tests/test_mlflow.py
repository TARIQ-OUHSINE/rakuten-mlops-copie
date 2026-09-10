"""Tests de l'integration MLflow (tracking, registry, fallback) — sans serveur.

Le tracking est teste sur un file store temporaire (tmp_path), jamais sur un
serveur : la suite doit passer hors ligne et en CI.
"""

import json

import pytest

from src.models.mlflow_utils import PRODUCTION_ALIAS, log_training_run
from src.models.predict_model import BaselinePredictor
from tests.conftest import TINY_TFIDF

MODEL_NAME = "rakuten-test"

FAKE_METRICS = {
    "f1_weighted": 0.81,
    "f1_macro": 0.79,
    "accuracy": 0.80,
    "report": {"0": {"f1-score": 0.9}},
}


def make_params(model_dir, tracking_uri) -> dict:
    return {
        "model": {"dir": str(model_dir), "version": "test-v1"},
        "mlflow": {
            "tracking_uri": str(tracking_uri),
            "experiment_name": "test-exp",
            "registered_model_name": MODEL_NAME,
        },
        "tfidf": TINY_TFIDF,
        "lgbm": {"n_estimators": 10},
        "early_stopping_rounds": 5,
    }


@pytest.fixture()
def tracked_model(model_dir, tmp_path, monkeypatch):
    """Log un run complet vers un file store temporaire, retourne (params, uri)."""
    tracking_uri = f"sqlite:///{tmp_path}/mlruns.db"
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    params = make_params(model_dir, tracking_uri)
    (model_dir / "metrics.json").write_text(json.dumps(FAKE_METRICS))
    log_training_run(model_dir, FAKE_METRICS, params)
    return params, str(tracking_uri)


def test_run_logged_and_model_registered(tracked_model):
    import mlflow
    from mlflow import MlflowClient

    params, tracking_uri = tracked_model
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name("test-exp")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.metrics["f1_weighted"] == pytest.approx(0.81)
    assert run.data.params["lgbm_n_estimators"] == "10"

    version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    assert str(version.version) == "1"


def test_alias_bootstrap_only_on_first_version(tracked_model, model_dir):
    import mlflow
    from mlflow import MlflowClient

    params, tracking_uri = tracked_model
    log_training_run(model_dir, FAKE_METRICS, params)  # version 2

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    alias_version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    assert str(alias_version.version) == "1"
    assert client.get_model_version(MODEL_NAME, "2") is not None


def test_predictor_loads_from_registry(tracked_model, monkeypatch):
    params, tracking_uri = tracked_model
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr("src.models.predict_model.load_params", lambda: params)

    predictor = BaselinePredictor()
    assert predictor.model_source == "mlflow:v1"
    assert predictor.version == f"{MODEL_NAME}-v1"
    preds = predictor.predict("Console Nintendo Switch OLED", top_k=3)
    assert len(preds) == 3


def test_predictor_falls_back_to_local_when_mlflow_unreachable(
    model_dir, monkeypatch
):
    params = make_params(model_dir, "unused")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:1")
    monkeypatch.setattr("src.models.predict_model.load_params", lambda: params)

    predictor = BaselinePredictor()
    assert predictor.model_source == "local"
    assert predictor.version == "test-v1"
    assert predictor.predict("Livre roman poche", top_k=1)


def test_predictor_ignores_mlflow_when_model_dir_given(model_dir, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:1")
    predictor = BaselinePredictor(model_dir=model_dir)
    assert predictor.model_source == "local"
