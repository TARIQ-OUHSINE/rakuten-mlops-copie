"""Tests du monitoring : /metrics, metriques custom, journal JSONL."""

import json

from fastapi.testclient import TestClient

from src.api.main import app, state


class FakePredictor:
    version = "test"
    model_source = "local"

    def predict(self, designation, description=None, top_k=3):
        return [{"prdtypecode": 10, "label_idx": 0, "proba": 0.9}]


def make_client():
    return TestClient(app)


def test_metrics_endpoint_exposes_http_metrics():
    with make_client() as client:
        state["predictor"] = FakePredictor()
        client.get("/health")
        r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_request_duration_seconds" in r.text


def test_prediction_metrics_incremented():
    with make_client() as client:
        state["predictor"] = FakePredictor()
        client.post("/predict", json={"designation": "livre"})
        r = client.get("/metrics")
    assert 'rakuten_predictions_total{prdtypecode="10"}' in r.text
    assert "rakuten_prediction_confidence" in r.text


def test_predictions_logged_to_jsonl(tmp_path, monkeypatch):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("PREDICTIONS_LOG_PATH", str(log_path))
    with make_client() as client:
        state["predictor"] = FakePredictor()
        client.post(
            "/predict",
            json={"designation": "Console Nintendo", "description": "jeu video"},
        )
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["prdtypecode"] == 10
    assert entry["proba"] == 0.9
    assert entry["designation_chars"] == len("Console Nintendo")
    assert "timestamp" in entry


def test_no_jsonl_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("PREDICTIONS_LOG_PATH", raising=False)
    with make_client() as client:
        state["predictor"] = FakePredictor()
        r = client.post("/predict", json={"designation": "livre"})
    assert r.status_code == 200  # logging desactive, pas d'erreur
