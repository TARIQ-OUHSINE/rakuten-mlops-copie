"""Tests du monitoring : /metrics, metriques custom, journal JSONL, derive."""

import json

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app, state
from src.monitoring.drift_report import build_reference, evaluate_drift, load_current


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
    # longueurs mesurees sur le texte combine nettoye (meme repr. que la reference)
    assert entry["text_chars"] > 0
    assert entry["text_words"] > 0
    assert "timestamp" in entry


def test_no_jsonl_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("PREDICTIONS_LOG_PATH", raising=False)
    with make_client() as client:
        state["predictor"] = FakePredictor()
        r = client.post("/predict", json={"designation": "livre"})
    assert r.status_code == 200  # logging desactive, pas d'erreur


def _fake_frame(chars=50, words=8, code=10, proba=0.9, n=50):
    # legere variance pour que les tests statistiques aient une distribution
    return pd.DataFrame({
        "text_chars": [chars + i % 5 for i in range(n)],
        "text_words": [words + i % 3 for i in range(n)],
        "prdtypecode": [code + (i % 2) * 30 for i in range(n)],
        "proba": [round(proba - (i % 4) * 0.02, 2) for i in range(n)],
    })


def test_no_drift_on_identical_data(tmp_path):
    ref = _fake_frame()
    result = evaluate_drift(ref, ref.copy(), reports_dir=tmp_path)
    assert result["drift_detected"] is False
    assert result["drift_share"] == 0.0
    assert (tmp_path / result["report_name"]).exists()


def test_drift_detected_on_shifted_data(tmp_path):
    ref = _fake_frame(chars=200, words=30, proba=0.9)
    cur = _fake_frame(chars=15, words=3, proba=0.4)
    result = evaluate_drift(ref, cur, reports_dir=tmp_path)
    assert result["drift_detected"] is True
    assert result["drift_share"] > 0.3


def test_build_reference_shape(model_dir, sample_df):
    ref = build_reference(sample_df.head(10), model_dir=model_dir)
    assert set(ref.columns) == {
        "text_chars", "text_words", "prdtypecode", "proba",
    }
    assert len(ref) == 10


def test_load_current_empty_file_returns_empty_frame(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text("")  # journal vide : pas de KeyError
    df = load_current(log_path)
    assert list(df.columns) == ["text_chars", "text_words", "prdtypecode", "proba"]
    assert len(df) == 0


def test_load_current_window_keeps_last_n(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text("\n".join(
        json.dumps({"text_chars": i, "text_words": i, "prdtypecode": 10, "proba": 0.5})
        for i in range(10)
    ))
    df = load_current(log_path, window=3)
    assert len(df) == 3
    assert list(df["text_chars"]) == [7, 8, 9]
