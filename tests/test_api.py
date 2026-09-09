"""Tests de l'API FastAPI (src/api/main.py) via TestClient."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, state
from src.models.predict_model import BaselinePredictor


@pytest.fixture()
def client(model_dir):
    state["predictor"] = BaselinePredictor(model_dir=model_dir)
    with TestClient(app) as c:
        state["predictor"] = BaselinePredictor(model_dir=model_dir)
        yield c
    state["predictor"] = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_happy_path(client):
    r = client.post(
        "/predict",
        json={"designation": "Console Nintendo Switch OLED", "description": None},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3
    assert body["model_version"] == "baseline-tfidf-lgbm-v1"
    top = body["predictions"][0]
    assert {"prdtypecode", "label_idx", "proba"} <= set(top)


def test_predict_top_k(client):
    r = client.post("/predict", json={"designation": "Livre roman", "top_k": 5})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 5


def test_predict_missing_designation(client):
    r = client.post("/predict", json={"description": "sans titre"})
    assert r.status_code == 422


def test_predict_empty_designation(client):
    r = client.post("/predict", json={"designation": "   "})
    assert r.status_code == 422


def test_predict_model_not_loaded(model_dir):
    state["predictor"] = None
    with TestClient(app) as c:
        state["predictor"] = None
        r = c.post("/predict", json={"designation": "Console"})
    assert r.status_code == 503
