"""Tests de l'auth par cle d'API (X-API-Key)."""

from fastapi.testclient import TestClient

from src.api.main import app, state


class FakePredictor:
    version = "test"
    model_source = "local"

    def predict(self, designation, description=None, top_k=3):
        return [{"prdtypecode": 10, "label_idx": 0, "proba": 1.0}]


class _ClientWithFake:
    """TestClient dont le predictor est injecte APRES le lifespan (qui ecrase state)."""

    def __enter__(self):
        self._client = TestClient(app)
        self._client.__enter__()
        state["predictor"] = FakePredictor()
        return self._client

    def __exit__(self, *exc):
        state["predictor"] = None
        return self._client.__exit__(*exc)


def make_client():
    return _ClientWithFake()


def test_predict_requires_key_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    with make_client() as client:
        r = client.post("/predict", json={"designation": "livre"})
    assert r.status_code == 401


def test_predict_accepts_valid_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    with make_client() as client:
        r = client.post(
            "/predict",
            json={"designation": "livre"},
            headers={"X-API-Key": "s3cret"},
        )
    assert r.status_code == 200


def test_predict_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    with make_client() as client:
        r = client.post(
            "/predict",
            json={"designation": "livre"},
            headers={"X-API-Key": "mauvaise"},
        )
    assert r.status_code == 401


def test_predict_open_when_no_key_configured(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with make_client() as client:
        r = client.post("/predict", json={"designation": "livre"})
    assert r.status_code == 200


def test_health_never_requires_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    with make_client() as client:
        r = client.get("/health")
    assert r.status_code == 200
