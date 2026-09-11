# Phase 4 Plan — Monitoring & Maintenance (MLOps Rakuten)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Deadline : 23/10** (soutenance 06/11) · Branche : `feat/phase4` (→ PR vers `feat/dev-tyu`) · Pré-requis : Phase 3 mergée (commit `7b35ca1`)

**Goal :** surveiller l'API en production (latence, débit, distribution des prédictions),
détecter la dérive des données avec Evidently, alerter quand elle dépasse un seuil, et
documenter le workflow de maintenance — le ré-entraînement restant une commande unique.

**Architecture :** l'API expose `/metrics` (prometheus-fastapi-instrumentator + métriques
custom) et journalise chaque prédiction en JSONL sur un volume partagé ; Prometheus
scrape les replicas api en direct (DNS compose) ; Grafana est provisionné par code
(datasource + dashboard committés) ; un job Evidently one-shot (profile compose `drift`)
compare les prédictions récentes à une référence issue du jeu de validation, produit un
rapport HTML et pousse son score vers Prometheus via un pushgateway.

**Tech stack :** prometheus-fastapi-instrumentator 8.1.0, prometheus-client 0.26.0,
evidently 0.7.23, prom/prometheus, grafana/grafana-oss, prom/pushgateway.

## Context

Les Phases 1–3 ont livré un pipeline reproductible, du tracking/versioning et une API
sécurisée, scalée et testée en CI. Mais une fois déployée, l'API est une boîte noire :
aucune métrique d'exploitation, aucune trace des prédictions servies, aucune détection
de dérive — et le kick-off Phase 4 demande exactement cela (Prometheus/Grafana,
Evidently, mise à jour du modèle, documentation).

**Décisions confirmées avec l'utilisateur (11/09) :**
- Collecte des prédictions : **JSONL local** sur volume partagé (simple, multi-replica ok).
- Ré-entraînement : **alerte + commande unique** (`make pipeline`) — human-in-the-loop,
  pas de retrain automatique (faux positifs, 40 min de train, mémoire VM).
- Monitoring : **métriques API + dashboard Grafana provisionné + métrique de drift
  exposée** (pushgateway). Pas d'Alertmanager (seuils visuels Grafana suffisent).

## Architecture cible

```
client ──:8000──▶ nginx ──▶ api xN ──/metrics──▶ prometheus (:9090) ──▶ grafana (:3000)
                             │                        ▲                   dashboard provisionné
                             ▼                        │
                   monitoring_data/predictions.jsonl  │
                             │                        │
              docker compose --profile drift run drift
              (Evidently : référence val.parquet vs prédictions récentes)
                             │                        │
                             ├─ rapport HTML ─▶ monitoring_data/reports/
                             └─ drift metrics ─▶ pushgateway (:9091) ─┘
```

## Fichiers

- Create : `src/api/metrics.py` (instrumentator + métriques custom + logging JSONL)
- Create : `src/monitoring/__init__.py`, `src/monitoring/drift_report.py` (job Evidently)
- Create : `src/monitoring/send_traffic.py` (générateur de trafic de démo)
- Create : `docker/prometheus.yml`, `docker/grafana/provisioning/datasources/datasource.yml`,
  `docker/grafana/provisioning/dashboards/dashboards.yml`, `docker/grafana/dashboards/rakuten-api.json`
- Create : `tests/test_monitoring.py`
- Create : `docs/phase4_summary.md`
- Modify : `src/api/main.py` (branchement instrumentator + log des prédictions),
  `requirements.txt` (+instrumentator), `requirements-dev.txt` (+evidently),
  `Dockerfile` (stage `monitoring`), `docker-compose.yml` (+prometheus, grafana,
  pushgateway, drift), `params.yaml` (section `monitoring`), `Makefile`
  (`traffic`, `drift-report`), `.gitignore` (+`/monitoring_data/`), `README.md`

---

## Stage 1 — Métriques Prometheus dans l'API

### Task 1.1 : /metrics + métriques custom

**Files :** Create `src/api/metrics.py`, Modify `src/api/main.py`, Modify `requirements.txt`, Test `tests/test_monitoring.py`

- [ ] **Ajouter la dépendance** — `requirements.txt` :

```
prometheus-fastapi-instrumentator==8.1.0
```

(`prometheus-client` vient avec.) Puis `.venv/bin/pip install -r requirements-dev.txt`.

- [ ] **Écrire les tests (rouges)** — `tests/test_monitoring.py` :

```python
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
```

- [ ] **Vérifier l'échec** : `pytest tests/test_monitoring.py -v` → 404 sur /metrics, KeyError…
- [ ] **Implémenter** — `src/api/metrics.py` :

```python
"""Metriques Prometheus et journal des predictions (Phase 4).

- instrumentator : latence/debit/status par endpoint (+ /metrics)
- metriques custom : volume par classe predite, distribution de confiance
- journal JSONL : une ligne par prediction si PREDICTIONS_LOG_PATH est defini
  (desactive sinon — dev local et tests Phase 1-3 inchanges)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

PREDICTIONS_TOTAL = Counter(
    "rakuten_predictions_total",
    "Nombre de predictions servies, par classe predite (top-1)",
    ["prdtypecode"],
)
PREDICTION_CONFIDENCE = Histogram(
    "rakuten_prediction_confidence",
    "Probabilite de la classe top-1",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


def setup_metrics(app) -> None:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def observe_prediction(top: dict) -> None:
    PREDICTIONS_TOTAL.labels(prdtypecode=str(top["prdtypecode"])).inc()
    PREDICTION_CONFIDENCE.observe(top["proba"])


def log_prediction(designation: str, description: str | None, top: dict) -> None:
    log_path = os.environ.get("PREDICTIONS_LOG_PATH")
    if not log_path:
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "designation_chars": len(designation),
        "designation_words": len(designation.split()),
        "description_chars": len(description) if description else 0,
        "prdtypecode": top["prdtypecode"],
        "proba": top["proba"],
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_APPEND : ecriture atomique de petites lignes, ok multi-replicas
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
```

- [ ] **Brancher dans `src/api/main.py`** :

```python
from src.api.metrics import log_prediction, observe_prediction, setup_metrics

# apres la creation de `app` et des middlewares :
setup_metrics(app)

# dans predict(), apres le calcul de `predictions` :
    top = predictions[0]
    observe_prediction(top)
    log_prediction(request.designation, request.description, top)
```

- [ ] **Vérifier** : `make lint && make test` → tout vert (les 4 nouveaux tests aussi)
- [ ] **Commit** : `git commit -m "Phase 4: /metrics Prometheus + journal JSONL des predictions"`

**Exemple de sortie attendue :**
```
$ curl -s localhost:8000/metrics | grep rakuten
rakuten_predictions_total{prdtypecode="10"} 1.0
rakuten_prediction_confidence_bucket{le="0.9"} 1.0
```

## Stage 2 — Générateur de trafic de démo

### Task 2.1 : `make traffic`

**Files :** Create `src/monitoring/__init__.py` (vide), Create `src/monitoring/send_traffic.py`, Modify `Makefile`

- [ ] **Écrire** `src/monitoring/send_traffic.py` :

```python
"""Envoie du trafic de demo a l'API depuis le jeu de validation.

Usage : python -m src.monitoring.send_traffic [N] [--drift]
--drift : tronque les textes a 3 mots pour simuler une derive de distribution.
"""

import os
import sys

import httpx
import pandas as pd

from src.config import load_params, resolve

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    drift = "--drift" in sys.argv
    val = pd.read_parquet(
        resolve(load_params()["data"]["processed_dir"]) / "val.parquet"
    ).sample(n, random_state=None if drift else 42)

    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    ok = 0
    with httpx.Client(timeout=30) as client:
        for text in val["text_combined"]:
            payload = " ".join(str(text).split()[:3]) if drift else str(text)
            r = client.post(
                f"{API_URL}/predict",
                json={"designation": payload or "produit"},
                headers=headers,
            )
            ok += r.status_code == 200
    print(f"{ok}/{n} predictions ok ({'drift simule' if drift else 'normal'})")


if __name__ == "__main__":
    main()
```

- [ ] **Makefile** :

```makefile
traffic:          ## Envoie N=100 requetes de demo (API_KEY/API_URL via env)
	$(BIN)/python -m src.monitoring.send_traffic $(N)
```

(ajouter `traffic` au `.PHONY`)

- [ ] **Vérifier** : `make lint` ; test manuel au Stage 3 (nécessite l'API up).
- [ ] **Commit** : `git commit -m "Phase 4: generateur de trafic de demo (make traffic)"`

## Stage 3 — Stack monitoring compose (Prometheus, Grafana, pushgateway)

### Task 3.1 : configuration Prometheus + Grafana provisionné

**Files :** Create `docker/prometheus.yml`, `docker/grafana/provisioning/datasources/datasource.yml`,
`docker/grafana/provisioning/dashboards/dashboards.yml`, `docker/grafana/dashboards/rakuten-api.json`,
Modify `docker-compose.yml`, `Makefile`, `.gitignore`

- [ ] **`docker/prometheus.yml`** — scrape les replicas api en direct (DNS compose),
  pas via nginx :

```yaml
global:
  scrape_interval: 10s

scrape_configs:
  - job_name: api
    dns_sd_configs:
      - names: ["api"]
        type: A
        port: 8000
  - job_name: pushgateway
    honor_labels: true
    static_configs:
      - targets: ["pushgateway:9091"]
```

- [ ] **`docker/grafana/provisioning/datasources/datasource.yml`** :

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

- [ ] **`docker/grafana/provisioning/dashboards/dashboards.yml`** :

```yaml
apiVersion: 1
providers:
  - name: rakuten
    folder: ""
    type: file
    options:
      path: /var/lib/grafana/dashboards
```

- [ ] **`docker/grafana/dashboards/rakuten-api.json`** — dashboard minimal (5 panneaux) :

```json
{
  "title": "Rakuten API — MLOps",
  "uid": "rakuten-api",
  "timezone": "browser",
  "refresh": "10s",
  "time": {"from": "now-30m", "to": "now"},
  "panels": [
    {"id": 1, "type": "timeseries", "title": "Requetes /predict par seconde",
     "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
     "targets": [{"expr": "sum(rate(http_requests_total{handler=\"/predict\"}[1m]))"}]},
    {"id": 2, "type": "timeseries", "title": "Latence p95 (s)",
     "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
     "targets": [{"expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{handler=\"/predict\"}[5m])) by (le))"}]},
    {"id": 3, "type": "piechart", "title": "Predictions par classe",
     "gridPos": {"x": 0, "y": 8, "w": 8, "h": 8},
     "targets": [{"expr": "sum(rakuten_predictions_total) by (prdtypecode)"}]},
    {"id": 4, "type": "timeseries", "title": "Confiance mediane (top-1)",
     "gridPos": {"x": 8, "y": 8, "w": 8, "h": 8},
     "targets": [{"expr": "histogram_quantile(0.5, sum(rate(rakuten_prediction_confidence_bucket[5m])) by (le))"}]},
    {"id": 5, "type": "stat", "title": "Drift share (Evidently)",
     "gridPos": {"x": 16, "y": 8, "w": 8, "h": 8},
     "fieldConfig": {"defaults": {"thresholds": {"steps": [
       {"color": "green", "value": null},
       {"color": "red", "value": 0.3}]}}},
     "targets": [{"expr": "rakuten_drift_share"}]}
  ],
  "schemaVersion": 39
}
```

- [ ] **`docker-compose.yml`** — trois services + volume partagé sur l'api :

```yaml
  prometheus:
    image: prom/prometheus:v3.2.1
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana-oss:11.5.2
    environment:
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Admin
    volumes:
      - ./docker/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./docker/grafana/dashboards:/var/lib/grafana/dashboards:ro
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

  pushgateway:
    image: prom/pushgateway:v1.11.0
    ports:
      - "9091:9091"
```

et sur le service `api`, ajouter :

```yaml
    environment:
      PREDICTIONS_LOG_PATH: /app/monitoring_data/predictions.jsonl
    volumes:
      - ./monitoring_data:/app/monitoring_data
```

NB : vérifier les derniers tags des 3 images à l'implémentation
(`prom/prometheus`, `grafana/grafana-oss`, `prom/pushgateway` sur Docker Hub) —
leçon Phase 3 : un tag inexistant n'échoue qu'à l'exécution.

- [ ] **`.gitignore`** :

```
# Monitoring (Phase 4) : donnees locales prometheus + journal de predictions
/prometheus_data/
/monitoring_data/
```

- [ ] **Vérifier en local** :

```
$ docker compose config --quiet && API_KEY=demo make up
$ API_KEY=demo make traffic N=50
$ curl -s localhost:9090/api/v1/targets | grep -o '"health":"up"' | wc -l   # >= 3 (2 api + pushgateway)
$ open http://localhost:3000/d/rakuten-api    # dashboard peuple sans configuration
```

- [ ] **Commit** : `git commit -m "Phase 4: stack monitoring compose (prometheus, grafana provisionne, pushgateway)"`

## Stage 4 — Job de dérive Evidently

### Task 4.1 : `src/monitoring/drift_report.py`

**Files :** Create `src/monitoring/drift_report.py`, Modify `params.yaml`, `requirements-dev.txt`,
`Dockerfile` (stage monitoring), `docker-compose.yml` (service drift), `Makefile`, Test `tests/test_monitoring.py`

- [ ] **params.yaml** — section monitoring :

```yaml
# Monitoring & derive (Phase 4)
monitoring:
  predictions_log: monitoring_data/predictions.jsonl
  reports_dir: monitoring_data/reports
  drift_share_threshold: 0.3      # part de colonnes derivees declenchant l'alerte
  min_current_rows: 20            # en dessous : pas assez de trafic pour statuer
```

- [ ] **requirements-dev.txt** : ajouter `evidently==0.7.23` (le job tourne dans
  l'image `monitoring`, pas dans l'API).

- [ ] **Tests (rouges)** — ajouter à `tests/test_monitoring.py` :

```python
import pandas as pd
import pytest

from src.monitoring.drift_report import build_reference, evaluate_drift


def _fake_frame(chars=50, words=8, code=10, proba=0.9, n=50):
    return pd.DataFrame({
        "designation_chars": [chars] * n,
        "designation_words": [words] * n,
        "prdtypecode": [code] * n,
        "proba": [proba] * n,
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
    ref = build_reference(sample_df, model_dir=model_dir)
    assert set(ref.columns) == {
        "designation_chars", "designation_words", "prdtypecode", "proba",
    }
    assert len(ref) == len(sample_df)
```

- [ ] **Implémenter** — `src/monitoring/drift_report.py` :

```python
"""Job de detection de derive (Evidently) — one-shot, human-in-the-loop.

Reference : features derivees du jeu de validation + predictions du modele courant.
Current  : journal JSONL des predictions servies par l'API.
Sorties  : rapport HTML, metriques poussees au pushgateway (si PUSHGATEWAY_URL),
           exit code 1 si derive -> l'operateur decide de relancer `make pipeline`.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import load_params, resolve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DRIFT_COLUMNS = ["designation_chars", "designation_words", "prdtypecode", "proba"]


def build_reference(val_df: pd.DataFrame, model_dir=None) -> pd.DataFrame:
    """Predit sur le jeu de validation avec le modele courant -> memes colonnes
    que le journal JSONL (le predictor charge Registry ou local, comme l'API)."""
    from src.models.predict_model import BaselinePredictor

    predictor = BaselinePredictor(model_dir=model_dir)
    rows = []
    for text in val_df["text_combined"].fillna(""):
        top = predictor.predict(str(text), top_k=1)[0]
        rows.append({
            "designation_chars": len(str(text)),
            "designation_words": len(str(text).split()),
            "prdtypecode": top["prdtypecode"],
            "proba": top["proba"],
        })
    return pd.DataFrame(rows)


def load_current(log_path: Path) -> pd.DataFrame:
    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    df = pd.DataFrame(entries)
    return df[DRIFT_COLUMNS]


def evaluate_drift(reference: pd.DataFrame, current: pd.DataFrame, reports_dir: Path) -> dict:
    report = Report([DataDriftPreset()])
    result = report.run(
        reference_data=reference[DRIFT_COLUMNS], current_data=current[DRIFT_COLUMNS]
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_name = f"drift_{stamp}.html"
    result.save_html(str(reports_dir / report_name))

    payload = json.loads(result.json())
    # DataDriftPreset -> metrique DriftedColumnsCount : {count, share}
    drift = next(
        m["value"] for m in payload["metrics"]
        if "DriftedColumnsCount" in m["metric_id"]
    )
    threshold = load_params()["monitoring"]["drift_share_threshold"]
    return {
        "drift_share": float(drift["share"]),
        "drifted_columns": int(drift["count"]),
        "drift_detected": float(drift["share"]) >= threshold,
        "report_name": report_name,
    }


def push_metrics(result: dict) -> None:
    url = os.environ.get("PUSHGATEWAY_URL")
    if not url:
        return
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    Gauge("rakuten_drift_share", "Part de colonnes derivees (Evidently)",
          registry=registry).set(result["drift_share"])
    Gauge("rakuten_drift_detected", "1 si derive au-dela du seuil",
          registry=registry).set(int(result["drift_detected"]))
    push_to_gateway(url, job="drift", registry=registry)
    log.info("Metriques poussees vers %s", url)


def main() -> int:
    params = load_params()["monitoring"]
    log_path = resolve(params["predictions_log"])
    if not log_path.exists():
        log.error("%s introuvable — envoyer du trafic d'abord (make traffic)", log_path)
        return 2
    current = load_current(log_path)
    if len(current) < params["min_current_rows"]:
        log.error("Seulement %d predictions (< %d) — pas assez pour statuer",
                  len(current), params["min_current_rows"])
        return 2

    val = pd.read_parquet(
        resolve(load_params()["data"]["processed_dir"]) / "val.parquet"
    )
    reference = build_reference(val)
    result = evaluate_drift(reference, current, resolve(params["reports_dir"]))
    push_metrics(result)

    log.info("drift_share=%.2f (seuil %.2f) — rapport : %s",
             result["drift_share"], params["drift_share_threshold"],
             result["report_name"])
    if result["drift_detected"]:
        log.warning("DERIVE DETECTEE — examiner le rapport puis relancer "
                    "`make pipeline` et promouvoir la nouvelle version si conforme")
        return 1
    log.info("Pas de derive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Vérifier les tests** : `pytest tests/test_monitoring.py -v` → tout vert.
  NB : l'API `Report`/`DataDriftPreset`/`DriftedColumnsCount` ci-dessus est celle
  d'evidently 0.7.x — si un nom diffère à l'implémentation, ajuster depuis
  `python -c "from evidently.presets import DataDriftPreset; help(DataDriftPreset)"`.

- [ ] **Dockerfile** — stage monitoring :

```dockerfile
# Image du job de derive (docker compose --profile drift)
FROM runtime AS monitoring
COPY requirements-dev.txt .
RUN pip install --no-cache-dir evidently==0.7.23
COPY src/monitoring/ src/monitoring/
CMD ["python", "-m", "src.monitoring.drift_report"]
```

- [ ] **docker-compose.yml** — service drift :

```yaml
  drift:
    build:
      context: .
      target: monitoring
    profiles: ["drift"]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      PUSHGATEWAY_URL: pushgateway:9091
    volumes:
      - ./monitoring_data:/app/monitoring_data
      - ./data:/app/data
    depends_on:
      mlflow:
        condition: service_healthy
```

- [ ] **Makefile** :

```makefile
drift-report:     ## Evalue la derive (rapport HTML + metriques pushgateway)
	docker compose --profile drift run --rm drift
```

(ajouter `drift-report` au `.PHONY`)

- [ ] **Vérifier en conditions réelles** :

```
$ API_KEY=demo make up && API_KEY=demo make traffic N=50
$ make drift-report          # exit 0, "Pas de derive"
$ API_KEY=demo .venv/bin/python -m src.monitoring.send_traffic 50 --drift
$ make drift-report          # exit 1, "DERIVE DETECTEE"
$ curl -s localhost:9091/metrics | grep rakuten_drift   # metriques presentes
# panneau "Drift share" du dashboard Grafana passe au rouge
```

- [ ] **Commit** : `git commit -m "Phase 4: job de derive Evidently (rapport HTML + pushgateway + exit code)"`

**Exemple de sortie attendue :**
```
$ make drift-report
INFO drift_share=0.75 (seuil 0.30) — rapport : drift_20261011T143000Z.html
WARNING DERIVE DETECTEE — examiner le rapport puis relancer `make pipeline` ...
```

## Stage 5 — Documentation & PR

### Task 5.1 : runbook de maintenance + docs

**Files :** Create `docs/phase4_summary.md`, Modify `README.md`

- [ ] **`docs/phase4_summary.md`** : tableau problème→solution (boîte noire→/metrics+Grafana,
  pas de trace→JSONL, pas de détection→Evidently, mise à jour→runbook), architecture,
  **runbook de maintenance** :
  1. Grafana :3000 — panneau drift rouge ou alerte du job (`make drift-report` exit 1)
  2. Examiner le rapport HTML (`monitoring_data/reports/`) : quelles colonnes dérivent ?
  3. Si dérive réelle : récupérer des données récentes → `make pipeline` → comparer les
     runs dans MLflow → promouvoir l'alias `production` → `docker compose restart api`
  4. Vérifier `/health` (nouvelle version) et le retour à la normale du drift share
  + commandes de démo soutenance (traffic normal → dashboard, traffic --drift → alerte).
- [ ] **README** : section Phase 4 (stack monitoring, `make traffic` / `drift-report`,
  URLs :3000/:9090/:9091) + mise à jour du Project Organization (src/monitoring, docker/).
- [ ] **Bilan Phase 4 dans `docs/knowledge_base.md`** (même format que les Phases 1–3).
- [ ] **Vérification finale complète** : `make lint && make test` ; stack up → traffic →
  dashboard peuplé → drift-report normal (exit 0) → drift simulé (exit 1, panneau rouge).
- [ ] **Commit + PR** vers `feat/dev-tyu` — preflight + Docker CI + Copilot review.

## Reuse / références

- `BaselinePredictor` (Registry/local) réutilisé tel quel par `build_reference` —
  la référence est prédite par le même modèle que la production.
- Fixtures de test existantes (`model_dir`, `sample_df` de `tests/conftest.py`).
- Pattern compose Phase 2/3 : profiles pour les jobs one-shot, images officielles
  taguées, healthchecks ; `make up` garde `--build`.
- Leçons Phase 3 à respecter : tags d'actions/images vérifiés, pas de `cmd | tail`
  dans les vérifications, `docker compose up` sans `--build` = piège.

## Out of scope

Alertmanager/notifications (mail, Slack), retrain déclenché automatiquement (décision :
human-in-the-loop), monitoring de la qualité avec labels réels (pas de ground truth en
production), TLS, Kubernetes, dashboards nginx/infra.

## Verification

1. `make test` : suite complète verte **hors ligne** (7 nouveaux tests monitoring/drift).
2. `make up` → Grafana :3000 affiche le dashboard provisionné sans configuration manuelle ;
   après `make traffic N=50`, les 4 panneaux API sont peuplés ; Prometheus :9090 montre
   les targets api (N replicas) et pushgateway « up ».
3. `make drift-report` sur trafic normal → exit 0 ; après `send_traffic --drift` →
   exit 1, rapport HTML généré, `rakuten_drift_share` visible sur :9091 et le panneau
   Grafana passe au rouge.
4. Le workflow de maintenance documenté fonctionne de bout en bout : drift détecté →
   `make pipeline` → promotion alias → restart api → `/health` nouvelle version.
5. PR : preflight + Docker CI verts (l'image runtime ne grossit que de
   l'instrumentator ; evidently reste dans l'image monitoring).
