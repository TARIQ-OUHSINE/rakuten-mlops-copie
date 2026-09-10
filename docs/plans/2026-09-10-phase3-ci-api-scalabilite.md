# Phase 3 Plan — Orchestration, CI & Déploiement (MLOps Rakuten)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Deadline : 18/09** · Branche : `feat/phase3` (→ PR vers `feat/dev-tyu`) · Pré-requis : Phase 2 mergée (commit `8f3a6d1`)

**Goal :** CI Docker complet (build, smoke test, scan Trivy, publication GHCR), API sécurisée
par clé d'API et servie derrière nginx avec plusieurs replicas, pipeline data→train→publish
orchestré de bout en bout par une seule commande.

**Architecture :** on étend l'existant sans nouvelle brique : le preflight GitHub Actions
gagne un workflow Docker séparé ; l'API FastAPI gagne un module `security.py` (dépendance
FastAPI) et deux middlewares ; la scalabilité se fait en docker-compose (nginx en frontal,
`--scale api=N`) ; l'orchestration reste `dvc repro` derrière un target Makefile unique.

**Tech stack :** FastAPI (APIKeyHeader), nginx, docker compose, GitHub Actions
(docker/build-push-action, aquasecurity/trivy-action), GHCR, DVC.

## Context

La Phase 2 a livré MLflow (tracking + Registry), DVC et 3 services compose, mais :
l'API est ouverte à tous (pas d'auth), un seul conteneur API sert les requêtes (pas de
scalabilité), le CI ne construit jamais l'image Docker (une régression du Dockerfile
passerait inaperçue), et l'image n'est publiée nulle part. La Phase 3 (kick-off :
« Orchestration & Déploiement ») ferme ces quatre trous.

**Décisions confirmées avec l'utilisateur (10/09) :**
- Auth : **clé d'API** (header `X-API-Key`, valeur en variable d'env). Pas d'OAuth2/JWT.
- Orchestration : **DVC + compose existants** (pas d'Airflow/Prefect) — un target
  Makefile unique enchaîne le pipeline.
- CI : **build image + push GHCR + smoke test compose + scan Trivy** (les quatre).

## Architecture cible

```
                   ┌─────────────┐
  client ──:8000──▶│    nginx    │──▶ api (replica 1) ─┐
                   │ (round robin│──▶ api (replica 2) ─┼──▶ mlflow:5000 (Registry)
                   └─────────────┘        ...          ─┘
  X-API-Key requis sur /predict ; /health ouvert (LB, CI, monitoring)

  CI (GitHub Actions)
  ├── preflight.yml (existant) : ruff + pytest + LOC 500 + Copilot review
  └── docker.yml (nouveau)
      ├── PR        : build runtime → Trivy scan → smoke test compose (health 200, predict 503/401)
      └── push dev  : build + push ghcr.io/<owner>/<repo>/api:{latest, sha}

  make pipeline : dvc repro → dvc push   (data → train → run MLflow → Registry → remote DVC)
```

## Fichiers

- Create : `src/api/security.py` (dépendance API key)
- Create : `docker/nginx.conf` (reverse proxy + résolution DNS dynamique)
- Create : `.github/workflows/docker.yml` (CI Docker)
- Create : `tests/test_security.py`
- Create : `docs/phase3_summary.md`
- Modify : `src/api/main.py` (dépendance sur /predict, middlewares GZip + temps de réponse)
- Modify : `docker-compose.yml` (service nginx, api sans port hôte, env API_KEY)
- Modify : `Makefile` (targets `pipeline`, `up` avec scale), `README.md`
- Modify : `.github/workflows/preflight.yml` (rien — le workflow Docker est séparé)

---

## Stage 1 — Sécurisation de l'API (clé d'API)

### Task 1.1 : dépendance `require_api_key`

**Files :** Create `src/api/security.py`, Create `tests/test_security.py`, Modify `src/api/main.py`

- [ ] **Écrire les tests (rouges d'abord)** — `tests/test_security.py` :

```python
"""Tests de l'auth par cle d'API (X-API-Key)."""

from fastapi.testclient import TestClient

from src.api.main import app, state


class FakePredictor:
    version = "test"
    model_source = "local"

    def predict(self, designation, description=None, top_k=3):
        return [{"prdtypecode": 10, "label_idx": 0, "proba": 1.0}]


def make_client():
    client = TestClient(app)
    state["predictor"] = FakePredictor()
    return client


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
```

- [ ] **Vérifier l'échec** : `pytest tests/test_security.py -v` → `ModuleNotFoundError`/401 manquants
- [ ] **Implémenter** — `src/api/security.py` :

```python
"""Auth par cle d'API : header X-API-Key compare a la variable d'env API_KEY.

Si API_KEY n'est pas definie, l'auth est desactivee (dev local, tests Phase 1/2).
"""

import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = os.environ.get("API_KEY")
    if not expected:
        return
    if key is None or not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Cle d'API invalide ou absente")
```

- [ ] **Brancher sur /predict** — `src/api/main.py` (imports + décorateur de route) :

```python
from fastapi import Depends, FastAPI, HTTPException
from src.api.security import require_api_key

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(request: PredictRequest) -> PredictResponse:
    ...  # corps inchange
```

`/health` reste sans dépendance (load balancer, healthcheck compose, CI).

- [ ] **Vérifier** : `make lint && make test` → tout vert (les tests Phase 1/2 passent
  car API_KEY n'est pas définie dans l'environnement de test)
- [ ] **Commit** : `git commit -m "Phase 3: auth par cle d'API sur /predict (X-API-Key)"`

**Exemple de sortie attendue :**
```
$ curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"designation":"livre"}'
401
$ curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -H 'X-API-Key: s3cret' -d '{"designation":"livre"}'
{"predictions":[...],"model_version":"rakuten-tfidf-lgbm-v1"}
```

## Stage 2 — Optimisation de l'API

### Task 2.1 : middlewares GZip + temps de réponse

**Files :** Modify `src/api/main.py`, Modify `tests/test_api.py`

- [ ] **Test (rouge)** — ajouter à `tests/test_api.py` :

```python
def test_response_has_process_time_header(client):
    r = client.get("/health")
    assert "x-process-time-ms" in r.headers
    assert float(r.headers["x-process-time-ms"]) >= 0
```

- [ ] **Implémenter** — `src/api/main.py` après la création de `app` :

```python
import time
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response
```

- [ ] **Workers uvicorn configurables** — `Dockerfile`, stage runtime, CMD en forme shell
  pour interpoler l'env (1 worker par défaut : chaque worker charge le modèle en mémoire) :

```dockerfile
CMD uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}
```

- [ ] **Vérifier** : `make test` vert ; `make docker-build` OK
- [ ] **Commit** : `git commit -m "Phase 3: middlewares gzip + X-Process-Time-Ms, workers configurables"`

## Stage 3 — Scalabilité Docker (nginx + replicas)

### Task 3.1 : reverse proxy nginx

**Files :** Create `docker/nginx.conf`, Modify `docker-compose.yml`, Modify `Makefile`

- [ ] **Écrire** `docker/nginx.conf` — résolution DNS **par requête** (le resolver Docker
  127.0.0.11 répartit en round-robin entre les replicas ; sans variable, nginx fige
  l'IP au démarrage) :

```nginx
events {}

http {
    resolver 127.0.0.11 valid=5s;

    server {
        listen 80;

        location / {
            set $api_upstream http://api:8000;
            proxy_pass $api_upstream;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 30s;
        }
    }
}
```

- [ ] **Modifier** `docker-compose.yml` — l'API perd son port hôte, nginx prend :8000
  (transparent pour tous les curl/scripts existants) :

```yaml
  nginx:
    image: nginx:1.27-alpine
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "8000:80"
    depends_on:
      - api

  api:
    build:
      context: .
      target: runtime
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      API_KEY: ${API_KEY:-}
    depends_on:
      mlflow:
        condition: service_healthy
    # plus de ports: — seul nginx est exposé
```

- [ ] **Makefile** — scale paramétrable :

```makefile
REPLICAS ?= 2

up:               ## Demarre mlflow + api (x$(REPLICAS)) + nginx
	docker compose up -d --scale api=$(REPLICAS)
```

- [ ] **Vérifier en local** :

```
$ make up
$ docker compose ps --format '{{.Name}}' | grep api      # 2 replicas
$ for i in 1 2 3 4; do curl -s localhost:8000/health | head -c 60; echo; done   # 200 x4
$ make down
```

- [ ] **Commit** : `git commit -m "Phase 3: nginx en frontal + scale api parametrable"`

**Exemple de sortie attendue :**
```
$ docker compose ps --format '{{.Name}}' | grep api
nov25_alt1_mle_mlops_rakuten-api-1
nov25_alt1_mle_mlops_rakuten-api-2
$ curl -s localhost:8000/health
{"status":"ok","model_loaded":true,"model_source":"mlflow:v1"}
```

## Stage 4 — CI Docker (build, Trivy, smoke test, GHCR)

### Task 4.1 : workflow `.github/workflows/docker.yml`

**Files :** Create `.github/workflows/docker.yml`

- [ ] **Écrire le workflow** :

```yaml
name: Docker CI

on:
  pull_request:
    branches:
      - feat/dev-tyu
  push:
    branches:
      - feat/dev-tyu
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: docker-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

env:
  IMAGE: ghcr.io/${{ github.repository }}/api

jobs:
  build-scan-smoke:
    name: Build, Trivy & Smoke Test
    runs-on: ubuntu-latest
    timeout-minutes: 25
    steps:
      - name: Check out repository
        uses: actions/checkout@v5
        with:
          persist-credentials: false
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Build runtime image
        uses: docker/build-push-action@v6
        with:
          context: .
          target: runtime
          load: true
          tags: rakuten-api:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@0.29.0
        with:
          image-ref: rakuten-api:ci
          format: table
          severity: CRITICAL,HIGH
          exit-code: "1"
          ignore-unfixed: true
      - name: Smoke test (compose, mode degrade sans artefacts)
        run: |
          # `docker compose up` reconstruit l'api depuis le cache local du daemon
          # (les layers viennent d'etre charges par build-push-action `load: true`)
          docker compose up -d
          for i in $(seq 1 30); do
            code=$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/health || true)
            [ "$code" = "200" ] && break
            sleep 2
          done
          curl -sf localhost:8000/health | grep -q '"model_loaded":false'
          predict_code=$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:8000/predict \
            -H 'Content-Type: application/json' -d '{"designation":"livre"}')
          [ "$predict_code" = "503" ]   # pas de modele dans le CI -> 503 attendu
          docker compose down
      - name: Show compose logs on failure
        if: failure()
        run: docker compose logs

  publish:
    name: Push image to GHCR
    if: github.event_name == 'push'
    needs: build-scan-smoke
    runs-on: ubuntu-latest
    timeout-minutes: 20
    permissions:
      contents: read
      packages: write
    steps:
      - name: Check out repository
        uses: actions/checkout@v5
        with:
          persist-credentials: false
      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          target: runtime
          push: true
          tags: |
            ${{ env.IMAGE }}:latest
            ${{ env.IMAGE }}:${{ github.sha }}
          cache-from: type=gha
```

- [ ] **Vérifier localement ce qui peut l'être** : `python -c "import yaml; yaml.safe_load(open('.github/workflows/docker.yml'))"`
  et rejouer le smoke test à la main (`docker compose up -d`, curls ci-dessus).
- [ ] **Commit + push** : la PR déclenchera le job `build-scan-smoke` — c'est la vraie
  vérification. Si Trivy bloque sur des CVE de dépendances de base sans correctif,
  `ignore-unfixed: true` est déjà posé ; sinon épingler/mettre à jour la dépendance.
- [ ] **Après merge** : vérifier `ghcr.io/<owner>/<repo>/api:latest` visible dans
  Packages du repo.

**Exemple de sortie attendue (job smoke test) :**
```
✓ health 200 apres N tentatives : {"status":"ok","model_loaded":false,"model_source":null}
✓ predict sans modele -> 503
```

## Stage 5 — Orchestration bout en bout + docs

### Task 5.1 : target `pipeline` + runbook

**Files :** Modify `Makefile`, Create `docs/phase3_summary.md`, Modify `README.md`

- [ ] **Makefile** :

```makefile
pipeline:         ## Pipeline complet : data -> train -> MLflow/Registry -> remote DVC
	$(BIN)/dvc repro
	$(BIN)/dvc push
```

- [ ] **Vérifier** : `make pipeline` → « didn't change, skipping » sur les 2 stages
  (idempotence), `dvc push` sans erreur.
- [ ] **Écrire `docs/phase3_summary.md`** : tableau problème→solution (API ouverte→API key,
  1 conteneur→nginx+replicas, pas de CI image→docker.yml, pipeline manuel→make pipeline),
  workflow de déploiement (CI publie sur GHCR → pull de l'image taguée), commandes de
  démo pour la soutenance.
- [ ] **README** : section Phase 3 (auth, scale, CI, `make pipeline`), variables d'env
  (`API_KEY`, `REPLICAS`, `UVICORN_WORKERS`).
- [ ] **Commit final + PR** vers `feat/dev-tyu` (le preflight + Docker CI + Copilot
  review tournent sur la PR).

## Reuse / références

- Auth : brancher via `dependencies=[Depends(...)]` sur la route existante
  `src/api/main.py:predict` — ne pas toucher à la logique de prédiction.
- Compose : `docker-compose.yml` Phase 2 (services mlflow/api/trainer) — le service
  trainer et ses volumes ne changent pas.
- CI : conventions du `preflight.yml` existant (concurrency group, timeout par job,
  `persist-credentials: false`, workflow_dispatch).
- Le smoke test CI s'appuie sur le mode dégradé validé en Phase 1/2 : l'API démarre
  sans artefacts (`model_loaded: false`, 503 sur /predict).

## Out of scope (Phase 4)

Monitoring Prometheus/Grafana, détection de dérive (Evidently), ré-entraînement
automatique, Kubernetes, TLS/HTTPS, rate limiting, rotation des clés d'API,
CD vers un environnement cloud.

## Verification

1. `make test` : suite complète verte **hors ligne** (tests auth inclus), `make lint` propre.
2. Local : `API_KEY=s3cret make up` → `/predict` sans clé = 401, avec clé = 200,
   `/health` sans clé = 200 ; `docker compose ps` montre 2 replicas api + nginx.
3. En coupant un replica (`docker stop <api-2>`), les requêtes continuent de passer
   (nginx re-résout via le DNS Docker).
4. PR : preflight vert + job `Build, Trivy & Smoke Test` vert (health 200 dégradé,
   predict 503, scan sans CRITICAL/HIGH non corrigées).
5. Après merge sur `feat/dev-tyu` : image `ghcr.io/.../api:{latest,sha}` publiée
   (job `publish`), visible dans l'onglet Packages.
6. `make pipeline` idempotent (« didn't change ») et `dvc push` in sync.
