# Phase 3 — Orchestration, CI & Déploiement

Livrée le 10/09/2026 (branche `feat/phase3`). Plan détaillé :
`docs/plans/2026-09-10-phase3-ci-api-scalabilite.md`.

## Ce qui change par rapport à la Phase 2

| Problème Phase 2 | Solution Phase 3 |
|---|---|
| API ouverte à tous | Clé d'API (`X-API-Key`) sur `/predict`, comparaison constant-time |
| Un seul conteneur API | nginx en frontal + N replicas (`make up REPLICAS=3`) |
| L'image Docker n'est jamais construite en CI | Workflow `docker.yml` : build + Trivy + smoke test sur chaque PR |
| Image publiée nulle part | Push GHCR (`:latest` + `:sha`) au merge sur `feat/dev-tyu` |
| Pipeline en plusieurs commandes | `make pipeline` (dvc repro → dvc push) |

## Sécurité de l'API

- `src/api/security.py` : dépendance FastAPI `require_api_key` (header `X-API-Key`
  comparé à la variable d'env `API_KEY` avec `secrets.compare_digest`).
- **Auth désactivée si `API_KEY` n'est pas définie** — le dev local et les tests
  Phase 1/2 fonctionnent sans configuration.
- `/health` reste ouvert : load balancer, healthchecks compose, CI et monitoring
  (Phase 4) doivent pouvoir sonder sans secret.

```bash
API_KEY=s3cret make up
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"designation":"livre"}'                    # -> 401
curl -X POST localhost:8000/predict -H 'X-API-Key: s3cret' \
  -H 'Content-Type: application/json' -d '{"designation":"livre"}'   # -> 200
```

## Scalabilité (nginx + replicas)

```
client ──:8000──▶ nginx ──▶ api-1 ─┐
                        ──▶ api-2 ─┼──▶ mlflow:5000
                        ──▶ api-N ─┘
```

- `docker/nginx.conf` : la résolution DNS se fait **par requête** (resolver Docker
  `127.0.0.11` + variable dans `proxy_pass`) — sans cela nginx fige l'IP d'un seul
  replica au démarrage. Round-robin entre replicas, failover vérifié (arrêt d'un
  replica → les requêtes continuent).
- L'API n'expose plus de port hôte ; nginx publie `:8000` (transparent pour tous
  les scripts existants).
- `make up REPLICAS=N` (2 par défaut) ; `UVICORN_WORKERS` règle le nombre de
  workers par conteneur (1 par défaut : chaque worker charge le modèle en mémoire).

## CI Docker (`.github/workflows/docker.yml`)

- **Sur PR** vers `feat/dev-tyu` :
  1. build de l'image `runtime` (cache GitHub Actions) ;
  2. scan **Trivy** (CRITICAL/HIGH, vulnérabilités sans correctif ignorées) ;
  3. **smoke test compose** en mode dégradé — le CI n'a ni données ni modèle :
     `/health` → 200 avec `model_loaded: false`, `/predict` → 401 sans clé,
     503 avec clé (pas de modèle). Vérifie l'image, l'auth, nginx et le
     comportement dégradé d'un coup.
- **Au merge** sur `feat/dev-tyu` : publication `ghcr.io/<repo>/api:{latest,sha}`
  (nom d'image forcé en minuscules — exigence Docker, le repo GitHub contient
  des majuscules).

## Orchestration bout en bout

```bash
make pipeline   # dvc repro (data -> train -> run MLflow -> Registry) + dvc push
```

Idempotent : si rien n'a changé, les deux stages sont sautés. Le déploiement d'un
nouveau modèle reste : promotion de l'alias `production` (UI MLflow) → restart API.

## Démo soutenance (commandes)

```bash
API_KEY=demo make up REPLICAS=2      # stack complète
curl localhost:8000/health           # model_source: mlflow:vN
docker stop <api-2> && curl ...      # failover
make pipeline                        # rien à refaire -> "didn't change"
# PR GitHub : preflight + Docker CI verts, image dans l'onglet Packages
```

## Hors périmètre (Phase 4)

Prometheus/Grafana, Evidently (drift), ré-entraînement automatique, TLS/HTTPS,
rate limiting, rotation des clés, CD cloud, Kubernetes.
