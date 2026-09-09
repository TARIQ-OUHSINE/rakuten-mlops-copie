# Phase 2 — Suivi d'expériences, versioning & microservices

Livré le 2026-09-09 (branche `feat/phase2`). Plan détaillé :
`docs/plans/2026-09-08-phase2-mlflow-dvc-microservices.md`.

## Ce qui change par rapport à la Phase 1

| Problème Phase 1 | Solution Phase 2 |
|---|---|
| `metrics.json` écrasé à chaque `make train`, pas d'historique | Chaque entraînement = un run MLflow (params, métriques, artefacts) |
| Modèles non versionnés (dossier gitignoré) | MLflow Model Registry (`rakuten-tfidf-lgbm`, alias `production`) |
| Données non versionnées | DVC (`data/*.csv.dvc` committés, remote local `~/dvc-store/rakuten`) |
| Pipeline relancé entièrement à chaque fois | `dvc repro` ne relance que les stages dont les deps ont changé |
| Tout dans un conteneur unique | 3 services docker-compose : `mlflow`, `api`, `trainer` |

## Architecture

```
docker-compose.yml
├── mlflow    : serveur MLflow (sqlite + volume ./mlflow_data, UI sur :5001)
├── api       : API FastAPI (port 8000) — charge le modèle depuis le Registry
│               (alias "production"), fallback models/baseline/*.joblib
└── trainer   : job one-shot (profile "train") — dvc repro → run MLflow
                → nouvelle version au Registry

dvc.yaml
├── prepare : python -m src.data.make_dataset   (deps: CSV bruts, src/data/)
└── train   : python -m src.models.train_model  (deps: data/processed/, params)
```

Le modèle est enregistré comme **pyfunc unique** (`src/models/mlflow_utils.py`)
embarquant les trois artefacts (model, vectorizer, label_encoder) : un seul
objet atomique à promouvoir. La **première** version reçoit automatiquement
l'alias `production` (bootstrap) ; les suivantes se promeuvent manuellement.

## Workflow type

```bash
# 1. Entraîner (local ou conteneurisé)
make train                    # local : run MLflow vers ./mlruns
make up && make train-service # conteneurisé : run vers le serveur mlflow

# 2. Comparer les runs
make mlflow-ui                # local — http://localhost:5001
# (ou l'UI du service mlflow si compose est up)

# 3. Promouvoir une version en production (UI ou API python)
#    Models > rakuten-tfidf-lgbm > version N > alias "production"

# 4. Redémarrer l'API pour charger la nouvelle version
docker compose restart api
curl localhost:8000/health    # -> {"model_loaded": true, "model_source": "mlflow:vN"}
```

## Versioning des données (DVC)

```bash
dvc repro   # relance uniquement ce qui a changé (prepare et/ou train)
dvc push    # envoie données + outputs vers le remote (~/dvc-store/rakuten)
dvc pull    # restaure données + modèles sur un clone frais
```

Le remote est un répertoire local (décision Phase 2) — swappable vers S3/GCS
en changeant `dvc remote` sans toucher au pipeline.

## Limites connues

- **Mémoire du trainer** : l'entraînement LightGBM (72k × 50k features) a besoin
  de **≥ 4 Go libres dans la VM Docker**. Si d'autres stacks tournent dans la même
  VM (Airflow, Supabase…), le job est OOM-killed (exit 137) — libérer de la
  mémoire ou augmenter la RAM allouée à Docker Desktop. `make train` local n'est
  pas concerné. En cas de job tué, restaurer les artefacts depuis le run MLflow
  (`mlruns/<exp>/<run>/artifacts/model/artifacts/`) ou relancer `make train`.

## Robustesse / dégradé

- `make train` fonctionne **sans** serveur MLflow (tracking file-based `./mlruns`) ;
  si le logging MLflow échoue, les artefacts locaux restent valides.
- L'API démarre **sans** MLflow : `MLFLOW_TRACKING_URI` absent ou serveur
  injoignable → fallback `models/baseline/*.joblib` (`model_source: "local"`).
- Les tests (`make test`) passent hors ligne : le tracking est testé sur un
  file store temporaire, aucun serveur requis.

## Hors périmètre (Phase 3/4)

CI d'entraînement, auth API, Kubernetes, monitoring/drift, industrialisation
du modèle v2 (dont la normalisation des accents — voir le plan Phase 2).
