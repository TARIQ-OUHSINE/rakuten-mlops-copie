# Phase 2 Plan — Microservices, Suivi & Versioning (MLOps Rakuten)

**Deadline : 11/09** · Branche : `feat/mlops` · Pré-requis : Phase 1 terminée (commit `0374da0`)

## Context

La Phase 1 a livré un pipeline baseline scripté (TF-IDF + LightGBM, F1 weighted = 0.8143),
une API FastAPI conteneurisée et 24 tests. Mais : les runs d'entraînement ne sont pas
tracés (metrics.json écrasé à chaque `make train`), les données et modèles ne sont pas
versionnés (dossiers gitignorés), et tout tourne dans un seul conteneur. La Phase 2
ajoute le suivi d'expériences (MLflow), le versioning données/modèles (DVC + MLflow
Model Registry) et un découpage en microservices orchestrés par docker-compose.

**Décisions confirmées avec l'utilisateur :**
- Versioning : **DVC pour les données**, **MLflow Model Registry pour les modèles**.
- Remote DVC : **répertoire local** (`~/dvc-store/rakuten`), swappable vers cloud plus tard.
- Microservices : **3 services** — API d'inférence, serveur MLflow, job d'entraînement
  one-shot. Streamlit reste local (refonte reportée).

## Architecture cible

```
docker-compose.yml
├── mlflow    : serveur MLflow (backend sqlite + volume ./mlflow_data, port 5001)
├── api       : image Phase 1 étendue — charge le modèle depuis le Registry
│               (stage "Production"), fallback models/baseline/ si MLflow absent
└── trainer   : one-shot (profile "train") — dvc repro → log MLflow → register model

dvc.yaml (orchestration simple du pipeline)
├── stage prepare : python -m src.data.make_dataset   (deps: x_train.csv, y_train.csv, src/data/)
└── stage train   : python -m src.models.train_model  (deps: data/processed/, params.yaml → outs: models/baseline/)
```

## Tasks (staged)

### Stage 1 — Suivi d'expériences MLflow
1. Ajouter `mlflow` aux requirements ; nouvelle section `mlflow:` dans `params.yaml`
   (`tracking_uri` par défaut `./mlruns`, `experiment_name: rakuten-baseline`,
   `registered_model_name: rakuten-tfidf-lgbm`). Surcharge par env `MLFLOW_TRACKING_URI`.
2. Instrumenter `src/models/train_model.py` : `mlflow.start_run()` → log des params
   (tfidf + lgbm), des métriques (f1_weighted, f1_macro, accuracy, + f1 par classe),
   des artefacts (metrics.json, vectorizer, label_encoder) et du modèle
   (`mlflow.sklearn.log_model(..., registered_model_name=...)`). Le comportement
   fichier local (models/baseline/) est conservé — l'API doit marcher sans MLflow.
3. `make mlflow-ui` : lance l'UI locale.

**Exemple de sortie attendue :**
```
$ make train
[mlflow] run_id=3f2a... experiment=rakuten-baseline
[eval  ] val weighted F1 = 0.8143 | macro F1 = 0.7988 | acc = 0.8109
[mlflow] modele enregistre : rakuten-tfidf-lgbm version 2
```

### Stage 2 — Versioning des données (DVC)
4. `pip install dvc` (dev) ; `dvc init` ; remote local :
   `dvc remote add -d localstore ~/dvc-store/rakuten`.
5. `dvc add data/x_train.csv data/y_train.csv` → les `.dvc` sont committés,
   ajuster `.gitignore` (DVC gère ses propres entrées).
6. `dvc.yaml` avec les 2 stages `prepare` et `train` (deps/outs ci-dessus) :
   `dvc repro` ne relance que ce qui a changé — c'est l'orchestration simple du pipeline.
   `dvc push` envoie données + outputs vers le remote.

**Exemple de sortie attendue :**
```
$ dvc repro
'prepare' didn't change, skipping
Running stage 'train': python -m src.models.train_model
$ dvc push
12 files pushed
```

### Stage 3 — Chargement du modèle depuis le Registry
7. `src/models/predict_model.py` : `BaselinePredictor` essaie d'abord
   `mlflow.sklearn.load_model("models:/rakuten-tfidf-lgbm/Production")` si
   `MLFLOW_TRACKING_URI` est défini ; sinon (ou en cas d'échec) fallback sur
   `models/baseline/*.joblib`. La version chargée est exposée dans `/health`
   (`model_source: "mlflow:v2" | "local"`).
8. Vectorizer + label_encoder loggés comme artefacts du run et récupérés avec le
   modèle (ou pyfunc wrapper qui embarque les trois — à trancher à l'implémentation,
   pyfunc recommandé pour un seul artefact atomique).

### Stage 4 — Microservices docker-compose
9. `docker-compose.yml` :
   - `mlflow` : `python:3.11-slim` + `pip install mlflow`, commande
     `mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --artifacts-destination /mlflow/artifacts --host 0.0.0.0`, volume `./mlflow_data:/mlflow`, port `5001:5000`.
   - `api` : build du Dockerfile Phase 1, env `MLFLOW_TRACKING_URI=http://mlflow:5000`,
     `depends_on: mlflow`, port 8000.
   - `trainer` : même image que l'API + deps dev, `profiles: ["train"]`, commande
     `dvc repro`, volumes data/ et models/ montés, env MLflow.
10. Makefile : `up`, `down`, `train-service` (`docker compose --profile train run trainer`).

### Stage 5 — Tests & docs
11. Tests : `test_model.py` — fallback local quand MLflow indisponible ;
    test tracking avec `mlflow.set_tracking_uri(tmp_path)` (pas de serveur requis).
12. `docs/phase2_summary.md` + mise à jour du README (schéma des services, workflow
    `dvc repro` → MLflow UI → promotion du modèle → restart API).

## Reuse / références
- Entraînement existant : `src/models/train_model.py` (fonctions `train/evaluate/save_artifacts` — insérer le logging sans les casser).
- Chargement : `src/models/predict_model.py:BaselinePredictor` (garder l'argument `model_dir` utilisé par les tests).
- Image API : `Dockerfile` Phase 1 (base des services `api` et `trainer`).

## Out of scope (Phase 3/4)
CI GitHub Actions, sécurité API (auth), Kubernetes, monitoring/drift, refonte Streamlit
vers l'API, industrialisation du modèle v2 MetaMLP.

## Verification
1. `make train` local (sans serveur) → run visible dans `mlflow ui` (backend ./mlruns), modèle enregistré au Registry.
2. `dvc repro` idempotent (2e run : « didn't change ») ; `dvc push` puis `dvc pull` sur un clone → données restaurées.
3. `docker compose up -d` → MLflow UI sur :5001, `curl :8000/health` → `model_source` renseigné.
4. `docker compose --profile train run trainer` → nouveau run + nouvelle version du modèle dans le Registry ; après promotion en Production et restart de l'API, `/health` affiche la nouvelle version.
5. `make test` : les 24 tests Phase 1 + nouveaux tests MLflow passent sans serveur MLflow actif.
