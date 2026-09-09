Classification multimodale de produits Rakuten — MLOps
==============================

Industrialisation du projet de classification automatique des produits Rakuten en
27 catégories (`prdtypecode`) à partir du titre et de la description. La phase
data-science (notebooks, démo Streamlit) vit dans le repo
[nov25_alt1_mle_rakuten](https://github.com/DataScientest-Studio/nov25_alt1_mle_rakuten) ;
ce repo porte l'industrialisation MLOps (pipeline scripté, API, Docker, puis
MLflow/DVC/CI/monitoring au fil des phases — voir `docs/plans/`).

Quickstart (Phase 1 — baseline + API)
------------

Prérequis : Python 3.11+, les CSV du challenge
([challengedata.ens.fr](https://challengedata.ens.fr/challenges/35)) placés dans
`data/` (`x_train.csv`, `y_train.csv`).

```bash
make setup          # venv + dépendances + stopwords NLTK
make data           # nettoyage texte + split stratifié -> data/processed/
make train          # TF-IDF + LightGBM -> models/baseline/ (+ metrics.json)
make test           # tests unitaires (pas besoin des données brutes)
make api            # API FastAPI sur http://localhost:8000
```

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"designation": "Console Nintendo Switch OLED"}'
```

Avec Docker (nécessite `make train` au préalable pour produire les artefacts) :

```bash
make docker-build && make docker-run
```

Baseline : TF-IDF (50k features, 1-2 grammes) + LightGBM — F1 weighted validation = 0.81.
Objectifs et métriques : `docs/phase1_objectives.md`. Plans de phase : `docs/plans/`.

Phase 2 — MLflow, DVC & microservices
------------

Chaque `make train` logge un run MLflow (params, métriques, artefacts) et
enregistre le modèle au Model Registry (`rakuten-tfidf-lgbm`) sous forme d'un
pyfunc unique ; l'API charge la version aliasée `production`, avec fallback
sur `models/baseline/` si MLflow est indisponible (`/health` expose
`model_source`). Les données sont versionnées avec DVC et le pipeline
s'orchestre via `dvc repro`.

```bash
make mlflow-ui        # UI MLflow locale (backend ./mlruns) sur :5001
dvc repro             # pipeline prepare -> train (relance le minimum)
dvc push              # pousse données/modèles vers le remote DVC

make up               # docker compose : serveur mlflow (:5001) + api (:8000)
make train-service    # job d'entraînement conteneurisé (dvc repro -> Registry)
make down
```

Workflow complet (promotion d'un modèle, restart API) : `docs/phase2_summary.md`.

Project Organization
------------

    ├── Dockerfile              <- Image API (stage runtime) + job train (stage trainer)
    ├── docker-compose.yml      <- Services mlflow / api / trainer (profile train)
    ├── Makefile                <- setup / data / train / test / api / up / down / ...
    ├── params.yaml             <- Configuration unique (chemins, TF-IDF, LightGBM, MLflow)
    ├── dvc.yaml                <- Pipeline DVC (prepare -> train)
    ├── requirements.txt        <- Dépendances runtime épinglées (pipeline + API + MLflow)
    ├── requirements-dev.txt    <- + pytest, ruff, httpx, dvc
    │
    ├── data                    <- CSV versionnés via DVC (*.dvc committés)
    ├── models                  <- Non versionné (models/baseline/ + Registry MLflow)
    ├── docs                    <- Objectifs, plans de phase, rapports, figures
    │
    ├── src
    │   ├── config.py           <- Chargement de params.yaml
    │   ├── data                <- download_data.py, make_dataset.py, text_cleaning.py
    │   ├── features            <- build_features.py (TF-IDF)
    │   ├── models              <- train_model.py, predict_model.py, mlflow_utils.py
    │   └── api                 <- main.py (FastAPI), schemas.py
    │
    └── tests                   <- Tests unitaires (fixtures committées, sans serveur)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
