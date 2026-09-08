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

Project Organization
------------

    ├── Dockerfile              <- Image Docker de l'API d'inférence
    ├── Makefile                <- setup / data / train / test / api / docker-*
    ├── params.yaml             <- Configuration unique (chemins, TF-IDF, LightGBM)
    ├── requirements.txt        <- Dépendances runtime épinglées (pipeline + API)
    ├── requirements-dev.txt    <- + pytest, ruff, httpx
    │
    ├── data                    <- Non versionné (CSV du challenge + data/processed/)
    ├── models                  <- Non versionné (models/baseline/ produit par make train)
    ├── docs                    <- Objectifs, plans de phase, rapports, figures
    │
    ├── src
    │   ├── config.py           <- Chargement de params.yaml
    │   ├── data                <- download_data.py, make_dataset.py, text_cleaning.py
    │   ├── features            <- build_features.py (TF-IDF)
    │   ├── models              <- train_model.py, predict_model.py
    │   └── api                 <- main.py (FastAPI), schemas.py
    │
    └── tests                   <- Tests unitaires (fixtures committées)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
