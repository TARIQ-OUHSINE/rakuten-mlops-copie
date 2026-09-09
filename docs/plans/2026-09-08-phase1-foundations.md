# Phase 1 Plan — Fondations & Conteneurisation (MLOps Rakuten)

## Context

The Rakuten multimodal classification project (84,916 products, 27 classes) has completed its data-science phase: notebooks 01–05 cover EDA → baseline (TF-IDF + LightGBM, F1≈0.82) → deep experiments → MetaMLP stacking (F1≈0.88), plus a Streamlit demo (`src/streamlit/app.py`). The industrialization phase now begins. Phase 1 ("Fondations & Conteneurisation", due ~24/07) requires: project objectives & metrics, a reproducible dev environment, data collection/preprocessing scripts, a base ML model with unit tests, and a simple inference API — containerized.

Today the repo has **no MLOps foundation**: `src/models/train_model.py`, `src/models/predict_model.py`, `src/features/build_features.py` are one-line stubs; there are no tests, no Dockerfile, no API. Logic lives only in notebooks. Data (~6.8 GB) and models are gitignored.

**Decisions confirmed with user:**
- Base model = **TF-IDF + LightGBM (text-only)**, retrainable from scripts. v2 MetaMLP fusion deferred to Phase 2/3.
- Containerize the **API only** (Streamlit stays local).
- Env tooling = **pip + venv + pinned requirements**.

## Target structure

```
├── Dockerfile                    # API image
├── Makefile                      # setup / data / train / api / test / docker targets
├── requirements.txt              # pinned runtime deps
├── requirements-dev.txt          # pytest, ruff, httpx...
├── params.yaml                   # paths + hyperparams (single source of config)
├── src/
│   ├── config.py                 # loads params.yaml, resolves paths
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_data.py      # fetch/check raw x_train.csv, y_train.csv (or verify local copy)
│   │   └── make_dataset.py       # cleaning from NB01: HTML strip, lowercase, text_clean col, stratified split
│   ├── features/
│   │   └── build_features.py     # TF-IDF vectorizer fit/transform (50k, 1-2 grams) — replaces stub
│   ├── models/
│   │   ├── train_model.py        # LightGBM training + eval (weighted F1) → models/baseline/ artifacts + metrics.json
│   │   └── predict_model.py      # load artifacts, predict(text) → top-k classes + probas
│   └── api/
│       ├── __init__.py
│       ├── main.py               # FastAPI app
│       └── schemas.py            # Pydantic request/response models
├── tests/
│   ├── test_data.py              # cleaning functions (HTML removal, empty text, accents)
│   ├── test_features.py          # vectorizer shape/vocab persistence round-trip
│   ├── test_model.py             # predict output shape, probas sum to 1, known-input sanity
│   └── test_api.py               # TestClient: /health, /predict happy path + validation errors
└── docs/plans/2026-09-08-phase1-foundations.md   # this plan, copied into repo
```

## Tasks (staged)

### Stage 0 — Branch
0. Create and switch to a new branch `feat/mlops` from `main`; all Phase 1 work is committed there (no direct commits to `main`).

### Stage 1 — Objectives, environment, repo hygiene
1. Copy this plan to `docs/plans/2026-09-08-phase1-foundations.md`.
2. Write `docs/phase1_objectives.md`: business objective (auto-categorization of Rakuten products), ML metric (**weighted F1**, baseline target ≥0.80; ref: NB02 = 0.82, SIGIR baseline = 0.8705), ops metrics (API latency p95 < 500 ms CPU, test coverage on core modules).
3. Pin dependencies: freeze current versions into `requirements.txt` (runtime: pandas, scikit-learn, lightgbm, fastapi, uvicorn, joblib, pyyaml...) and add `requirements-dev.txt` (pytest, httpx, ruff). Torch/transformers/streamlit move to a separate `requirements-streamlit.txt` so the API image stays slim.
4. `Makefile` with targets: `setup`, `data`, `train`, `test`, `api`, `docker-build`, `docker-run`.
5. Update `README.md` quickstart (venv setup → make data → make train → make api).
6. Repo hygiene: add `.dockerignore`; gitignore `.DS_Store`; keep `data/`, `models/` ignored.

### Stage 2 — Data pipeline (from NB01)
7. `src/config.py` + `params.yaml`: data paths, split ratio/seed, TF-IDF params, LightGBM params.
8. `src/data/make_dataset.py`: extract cleaning from `notebooks/main_notebooks/01_preprocessing_eda.ipynb` — merge `x_train.csv`+`y_train.csv`, strip HTML (designation+description → `text_clean`), stratified train/val split (seed fixed) → `data/processed/{train,val}.parquet`. CLI: `python -m src.data.make_dataset`.
9. `src/data/download_data.py`: verify presence + expected columns of raw CSVs, clear error message pointing to the challenge source if missing (data is 52 MB, not committed).

### Stage 3 — Baseline model + tests (from NB02)
10. `src/features/build_features.py`: TfidfVectorizer (max_features=50k, ngram 1-2, sublinear_tf, min_df=2) fit on train / save with joblib.
11. `src/models/train_model.py`: train LightGBM (class-weight balanced, params from `params.yaml`), evaluate on val → print + save `models/baseline/metrics.json` (weighted F1, macro F1, accuracy, per-class report), save `model.joblib`, `vectorizer.joblib`, `label_encoder.joblib`.
12. `src/models/predict_model.py`: `BaselinePredictor` class — load artifacts once, `predict(text: str, top_k=3)`.
13. `tests/` as in tree above; small fixture dataset (~50 rows sampled) committed under `tests/fixtures/` so tests run without the 52 MB raw data. `pytest.ini`/`pyproject` config; all tests must pass via `make test`.

### Stage 4 — Inference API + Docker
14. `src/api/main.py` (FastAPI):
    - `GET /health` → `{"status":"ok","model_loaded":true}`
    - `POST /predict` → body `{"designation": str, "description": str|null}`, response below.
    - Model loaded at startup (lifespan), Pydantic validation, 422 on empty text.
15. `Dockerfile`: `python:3.11-slim`, install `requirements.txt`, copy `src/` + `models/baseline/`, `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`. Multi-stage not needed yet.
16. Verify container end-to-end (see Verification).

**Expected API output example:**
```json
POST /predict  {"designation": "Console Nintendo Switch OLED", "description": null}
→ 200
{
  "predictions": [
    {"prdtypecode": 2905, "label_idx": 26, "proba": 0.87},
    {"prdtypecode": 60,   "label_idx": 3,  "proba": 0.06},
    {"prdtypecode": 2462, "label_idx": 21, "proba": 0.03}
  ],
  "model_version": "baseline-tfidf-lgbm-v1"
}
```

**Expected training output example:**
```
$ make train
[train] TF-IDF: (67932, 50000) — LightGBM 27 classes, class_weight=balanced
[eval ] val weighted F1 = 0.81  | macro F1 = 0.79 | acc = 0.83
[save ] models/baseline/{model,vectorizer,label_encoder}.joblib + metrics.json
```

## Reuse / references
- Cleaning + split logic: `notebooks/main_notebooks/01_preprocessing_eda.ipynb`
- TF-IDF + LightGBM params & eval: `notebooks/main_notebooks/02_baseline_multimodale.ipynb`
- Text-cleaning regexes also present in `src/streamlit/app.py` (~line 16+, `pickle`/`re` usage) — extract once into `src/data/make_dataset.py` and have future code import from there.
- Existing v1 artifacts (`models/lgb_text_model.pkl`, `models/tfidf_vectorizer.pkl`) serve as sanity reference for metric parity, not reused directly (we retrain reproducibly).

## Out of scope (later phases)
MLflow tracking, DVC data versioning, microservices split (Phase 2); CI pipeline, K8s, API auth (Phase 3); monitoring/drift (Phase 4). v2 MetaMLP serving deferred.

## Verification
1. `make setup && make data && make train` on a fresh venv → metrics.json written, weighted F1 ≥ 0.78.
2. `make test` → all pytest green (runs on committed fixtures, no big data needed).
3. `make api` locally → `curl localhost:8000/health` and a `/predict` call return expected JSON.
4. `make docker-build && make docker-run` → same two curl checks against the container.
