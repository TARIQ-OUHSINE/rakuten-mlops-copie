PYTHON      ?= python3
VENV        ?= .venv
BIN         := $(VENV)/bin
IMAGE       := rakuten-api
PORT        ?= 8000

.PHONY: setup data train test api lint docker-build docker-run clean mlflow-ui up down train-service pipeline traffic drift-report

setup:            ## Cree le venv et installe les dependances dev
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/python -m nltk.downloader stopwords -q

data:             ## Verifie les CSV bruts puis construit data/processed/
	$(BIN)/python -m src.data.make_dataset

train:            ## Entraine le baseline TF-IDF + LightGBM -> models/baseline/
	$(BIN)/python -m src.models.train_model

test:             ## Lance les tests unitaires
	$(BIN)/python -m pytest tests/ -v

lint:             ## Lint du code source
	$(BIN)/ruff check src/ tests/

mlflow-ui:        ## UI MLflow locale (backend sqlite local)
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5001

api:              ## Lance l'API en local
	$(BIN)/uvicorn src.api.main:app --host 0.0.0.0 --port $(PORT) --reload

docker-build:     ## Construit l'image Docker de l'API
	docker build --target runtime -t $(IMAGE) .

REPLICAS ?= 2

up:               ## Demarre mlflow + api (x$(REPLICAS)) + nginx
	docker compose up -d --build --scale api=$(REPLICAS)

down:             ## Arrete les services docker compose
	docker compose down

train-service:    ## Lance le job d'entrainement conteneurise (dvc repro)
	docker compose --profile train run --rm trainer

pipeline:         ## Pipeline complet : data -> train -> MLflow/Registry -> remote DVC
	$(BIN)/dvc repro
	$(BIN)/dvc push

traffic:          ## Envoie N=100 requetes de demo ; ARGS="--drift" pour simuler la derive
	$(BIN)/python -m src.monitoring.send_traffic $(N) $(ARGS)

drift-report:     ## Evalue la derive (rapport HTML + metriques pushgateway)
	docker compose --profile drift run --rm --build drift

docker-run:       ## Lance le conteneur API
	docker run --rm -p $(PORT):8000 $(IMAGE)

clean:            ## Supprime venv et artefacts temporaires
	rm -rf $(VENV) .pytest_cache .ruff_cache
