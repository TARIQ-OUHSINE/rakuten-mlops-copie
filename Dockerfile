FROM python:3.11-slim AS runtime

# apt-get upgrade : applique les mises a jour securite Debian de l'image de base
# (perl-base, gzip, libpcre2, libsqlite3... — CVE OS avec correctif publie).
# LightGBM a besoin de libgomp.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# --upgrade pip/setuptools/wheel : les versions embarquees dans l'image de base
# trainent des CVE corrigees (ex. wheel < 0.46.2)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader stopwords -d /usr/local/nltk_data

COPY params.yaml .
COPY src/config.py src/config.py
COPY src/__init__.py src/__init__.py
COPY src/data/ src/data/
COPY src/features/ src/features/
COPY src/models/ src/models/
COPY src/api/ src/api/
# models/ ne contient que .gitkeep sur un clone frais : l'image reste buildable
# sans artefacts (le lifespan de l'API gere leur absence), et embarque
# models/baseline/ quand `make train` a ete lance avant le build.
COPY models/ models/

EXPOSE 8000
# Forme shell pour interpoler UVICORN_WORKERS (1 par defaut : chaque worker
# charge le modele en memoire)
CMD exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}

# Image du job de derive (docker compose --profile drift) : evidently reste
# hors de l'image API (lourd, inutile au serving).
FROM runtime AS monitoring
RUN pip install --no-cache-dir evidently==0.7.23
COPY src/monitoring/ src/monitoring/
CMD ["python", "-m", "src.monitoring.drift_report"]

# Image du job d'entrainement (docker compose --profile train) :
# la meme base + git/dvc pour `dvc repro` sur le repo monte dans /app.
FROM runtime AS trainer
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir dvc==3.67.1
CMD ["dvc", "repro"]
