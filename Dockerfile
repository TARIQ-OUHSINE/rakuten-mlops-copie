FROM python:3.11-slim

# LightGBM a besoin de libgomp
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
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
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
