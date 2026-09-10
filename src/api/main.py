"""API d'inference baseline (FastAPI) — Phase 1 MLOps.

Lancement local : uvicorn src.api.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.models.predict_model import BaselinePredictor

log = logging.getLogger("uvicorn.error")

state: dict = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["predictor"] = BaselinePredictor()
        log.info(
            "Modele baseline charge (%s, source=%s)",
            state["predictor"].version, state["predictor"].model_source,
        )
    except FileNotFoundError as exc:
        state["predictor"] = None
        log.error("Artefacts introuvables (%s) — lancer `make train` d'abord", exc)
    yield
    state["predictor"] = None


app = FastAPI(
    title="Rakuten Product Classification API",
    description="Classification de produits e-commerce en 27 categories "
    "(baseline TF-IDF + LightGBM).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor = state["predictor"]
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_source=predictor.model_source if predictor else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predictor = state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    predictions = predictor.predict(
        request.designation, request.description, top_k=request.top_k
    )
    return PredictResponse(predictions=predictions, model_version=predictor.version)
