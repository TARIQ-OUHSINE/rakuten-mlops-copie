"""Schemas Pydantic de l'API d'inference."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    designation: str = Field(..., min_length=1, description="Titre du produit")
    description: str | None = Field(None, description="Description longue (optionnelle)")
    top_k: int = Field(3, ge=1, le=27, description="Nombre de classes retournees")


class Prediction(BaseModel):
    prdtypecode: int
    label_idx: int
    proba: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
