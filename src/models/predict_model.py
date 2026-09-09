"""Inference baseline : chargement des artefacts et prediction top-k."""

import joblib
import numpy as np

from src.config import load_params, resolve
from src.data.text_cleaning import build_text_combined


class BaselinePredictor:
    """Charge les artefacts une fois, predit des codes produit depuis du texte."""

    def __init__(self, model_dir=None):
        params = load_params()["model"]
        model_dir = resolve(model_dir or params["dir"])
        self.model = joblib.load(model_dir / "model.joblib")
        self.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        self.label_encoder = joblib.load(model_dir / "label_encoder.joblib")
        self.version = params["version"]

    def predict(self, designation: str, description: str | None = None, top_k: int = 3):
        """Retourne les top_k classes [{prdtypecode, label_idx, proba}]."""
        text = build_text_combined(designation, description)
        x = self.vectorizer.transform([text])
        probas = self.model.predict_proba(x)[0]
        top_idx = np.argsort(probas)[::-1][:top_k]
        return [
            {
                "prdtypecode": int(self.label_encoder.classes_[i]),
                "label_idx": int(i),
                "proba": round(float(probas[i]), 4),
            }
            for i in top_idx
        ]
