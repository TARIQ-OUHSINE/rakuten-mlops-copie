"""Inference baseline : chargement des artefacts et prediction top-k.

Ordre de chargement (Phase 2) :
1. MLflow Model Registry (alias 'production') si MLFLOW_TRACKING_URI est defini ;
2. fallback : artefacts locaux models/baseline/*.joblib.
"""

import logging
import os

import joblib
import numpy as np

from src.config import load_params, resolve
from src.data.text_cleaning import build_text_combined

log = logging.getLogger(__name__)


class BaselinePredictor:
    """Charge les artefacts une fois, predit des codes produit depuis du texte."""

    def __init__(self, model_dir=None):
        params = load_params()
        self.model_source = None

        if model_dir is None and os.environ.get("MLFLOW_TRACKING_URI"):
            try:
                self._load_from_mlflow(params["mlflow"])
            except Exception as exc:
                log.warning(
                    "Chargement MLflow impossible (%s) — fallback local",
                    exc, exc_info=True,
                )

        if self.model_source is None:
            self._load_from_local_dir(model_dir, params["model"])

    def _load_from_mlflow(self, mlflow_params: dict) -> None:
        import mlflow
        from mlflow import MlflowClient

        from src.models.mlflow_utils import PRODUCTION_ALIAS

        # set_tracking_uri explicite : l'URI globale de mlflow peut avoir ete
        # modifiee ailleurs dans le process et primerait sur la variable d'env.
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        name = mlflow_params["registered_model_name"]
        loaded = mlflow.pyfunc.load_model(f"models:/{name}@{PRODUCTION_ALIAS}")
        impl = loaded.unwrap_python_model()
        self.model = impl.model
        self.vectorizer = impl.vectorizer
        self.label_encoder = impl.label_encoder

        registry_version = MlflowClient().get_model_version_by_alias(
            name, PRODUCTION_ALIAS
        ).version
        self.model_source = f"mlflow:v{registry_version}"
        self.version = f"{name}-v{registry_version}"
        log.info("Modele charge depuis le Registry MLflow (%s)", self.model_source)

    def _load_from_local_dir(self, model_dir, model_params: dict) -> None:
        model_dir = resolve(model_dir or model_params["dir"])
        self.model = joblib.load(model_dir / "model.joblib")
        self.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        self.label_encoder = joblib.load(model_dir / "label_encoder.joblib")
        self.model_source = "local"
        self.version = model_params["version"]

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
