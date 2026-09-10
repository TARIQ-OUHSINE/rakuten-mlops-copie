"""Integration MLflow : wrapper pyfunc et logging du run d'entrainement.

Le modele est enregistre au Model Registry comme un pyfunc unique embarquant
les trois artefacts (model, vectorizer, label_encoder) — un seul objet
atomique a promouvoir/charger. L'entrainement local (models/baseline/)
reste la source de verite : ce module ne fait que publier vers MLflow.
"""

import logging
import os
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient

log = logging.getLogger(__name__)

PRODUCTION_ALIAS = "production"


class RakutenPyfuncModel(mlflow.pyfunc.PythonModel):
    """Pyfunc embarquant model + vectorizer + label_encoder.

    predict() accepte un DataFrame [designation, description] et retourne
    les probabilites par classe ; l'API recupere les objets sous-jacents
    via unwrap_python_model() pour garder sa logique top-k existante.
    """

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
        self.vectorizer = joblib.load(context.artifacts["vectorizer"])
        self.label_encoder = joblib.load(context.artifacts["label_encoder"])

    def predict(self, context, model_input, params=None):
        from src.data.text_cleaning import build_text_combined

        texts = [
            build_text_combined(row.get("designation"), row.get("description"))
            for row in model_input.to_dict(orient="records")
        ]
        return self.model.predict_proba(self.vectorizer.transform(texts))


def resolve_tracking_uri(mlflow_params: dict) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or mlflow_params["tracking_uri"]


def log_training_run(model_dir: Path, metrics: dict, params: dict) -> None:
    """Log params/metriques/artefacts et enregistre le pyfunc au Registry."""
    mlf = params["mlflow"]
    mlflow.set_tracking_uri(resolve_tracking_uri(mlf))
    mlflow.set_experiment(mlf["experiment_name"])
    model_name = mlf["registered_model_name"]

    with mlflow.start_run() as run:
        log.info("[mlflow] run_id=%s experiment=%s", run.info.run_id, mlf["experiment_name"])
        mlflow.log_params({f"tfidf_{k}": v for k, v in params["tfidf"].items()})
        mlflow.log_params({f"lgbm_{k}": v for k, v in params["lgbm"].items()})
        mlflow.log_param("early_stopping_rounds", params["early_stopping_rounds"])

        mlflow.log_metrics(
            {k: metrics[k] for k in ("f1_weighted", "f1_macro", "accuracy")}
        )
        label_encoder = joblib.load(model_dir / "label_encoder.joblib")
        for idx, klass in enumerate(label_encoder.classes_):
            per_class = metrics["report"].get(str(idx))
            if per_class:
                mlflow.log_metric(f"f1_class_{klass}", per_class["f1-score"])

        mlflow.log_artifact(str(model_dir / "metrics.json"))
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RakutenPyfuncModel(),
            artifacts={
                "model": str(model_dir / "model.joblib"),
                "vectorizer": str(model_dir / "vectorizer.joblib"),
                "label_encoder": str(model_dir / "label_encoder.joblib"),
            },
            registered_model_name=model_name,
        )

    version = model_info.registered_model_version
    log.info("[mlflow] modele enregistre : %s version %s", model_name, version)
    _bootstrap_production_alias(model_name, version)


def _bootstrap_production_alias(model_name: str, version: str) -> None:
    """Pose l'alias 'production' sur la 1re version ; ensuite promotion manuelle."""
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
        return
    except mlflow.exceptions.MlflowException as exc:
        # Ne bootstrapper que si l'alias est reellement absent : une erreur
        # transitoire (reseau, 500) ne doit pas re-pointer 'production'.
        if exc.error_code not in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
            raise
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version)
    log.info(
        "[mlflow] alias '%s' pose sur la version %s (bootstrap)",
        PRODUCTION_ALIAS, version,
    )
