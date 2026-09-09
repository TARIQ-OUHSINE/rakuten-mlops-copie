"""Entrainement du modele baseline TF-IDF + LightGBM.

Entree  : data/processed/{train,val}.parquet (via src.data.make_dataset)
Sortie  : models/baseline/{model,vectorizer,label_encoder}.joblib + metrics.json
"""

import json
import logging

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from src.config import load_params, resolve
from src.features.build_features import fit_transform_texts, save_vectorizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_splits():
    processed = resolve(load_params()["data"]["processed_dir"])
    train_path, val_path = processed / "train.parquet", processed / "val.parquet"
    for path in (train_path, val_path):
        if not path.exists():
            raise SystemExit(
                f"{path} introuvable — lancer d'abord `make data` "
                "(python -m src.data.make_dataset)"
            )
    return pd.read_parquet(train_path), pd.read_parquet(val_path)


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict):
    vectorizer, x_train, x_val = fit_transform_texts(
        train_df["text_combined"].fillna(""),
        val_df["text_combined"].fillna(""),
        params["tfidf"],
    )
    log.info("TF-IDF : %s", x_train.shape)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["prdtypecode"])
    y_val = le.transform(val_df["prdtypecode"])

    lgbm_params = dict(params["lgbm"])
    lgbm_params["num_class"] = len(le.classes_)
    clf = lgb.LGBMClassifier(**lgbm_params)
    clf.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[
            lgb.early_stopping(params["early_stopping_rounds"], verbose=False),
        ],
    )
    return clf, vectorizer, le, x_val, y_val


def evaluate(clf, x_val, y_val) -> dict:
    y_pred = clf.predict(x_val)
    metrics = {
        "f1_weighted": round(f1_score(y_val, y_pred, average="weighted"), 4),
        "f1_macro": round(f1_score(y_val, y_pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "report": classification_report(y_val, y_pred, output_dict=True),
    }
    log.info(
        "val weighted F1 = %.4f | macro F1 = %.4f | acc = %.4f",
        metrics["f1_weighted"], metrics["f1_macro"], metrics["accuracy"],
    )
    return metrics


def save_artifacts(clf, vectorizer, le, metrics: dict) -> None:
    params = load_params()
    model_dir = resolve(params["model"]["dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, model_dir / "model.joblib")
    save_vectorizer(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(le, model_dir / "label_encoder.joblib")
    metrics["model_version"] = params["model"]["version"]
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("Artefacts sauvegardes dans %s", model_dir)


def main() -> None:
    params = load_params()
    train_df, val_df = load_splits()
    clf, vectorizer, le, x_val, y_val = train(train_df, val_df, params)
    metrics = evaluate(clf, x_val, y_val)
    save_artifacts(clf, vectorizer, le, metrics)


if __name__ == "__main__":
    main()
