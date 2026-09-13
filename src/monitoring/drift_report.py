"""Job de detection de derive (Evidently) — one-shot, human-in-the-loop.

Reference : features derivees du jeu de validation + predictions du modele courant.
Current  : journal JSONL des predictions servies par l'API.
Sorties  : rapport HTML, metriques poussees au pushgateway (si PUSHGATEWAY_URL),
           exit code 1 si derive -> l'operateur decide de relancer `make pipeline`.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import load_params, resolve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DRIFT_COLUMNS = ["text_chars", "text_words", "prdtypecode", "proba"]


def build_reference(val_df: pd.DataFrame, model_dir=None) -> pd.DataFrame:
    """Predit sur le jeu de validation avec le modele courant -> memes colonnes
    que le journal JSONL (le predictor charge Registry ou local, comme l'API).

    Les longueurs sont mesurees sur text_combined (texte nettoye) : l'API mesure
    la meme representation (cf. metrics.log_prediction), sinon derive artificielle.
    """
    from src.models.predict_model import BaselinePredictor

    predictor = BaselinePredictor(model_dir=model_dir)
    rows = []
    for text in val_df["text_combined"].fillna(""):
        top = predictor.predict(str(text), top_k=1)[0]
        rows.append({
            "text_chars": len(str(text)),
            "text_words": len(str(text).split()),
            "prdtypecode": top["prdtypecode"],
            "proba": top["proba"],
        })
    return pd.DataFrame(rows)


def load_current(log_path: Path, window: int | None = None) -> pd.DataFrame:
    """Charge le journal JSONL. `window` : ne garder que les N dernieres lignes
    (fenetre glissante — evite de melanger tout l'historique et borne l'I/O)."""
    lines = [line for line in log_path.read_text().splitlines() if line]
    if window:
        lines = lines[-window:]
    entries = [json.loads(line) for line in lines]
    # colonnes explicites : un journal vide ne doit pas lever KeyError avant
    # le garde-fou min_current_rows
    return pd.DataFrame(entries, columns=DRIFT_COLUMNS)


def evaluate_drift(reference: pd.DataFrame, current: pd.DataFrame, reports_dir: Path) -> dict:
    result = Report([DataDriftPreset()]).run(
        reference_data=reference[DRIFT_COLUMNS], current_data=current[DRIFT_COLUMNS]
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_name = f"drift_{stamp}.html"
    result.save_html(str(reports_dir / report_name))

    payload = json.loads(result.json())
    # DataDriftPreset -> metrique DriftedColumnsCount : value = {count, share}
    drift = next(
        m["value"] for m in payload["metrics"]
        if m["metric_name"].startswith("DriftedColumnsCount")
    )
    threshold = load_params()["monitoring"]["drift_share_threshold"]
    return {
        "drift_share": float(drift["share"]),
        "drifted_columns": int(drift["count"]),
        "drift_detected": float(drift["share"]) >= threshold,
        "report_name": report_name,
    }


def push_metrics(result: dict) -> None:
    """Pousse les gauges vers le pushgateway (best-effort : le rapport HTML et
    l'exit code restent valides meme si le pushgateway est absent)."""
    url = os.environ.get("PUSHGATEWAY_URL")
    if not url:
        return
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    Gauge("rakuten_drift_share", "Part de colonnes derivees (Evidently)",
          registry=registry).set(result["drift_share"])
    Gauge("rakuten_drift_detected", "1 si derive au-dela du seuil",
          registry=registry).set(int(result["drift_detected"]))
    try:
        push_to_gateway(url, job="drift", registry=registry)
        log.info("Metriques poussees vers %s", url)
    except OSError as exc:
        log.warning("Pushgateway injoignable (%s) — metriques non publiees", exc)


def main() -> int:
    params = load_params()["monitoring"]
    log_path = resolve(params["predictions_log"])
    if not log_path.exists():
        log.error("%s introuvable — envoyer du trafic d'abord (make traffic)", log_path)
        return 2
    current = load_current(log_path, window=params.get("current_window"))
    if len(current) < params["min_current_rows"]:
        log.error("Seulement %d predictions (< %d) — pas assez pour statuer",
                  len(current), params["min_current_rows"])
        return 2

    val = pd.read_parquet(
        resolve(load_params()["data"]["processed_dir"]) / "val.parquet"
    )
    sample_size = min(params["reference_sample"], len(val))
    reference = build_reference(val.sample(sample_size, random_state=42))
    result = evaluate_drift(reference, current, resolve(params["reports_dir"]))
    push_metrics(result)

    log.info("drift_share=%.2f (seuil %.2f) — rapport : %s",
             result["drift_share"], params["drift_share_threshold"],
             result["report_name"])
    if result["drift_detected"]:
        log.warning("DERIVE DETECTEE — examiner le rapport puis relancer "
                    "`make pipeline` et promouvoir la nouvelle version si conforme")
        return 1
    log.info("Pas de derive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
