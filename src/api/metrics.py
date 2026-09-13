"""Metriques Prometheus et journal des predictions (Phase 4).

- instrumentator : latence/debit/status par endpoint (+ /metrics)
- metriques custom : volume par classe predite, distribution de confiance
- journal JSONL : une ligne par prediction si PREDICTIONS_LOG_PATH est defini
  (desactive sinon — dev local et tests Phase 1-3 inchanges)

NB registre Prometheus : les collecteurs vivent dans le registre par defaut
(process-local). Garder UVICORN_WORKERS=1 par conteneur et scaler via les
replicas (`make up REPLICAS=N`, chaque replica scrape separement) — avec
plusieurs workers dans un meme conteneur, chaque scrape ne verrait qu'un
worker. La journalisation et les metriques sont best-effort : une panne de
telemetrie ne doit jamais transformer une inference reussie en 500.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.data.text_cleaning import build_text_combined

log = logging.getLogger(__name__)

PREDICTIONS_TOTAL = Counter(
    "rakuten_predictions_total",
    "Nombre de predictions servies, par classe predite (top-1)",
    ["prdtypecode"],
)
PREDICTION_CONFIDENCE = Histogram(
    "rakuten_prediction_confidence",
    "Probabilite de la classe top-1",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


def setup_metrics(app) -> None:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def observe_prediction(top: dict) -> None:
    PREDICTIONS_TOTAL.labels(prdtypecode=str(top["prdtypecode"])).inc()
    PREDICTION_CONFIDENCE.observe(top["proba"])


def log_prediction(designation: str, description: str | None, top: dict) -> None:
    """Journalise une prediction (best-effort).

    Les longueurs sont mesurees sur le TEXTE COMBINE NETTOYE — la meme
    representation que la reference de derive (build_reference lit text_combined),
    sinon un designation brut court paraitrait systematiquement decale.
    """
    log_path = os.environ.get("PREDICTIONS_LOG_PATH")
    if not log_path:
        return
    try:
        combined = build_text_combined(designation, description)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text_chars": len(combined),
            "text_words": len(combined.split()),
            "prdtypecode": top["prdtypecode"],
            "proba": top["proba"],
        }
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_APPEND : ecriture atomique de petites lignes, ok multi-replicas
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        # Volume plein/inaccessible : on degrade la telemetrie, pas l'inference
        log.warning("Journalisation de prediction impossible : %s", exc)
