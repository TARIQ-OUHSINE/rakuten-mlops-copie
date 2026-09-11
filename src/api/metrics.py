"""Metriques Prometheus et journal des predictions (Phase 4).

- instrumentator : latence/debit/status par endpoint (+ /metrics)
- metriques custom : volume par classe predite, distribution de confiance
- journal JSONL : une ligne par prediction si PREDICTIONS_LOG_PATH est defini
  (desactive sinon — dev local et tests Phase 1-3 inchanges)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

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
    log_path = os.environ.get("PREDICTIONS_LOG_PATH")
    if not log_path:
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "designation_chars": len(designation),
        "designation_words": len(designation.split()),
        "description_chars": len(description) if description else 0,
        "prdtypecode": top["prdtypecode"],
        "proba": top["proba"],
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_APPEND : ecriture atomique de petites lignes, ok multi-replicas
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
