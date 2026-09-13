"""Envoie du trafic de demo a l'API depuis le jeu de validation.

Usage : python -m src.monitoring.send_traffic [N] [--drift]
--drift : tronque les textes a 3 mots pour simuler une derive de distribution.
"""

import os
import sys

import httpx
import pandas as pd

from src.config import load_params, resolve

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    drift = "--drift" in sys.argv
    val = pd.read_parquet(
        resolve(load_params()["data"]["processed_dir"]) / "val.parquet"
    ).sample(n, random_state=None if drift else 42)

    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    ok = 0
    with httpx.Client(timeout=30) as client:
        for text in val["text_combined"]:
            payload = " ".join(str(text).split()[:3]) if drift else str(text)
            try:
                r = client.post(
                    f"{API_URL}/predict",
                    json={"designation": payload or "produit"},
                    headers=headers,
                )
                ok += r.status_code == 200
            except httpx.HTTPError as exc:
                # replica en cours de demarrage, reset transitoire : on continue
                print(f"requete echouee : {exc}")
    print(f"{ok}/{n} predictions ok ({'drift simule' if drift else 'normal'})")
    # exit non nul si au moins une requete a echoue (ex: mauvaise cle) : une
    # automation ne doit pas continuer avec un journal incomplet en croyant OK
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
