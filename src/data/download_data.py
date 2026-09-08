"""Verification de la presence des donnees brutes Rakuten.

Les CSV (52 MB) ne sont pas versionnes : ils proviennent du challenge
https://challengedata.ens.fr/challenges/35 et doivent etre places dans data/.
"""

import logging
import sys

from src.config import load_params, resolve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "raw_x": ["designation", "description", "productid", "imageid"],
    "raw_y": ["prdtypecode"],
}


def check_raw_data() -> bool:
    """Retourne True si les fichiers bruts sont presents et bien formes."""
    import pandas as pd

    params = load_params()["data"]
    ok = True
    for key in ("raw_x", "raw_y"):
        path = resolve(params[key])
        if not path.exists():
            log.error(
                "Fichier manquant : %s — telechargez les donnees du challenge "
                "Rakuten (https://challengedata.ens.fr/challenges/35) et placez "
                "les CSV dans data/.", path,
            )
            ok = False
            continue
        header = pd.read_csv(path, nrows=0)
        missing = set(EXPECTED_COLUMNS[key]) - set(header.columns)
        if missing:
            log.error("%s : colonnes manquantes %s", path.name, sorted(missing))
            ok = False
        else:
            log.info("OK %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return ok


if __name__ == "__main__":
    sys.exit(0 if check_raw_data() else 1)
