"""Chargement centralise de la configuration (params.yaml)."""

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"


@lru_cache(maxsize=1)
def load_params() -> dict:
    with open(PARAMS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    """Resout un chemin relatif de params.yaml par rapport a la racine du repo."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
