"""Auth par cle d'API : header X-API-Key compare a la variable d'env API_KEY.

Si API_KEY n'est pas definie, l'auth est desactivee (dev local, tests Phase 1/2).
"""

import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = os.environ.get("API_KEY")
    if not expected:
        return
    if key is None or not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Cle d'API invalide ou absente")
