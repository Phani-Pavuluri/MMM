"""HMAC-SHA256 signing for replay / calibration payload dictionaries."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_payload(payload: dict[str, Any], secret: str) -> str:
    """Return hex digest of HMAC-SHA256 over canonical JSON."""
    return hmac.new(secret.encode("utf-8"), _canonical_bytes(payload), hashlib.sha256).hexdigest()


def verify_payload(payload: dict[str, Any], signature: str, secret: str) -> bool:
    """Constant-time compare to expected signature."""
    try:
        expected = sign_payload(payload, secret)
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(expected, signature)
