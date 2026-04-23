"""Reject misleading finance phrasing for proxy / diagnostic reporting surfaces."""

from __future__ import annotations

_FORBIDDEN = (
    "true roi",
    "true roas",
    "channel contribution dollars",
    "exact dollar attribution",
    "audited roi",
)


def assert_safe_reporting_language(text: str, *, context: str = "report") -> None:
    lower = text.lower()
    for phrase in _FORBIDDEN:
        if phrase in lower:
            raise ValueError(
                f"{context}: forbidden misleading phrase {phrase!r} for proxy/diagnostic metrics "
                "(use Δμ / full-model simulation language instead)."
            )
