"""Minimal TrustReport-shaped diagnostic stub for Bayes-H3 sandbox (not production TrustReport)."""

from __future__ import annotations

from typing import Any

from mmm.research.bayes_h3_sandbox.labels import apply_research_only_envelope


def build_diagnostic_trust_stub(
    *,
    posterior_summary: dict[str, Any] | None = None,
    convergence_diagnostics: dict[str, Any] | None = None,
    hierarchy_evidence: dict[str, Any] | None = None,
    pooling_diagnostics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a research-only diagnostic trust block — not a production TrustReport."""
    block: dict[str, Any] = {
        "trust_report_kind": "bayes_h3_diagnostic_stub",
        "posterior_summary": posterior_summary or {"status": "stub", "note": "diagnostic only"},
        "convergence_diagnostics": convergence_diagnostics or {"status": "stub"},
        "hierarchy_evidence": hierarchy_evidence or {},
        "pooling_diagnostics": pooling_diagnostics or {},
        "outputs_are_diagnostic_only": True,
    }
    if extra:
        block.update(extra)
    return apply_research_only_envelope(block)
