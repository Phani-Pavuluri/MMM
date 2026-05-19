"""Auto-generated model_card.md from extension reports and decision bundles."""

from __future__ import annotations

from typing import Any

MODEL_CARD_SECTIONS = (
    "model purpose",
    "supported decisions",
    "unsupported questions",
    "training data summary",
    "calibration evidence",
    "separability warnings",
    "planning assumptions",
    "uncertainty disclosure",
    "governance status",
    "release approval state",
)


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def generate_model_card(
    *,
    extension_report: dict[str, Any] | None = None,
    decision_bundle: dict[str, Any] | None = None,
) -> str:
    """Build ``model_card.md`` content from run artifacts (not hand-maintained)."""
    er = extension_report or {}
    bundle = decision_bundle or er.get("decision_bundle") if isinstance(er.get("decision_bundle"), dict) else {}
    if not bundle and isinstance(er.get("decision_bundle"), dict):
        bundle = er["decision_bundle"]

    cfg_snap = bundle.get("resolved_config_snapshot") or {}
    mr_fw = er.get("model_release", {})
    mr_framework = mr_fw.get("framework") if isinstance(mr_fw, dict) else None
    framework = str(
        bundle.get("framework") or cfg_snap.get("framework") or mr_framework or "unknown"
    )
    run_env = str(bundle.get("run_environment") or cfg_snap.get("run_environment") or "unknown")

    purpose = (
        f"Marketing mix model ({framework}) for {run_env} decision support. "
        "Budget and scenario questions use full-panel Δμ simulation when a Ridge fit is available."
    )

    supported = [
        "Full-panel counterfactual Δμ simulation (Ridge BO path).",
        "Diagnostic response curves and ROI bridges (non-authoritative for budgeting).",
    ]
    if bundle.get("governance", {}).get("approved_for_optimization"):
        supported.append("Optimization subject to governance gates (when explicitly enabled).")

    unsupported = bundle.get("unsupported_questions") or []
    if not isinstance(unsupported, list):
        unsupported = []
    uq = bundle.get("decision_uncertainty") or er.get("decision_uncertainty") or {}
    disclosure = str(uq.get("disclosure_text") or "See decision_uncertainty artifact.")

    fp = bundle.get("data_fingerprint") or er.get("data_fingerprint") or {}
    n_rows = fp.get("n_rows", "—")
    data = cfg_snap.get("data") or {}
    training = (
        f"- Rows: {n_rows}\n"
        f"- Target: {data.get('target_column', '—')}\n"
        f"- Channels: {', '.join(data.get('channel_columns') or []) or '—'}\n"
        f"- Controls: {', '.join(data.get('control_columns') or []) or 'none'}\n"
        f"- Data version id: {data.get('data_version_id') or 'not set'}\n"
    )

    cal = bundle.get("calibration_summary") or er.get("calibration_summary") or {}
    cal_lines = [
        f"- Replay calibration active: {cal.get('replay_calibration_active')}",
        f"- Replay loss: {cal.get('replay_loss')}",
    ]

    sep = er.get("feature_separability") or er.get("identifiability") or {}
    sep_warn = "No separability extension block in this run."
    if isinstance(sep, dict) and sep:
        sep_warn = json_pretty(sep)

    pa = bundle.get("planning_assumptions") or {}
    pa_body = json_pretty(pa) if pa else "Not attached to this bundle (train-time extension only)."

    gov = bundle.get("governance") or er.get("governance") or {}
    gov_body = (
        f"- Approved for optimization: {gov.get('approved_for_optimization')}\n"
        f"- Approved for reporting: {gov.get('approved_for_reporting')}\n"
        f"- Artifact tier: {bundle.get('artifact_tier', '—')}\n"
    )

    release = bundle.get("model_release") or er.get("model_release") or {}
    release_body = json_pretty(release) if release else "No model_release block."

    parts = [
        "# Model card\n",
        "_Auto-generated from extension_report / decision_bundle. Do not edit by hand._\n",
        _section("Model purpose", purpose),
        _section("Supported decisions", "\n".join(f"- {s}" for s in supported)),
        _section("Unsupported questions", "\n".join(f"- {u}" for u in unsupported) or "See bundle."),
        _section("Training data summary", training),
        _section("Calibration evidence", "\n".join(cal_lines)),
        _section("Separability warnings", sep_warn),
        _section("Planning assumptions", pa_body),
        _section("Uncertainty disclosure", disclosure),
        _section("Governance status", gov_body),
        _section("Release approval state", release_body),
    ]
    return "\n".join(parts)


def json_pretty(obj: Any) -> str:
    import json

    return "```json\n" + json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n```\n"


def required_sections_present(markdown: str) -> list[str]:
    """Return missing required section headings (for tests)."""
    lower = markdown.lower()
    missing = []
    for sec in MODEL_CARD_SECTIONS:
        if f"## {sec}" not in lower:
            missing.append(sec)
    return missing
