"""Auto-generated model_card.md from extension reports and decision bundles."""

from __future__ import annotations

import json
from typing import Any

MODEL_CARD_SECTIONS = (
    "intended use",
    "unsupported use",
    "assumptions",
    "risks",
    "calibration coverage",
    "governance summary",
    "decision boundaries",
    "reproducibility information",
    "lineage summary",
    "known limitations",
    "release notes",
    "approval section",
)


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def json_pretty(obj: Any) -> str:
    return "```json\n" + json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n```\n"


def generate_model_card(
    *,
    extension_report: dict[str, Any] | None = None,
    decision_bundle: dict[str, Any] | None = None,
) -> str:
    """Build stakeholder-facing ``model_card.md`` from run artifacts."""
    er = extension_report or {}
    bundle = decision_bundle if isinstance(decision_bundle, dict) else {}
    if not bundle and isinstance(er.get("decision_bundle"), dict):
        bundle = er["decision_bundle"]

    cfg_snap = bundle.get("resolved_config_snapshot") or {}
    mr = er.get("model_release") or bundle.get("model_release") or {}
    framework = str(
        bundle.get("framework") or cfg_snap.get("framework") or mr.get("framework") or "unknown"
    )
    run_env = str(bundle.get("run_environment") or cfg_snap.get("run_environment") or "unknown")

    intended = (
        f"Marketing mix model ({framework}) trained for **{run_env}** surfaces. "
        "Use for budget scenario exploration and optimization only when governance artifacts "
        "approve the run and full-panel Δμ simulation is available (Ridge BO production path)."
    )

    unsupported = bundle.get("unsupported_questions") or []
    if not isinstance(unsupported, list):
        unsupported = []
    unsupported_body = "\n".join(f"- {u}" for u in unsupported) or (
        "- Curve-based ROI or univariate response curves as budget truth.\n"
        "- Calibrated Ridge monetary confidence intervals.\n"
        "- Automatic retraining triggered by drift diagnostics."
    )

    pa = bundle.get("planning_assumptions") or er.get("planning_assumptions") or {}
    transform = er.get("transform_policy") or {}
    assumptions_body = (
        "- Planning uses full-panel counterfactual simulation when Ridge fit artifacts exist.\n"
        f"- Planning assumptions artifact: {'present' if pa else 'not attached to bundle'}.\n"
    )
    if pa:
        assumptions_body += json_pretty(pa)
    if transform:
        assumptions_body += "\nTransform policy:\n" + json_pretty(transform)

    risks: list[str] = []
    sep = er.get("feature_separability") or {}
    if isinstance(sep, dict) and sep.get("high_risk_pairs"):
        risks.append("Feature separability flagged high-risk channel pairs.")
    drift = er.get("drift_report") or {}
    if isinstance(drift, dict) and drift.get("drift_severity") in ("warning", "critical"):
        risks.append(f"Drift severity: {drift.get('drift_severity')} — review before decisioning.")
    cg = er.get("control_governance") or {}
    if isinstance(cg, dict) and cg.get("potentially_post_treatment"):
        risks.append("Potential post-treatment controls detected — verify causal timing.")
    if not risks:
        risks.append("No elevated separability/drift/control flags in extension report.")
    risks_body = "\n".join(f"- {r}" for r in risks)

    cal = bundle.get("calibration_summary") or er.get("calibration_summary") or {}
    cal_body = (
        f"- Replay calibration active: {cal.get('replay_calibration_active')}\n"
        f"- Replay loss: {cal.get('replay_loss')}\n"
        f"- Coverage note: calibration evidence is **evidence**, not an automatic approval gate.\n"
    )

    gov = bundle.get("governance") or er.get("governance") or {}
    gov_body = (
        f"- Approved for optimization: {gov.get('approved_for_optimization')}\n"
        f"- Approved for reporting: {gov.get('approved_for_reporting')}\n"
        f"- Artifact tier: {bundle.get('artifact_tier', er.get('artifact_tier', '—'))}\n"
        f"- Operational health: {json.dumps(er.get('operational_health') or {}, default=str)[:200]}…\n"
    )
    if gov.get("baseline_beat_waiver_active"):
        gov_body += (
            "- **WARNING:** baseline beat check waived (`require_beats_baselines_for_approval=false`).\n"
        )
    rhv = er.get("replay_holdout_validation") or {}
    if isinstance(rhv, dict) and rhv.get("status") not in (None, "skipped", "disabled"):
        gov_body += (
            f"- Replay holdout validation: status={rhv.get('status')}, "
            f"train_loss={rhv.get('train_replay_loss')}, holdout_loss={rhv.get('holdout_replay_loss')}\n"
        )

    boundaries = (
        "- **Decide:** full-panel Δμ (`simulate`) on Ridge BO when `allow_unsafe_decision_apis` policy permits.\n"
        "- **Diagnose:** response curves, ROI bridges, curve–Δμ alignment, drift, separability.\n"
        "- Curves explain; simulation decides (see `curve_decision_alignment` when present).\n"
    )
    align = er.get("curve_decision_alignment") or {}
    if isinstance(align, dict) and align.get("policy_note"):
        boundaries += f"- Alignment policy: {align['policy_note']}\n"

    seeds = er.get("seed_resolution") or bundle.get("seed_resolution") or {}
    fp = bundle.get("data_fingerprint") or er.get("data_fingerprint") or {}
    repro = (
        f"- Fingerprint version: {fp.get('fingerprint_version', '—')}\n"
        f"- Combined hash: {fp.get('sha256_combined', '—')}\n"
        f"- Config fingerprint: see `config_fingerprint` artifact in run dir.\n"
        f"- Seeds: {json.dumps(seeds, default=str) if seeds else 'see seed_resolution artifact'}\n"
    )
    backend = er.get("artifact_backend") or {}
    if backend:
        repro += f"- Artifact backend: {backend.get('backend')} ({backend.get('support_tier')})\n"

    lineage = (
        f"- Run environment: {run_env}\n"
        f"- Framework: {framework}\n"
        f"- Data version id: {(cfg_snap.get('data') or {}).get('data_version_id') or 'not set'}\n"
    )
    if isinstance(mr, dict) and mr:
        lineage += json_pretty(mr)

    uq = bundle.get("decision_uncertainty") or er.get("decision_uncertainty") or {}
    limitations = [str(uq.get("disclosure_text") or "See decision_uncertainty artifact.")]
    inv = uq.get("methods_investigated") or {}
    if inv:
        limitations.append(f"Uncertainty methods investigated: {json.dumps(inv)}")
    limitations.append("Drift and separability outputs do not auto-block training or planning.")
    limitations_body = "\n".join(f"- {x}" for x in limitations)

    release_body = json_pretty(mr) if mr else "No model_release block in this run."

    opt_gate = er.get("optimization_gate")
    opt_allowed = opt_gate.get("allowed") if isinstance(opt_gate, dict) else "—"
    approval = (
        f"- Model release state: {mr.get('state') if isinstance(mr, dict) else 'unknown'}\n"
        f"- Optimization gate allowed: {opt_allowed}\n"
        "- Human sign-off required before production budget commits.\n"
    )

    data = cfg_snap.get("data") or {}
    training_note = (
        f"Rows: {fp.get('n_rows', '—')}; target: {data.get('target_column', '—')}; "
        f"channels: {', '.join(data.get('channel_columns') or []) or '—'}."
    )

    parts = [
        "# Model card\n",
        "_Auto-generated from extension_report / decision_bundle. Suitable for PR/release review._\n",
        f"_Training snapshot: {training_note}_\n",
        _section("Intended use", intended),
        _section("Unsupported use", unsupported_body),
        _section("Assumptions", assumptions_body),
        _section("Risks", risks_body),
        _section("Calibration coverage", cal_body),
        _section("Governance summary", gov_body),
        _section("Decision boundaries", boundaries),
        _section("Reproducibility information", repro),
        _section("Lineage summary", lineage),
        _section("Known limitations", limitations_body),
        _section("Release notes", release_body),
        _section("Approval section", approval),
    ]
    return "\n".join(parts)


def required_sections_present(markdown: str) -> list[str]:
    """Return missing required section headings (for tests)."""
    lower = markdown.lower()
    missing = []
    for sec in MODEL_CARD_SECTIONS:
        if f"## {sec}" not in lower:
            missing.append(sec)
    return missing
