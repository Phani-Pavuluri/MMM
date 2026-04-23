"""Central runtime policy and guardrails for decision paths (prod fail-closed)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field

from mmm.config.schema import CVSplitAxis, Framework, MMMConfig, RunEnvironment
from mmm.contracts.estimands import EstimandKind
from mmm.governance.environment_policy import (
    EnvironmentPolicy,
    approved_for_optimization_with_policy,
    policy_for_environment,
)
from mmm.governance.model_release import ModelReleaseState
from mmm.governance.semantics import ArtifactTier, Surface


class PolicyError(RuntimeError):
    """Raised when a runtime or configuration policy blocks a decision path."""


class RuntimePolicy(BaseModel):
    """Resolved policy for decision simulate / optimize (single source of truth for prod gates)."""

    prod: bool
    require_planning_allowed: bool = True
    require_panel_qa_pass: bool = True
    require_replay_calibration: bool = True
    allow_bayesian_decisioning: bool = False
    allowed_cv_modes: list[str] = Field(default_factory=lambda: ["calendar"])
    allow_unsafe_decision_apis: bool = False

    def policy_fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


ALLOWED_INPUT_TIERS: dict[Surface, frozenset[ArtifactTier]] = {
    Surface.DIAGNOSTIC: frozenset({ArtifactTier.DIAGNOSTIC, ArtifactTier.RESEARCH, ArtifactTier.DECISION}),
    Surface.RESEARCH: frozenset({ArtifactTier.RESEARCH, ArtifactTier.DECISION}),
    Surface.DECISION: frozenset({ArtifactTier.DECISION}),
}


def runtime_policy_from_config(cfg: MMMConfig) -> RuntimePolicy:
    return RuntimePolicy(
        prod=cfg.run_environment == RunEnvironment.PROD,
        require_planning_allowed=True,
        require_panel_qa_pass=True,
        require_replay_calibration=True,
        allow_bayesian_decisioning=False,
        allowed_cv_modes=["calendar"],
        allow_unsafe_decision_apis=bool(cfg.allow_unsafe_decision_apis),
    )


def _tier_from_value(v: Any) -> ArtifactTier:
    if isinstance(v, ArtifactTier):
        return v
    s = str(v or "").lower()
    for t in ArtifactTier:
        if t.value == s:
            return t
    raise PolicyError(f"unknown artifact tier: {v!r}")


def require_surface_compatible(artifact_tier: ArtifactTier | str, surface: Surface) -> None:
    tier = _tier_from_value(artifact_tier)
    allowed = ALLOWED_INPUT_TIERS.get(surface)
    if allowed is None or tier not in allowed:
        names = sorted(x.value for x in allowed) if allowed else []
        raise PolicyError(
            f"surface={surface.value} rejects artifact_tier={tier.value} (allowed tiers: {names})"
        )


def require_decision_safe_result(result: dict[str, Any], policy: RuntimePolicy) -> None:
    """Require decision-grade JSON (from writer) before feeding optimizers / decision APIs."""
    if not policy.prod:
        return
    if not result.get("decision_safe"):
        raise PolicyError("prod decision path requires decision_safe=True on result payload")
    if result.get("estimand_kind") != EstimandKind.FULL_PANEL_DELTA_MU.value:
        raise PolicyError(
            "prod decision path requires estimand_kind=full_panel_delta_mu "
            f"(got {result.get('estimand_kind')!r}); approximate quantities cannot substitute."
        )
    if result.get("semantics") != "full_panel_delta_mu":
        raise PolicyError(
            f"prod decision path requires semantics=full_panel_delta_mu (got {result.get('semantics')!r})"
        )
    tier = result.get("tier") or result.get("artifact_tier")
    if str(tier) != ArtifactTier.DECISION.value:
        raise PolicyError(f"prod decision path requires tier=decision (got {tier!r})")


def require_identifiability_for_prod_decision(cfg: MMMConfig, er: dict[str, Any], policy: RuntimePolicy) -> None:
    """Block prod decision paths when extension identifiability is materially worse than governance limits."""
    if not policy.prod:
        return
    idv = er.get("identifiability") if isinstance(er, dict) else {}
    score = float(idv.get("identifiability_score", 0.0)) if isinstance(idv, dict) else 0.0
    gv = cfg.extensions.governance
    lim = float(gv.max_identifiability_risk)
    margin = float(gv.identifiability_decision_safety_margin)
    threshold = lim * margin
    if score <= threshold + 1e-12:
        return
    from mmm.governance.identifiability_waiver import parse_waiver_from_extension, waiver_allows_identifiability_block

    waiver = parse_waiver_from_extension(er)
    mr = er.get("model_release") if isinstance(er, dict) else {}
    mr_id = (
        str(abs(hash(tuple(sorted((str(k), str(v)) for k, v in mr.items())))))
        if isinstance(mr, dict) and mr
        else None
    )
    cf = er.get("config_fingerprint_sha256") if isinstance(er, dict) else None
    dvid = str(cfg.data.data_version_id or "").strip() or None
    ok, used = waiver_allows_identifiability_block(
        waiver=waiver,
        score=score,
        threshold=threshold,
        allow_waiver_policy=bool(gv.allow_identifiability_waiver),
        model_release_id=mr_id,
        config_fingerprint_sha256=str(cf) if cf else None,
        dataset_snapshot_id=dvid,
    )
    if not ok:
        raise PolicyError(
            f"prod decision path blocked: identifiability_score={score:.4f} exceeds threshold={threshold:.4f} "
            f"(max_identifiability_risk={lim} * identifiability_decision_safety_margin={margin}). "
            "Provide extensions.governance.allow_identifiability_waiver=True and a validated "
            "extension_report.identifiability_waiver artifact, or reduce identifiability risk."
        )
    if used is not None and isinstance(er, dict):
        er["_identifiability_waiver_applied"] = used.model_dump(mode="json")


def require_safe_cv(cv_mode: str, policy: RuntimePolicy) -> None:
    if not policy.prod:
        return
    if cv_mode not in policy.allowed_cv_modes:
        raise PolicyError(
            f"prod CV policy: mode {cv_mode!r} not in allowed_cv_modes={policy.allowed_cv_modes!r}"
        )


def cv_mode_key_from_config(cfg: MMMConfig) -> str:
    if cfg.cv.split_axis == CVSplitAxis.CALENDAR_WEEK:
        return "calendar"
    return str(cfg.cv.split_axis.value)


def require_planning_allowed(release_state: str | None, policy: RuntimePolicy) -> None:
    if not policy.prod or not policy.require_planning_allowed:
        return
    if release_state != ModelReleaseState.PLANNING_ALLOWED.value:
        raise PolicyError(
            f"prod requires model_release.state=planning_allowed (got {release_state!r})"
        )


def require_panel_qa_pass(panel_qa_summary: dict[str, Any] | None, policy: RuntimePolicy) -> None:
    if not policy.prod or not policy.require_panel_qa_pass:
        return
    if not isinstance(panel_qa_summary, dict):
        raise PolicyError("prod requires extension_report.panel_qa dict")
    sev = str(panel_qa_summary.get("max_severity", "")).lower()
    if sev == "block":
        raise PolicyError("prod blocks decision paths when panel_qa.max_severity=block")


def require_replay_calibration(
    calibration_summary: dict[str, Any] | None,
    experiment_matching: dict[str, Any] | None,
    policy: RuntimePolicy,
) -> None:
    """Require replay calibration evidence on extension (prod)."""
    if not policy.prod or not policy.require_replay_calibration:
        return
    if isinstance(calibration_summary, dict) and bool(calibration_summary):
        return
    if isinstance(experiment_matching, dict) and bool(experiment_matching):
        return
    raise PolicyError(
        "prod requires extension_report.calibration_summary or experiment_matching (non-empty) for replay gate"
    )


def require_bayesian_block(model_family: Framework, policy: RuntimePolicy) -> None:
    if not policy.prod:
        return
    if model_family == Framework.BAYESIAN and not policy.allow_bayesian_decisioning:
        raise PolicyError(
            "prod blocks Bayesian decisioning: RuntimePolicy.allow_bayesian_decisioning=False. "
            "Bayesian estimation/diagnostics are allowed in research/dev, but prod decision surfaces "
            "(simulate/optimize-budget) cannot treat posterior draw machinery as decision-grade truth. "
            "See artifact key bayesian_prod_policy on Bayesian bundles for allowed_surfaces metadata."
        )


def require_allow_unsafe(policy: RuntimePolicy) -> None:
    if policy.prod and policy.allow_unsafe_decision_apis:
        raise PolicyError("prod requires allow_unsafe_decision_apis=False on config")


__all__ = [
    "ALLOWED_INPUT_TIERS",
    "EnvironmentPolicy",
    "PolicyError",
    "RuntimePolicy",
    "approved_for_optimization_with_policy",
    "cv_mode_key_from_config",
    "policy_for_environment",
    "require_allow_unsafe",
    "require_bayesian_block",
    "require_decision_safe_result",
    "require_panel_qa_pass",
    "require_planning_allowed",
    "require_replay_calibration",
    "require_safe_cv",
    "require_surface_compatible",
    "runtime_policy_from_config",
]
