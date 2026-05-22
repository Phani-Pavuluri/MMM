"""Model-form policy: LOG_LOG is research-only until formally validated for prod."""

from __future__ import annotations

from typing import Any

from mmm.config.schema import Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.governance.policy import PolicyError

LOG_LOG_PROD_POLICY_MESSAGE = (
    "LOG_LOG is research-only until formal elasticity and Δμ recovery validation is complete. "
    "Use SEMI_LOG for prod Ridge decisions."
)

LOG_LOG_HIERARCHY_POLICY_MESSAGE = (
    "LOG_LOG hierarchy support is blocked pending formal model-form validation."
)

LOG_LOG_UNSUPPORTED_QUESTIONS: tuple[str, ...] = (
    "Production log-log budget optimization.",
    "Production elasticity interpretation from LOG_LOG.",
    "Hierarchical borrowing over LOG_LOG coefficients.",
)


def _normalize_model_form(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, ModelForm):
        return value.value
    s = str(value).strip().lower()
    return s if s in {ModelForm.SEMI_LOG.value, ModelForm.LOG_LOG.value} else None


def is_log_log_model_form(value: Any) -> bool:
    return _normalize_model_form(value) == ModelForm.LOG_LOG.value


def assert_log_log_not_in_prod_config(config: MMMConfig) -> None:
    """Block ``model_form=log_log`` when ``run_environment=prod``."""
    if config.run_environment == RunEnvironment.PROD and config.model_form == ModelForm.LOG_LOG:
        raise PolicyError(LOG_LOG_PROD_POLICY_MESSAGE)


def _hierarchy_active_for_model_form_guard(config: MMMConfig) -> bool:
    return bool(config.hierarchy.enabled) or (
        config.framework == Framework.BAYESIAN and bool(config.bayesian.use_hierarchy)
    )


def assert_log_log_hierarchy_blocked(config: MMMConfig) -> None:
    if _hierarchy_active_for_model_form_guard(config) and config.model_form == ModelForm.LOG_LOG:
        raise PolicyError(LOG_LOG_HIERARCHY_POLICY_MESSAGE)


def detect_log_log_in_extension_report(extension_report: dict[str, Any] | None) -> bool:
    """True when extension artifacts indicate LOG_LOG (e.g. stale training reports)."""
    if not isinstance(extension_report, dict):
        return False
    econ = extension_report.get("economics_contract")
    if isinstance(econ, dict) and is_log_log_model_form(econ.get("model_form")):
        return True
    tp = extension_report.get("transform_policy")
    if isinstance(tp, dict):
        mf = tp.get("model_form") or (tp.get("resolved") or {}).get("model_form")
        if is_log_log_model_form(mf):
            return True
    rs = extension_report.get("ridge_fit_summary")
    if isinstance(rs, dict) and is_log_log_model_form(rs.get("model_form")):
        return True
    snap = extension_report.get("resolved_config_snapshot")
    return isinstance(snap, dict) and is_log_log_model_form(snap.get("model_form"))


def assert_prod_decision_not_log_log(
    config: MMMConfig,
    extension_report: dict[str, Any] | None = None,
) -> None:
    """
    Prod simulate/optimize: reject LOG_LOG on live config and on extension-report lineage.
    """
    if config.run_environment != RunEnvironment.PROD:
        return
    if config.model_form == ModelForm.LOG_LOG:
        raise PolicyError(LOG_LOG_PROD_POLICY_MESSAGE)
    if detect_log_log_in_extension_report(extension_report):
        raise PolicyError(
            f"{LOG_LOG_PROD_POLICY_MESSAGE} "
            "(extension_report or decision bundle indicates model_form=log_log; "
            "stale LOG_LOG artifacts cannot be used for prod decisions)."
        )


def log_log_active(
    config: MMMConfig,
    extension_report: dict[str, Any] | None = None,
) -> bool:
    if config.model_form == ModelForm.LOG_LOG:
        return True
    return detect_log_log_in_extension_report(extension_report)


def log_log_unsupported_questions(
    config: MMMConfig,
    extension_report: dict[str, Any] | None = None,
) -> list[str]:
    if not log_log_active(config, extension_report):
        return []
    return list(LOG_LOG_UNSUPPORTED_QUESTIONS)
