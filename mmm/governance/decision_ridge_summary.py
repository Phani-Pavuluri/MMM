"""Prod decide-time validation for ``ridge_fit_summary`` completeness and transform alignment."""

from __future__ import annotations

from typing import Any

from mmm.config.schema import Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.config.transform_policy import TRANSFORM_POLICY_VERSION
from mmm.config.validators import PROD_RIDGE_BO_MODEL_FORM_CONTRACTS
from mmm.governance.model_form_policy import assert_prod_decision_not_log_log
from mmm.governance.policy import PolicyError

_REQUIRED_BEST_PARAM_KEYS = ("decay", "hill_half", "hill_slope")


def _transform_policy_from_report(er: dict[str, Any]) -> dict[str, Any]:
    tp = er.get("transform_policy")
    if isinstance(tp, dict):
        return tp
    rfs = er.get("ridge_fit_summary")
    if isinstance(rfs, dict):
        nested = rfs.get("transform_policy")
        if isinstance(nested, dict):
            return nested
    return {}


def validate_ridge_fit_summary_for_prod_decide(cfg: MMMConfig, extension_report: dict[str, Any]) -> None:
    """
    Fail closed when Ridge full-panel decide inputs are incomplete or inconsistent with prod contract.
    """
    if cfg.run_environment != RunEnvironment.PROD:
        return
    if cfg.framework != Framework.RIDGE_BO:
        return

    assert_prod_decision_not_log_log(cfg, extension_report)

    expected_contract = PROD_RIDGE_BO_MODEL_FORM_CONTRACTS.get(ModelForm.SEMI_LOG)
    got_contract = (cfg.prod_canonical_modeling_contract_id or "").strip()
    if expected_contract and got_contract != expected_contract:
        raise PolicyError(
            f"prod_canonical_modeling_contract_id must be {expected_contract!r} for semi_log Ridge prod decide "
            f"(got {got_contract!r})"
        )

    rfs = extension_report.get("ridge_fit_summary")
    if not isinstance(rfs, dict):
        raise PolicyError("extension_report.ridge_fit_summary must be a dict for prod decide")

    coef = rfs.get("coef")
    intercept = rfs.get("intercept")
    if not coef:
        raise PolicyError("ridge_fit_summary.coef is required for prod decide")
    if intercept is None:
        raise PolicyError("ridge_fit_summary.intercept is required for prod decide")

    mf = str(rfs.get("model_form") or cfg.model_form.value)
    if mf == ModelForm.LOG_LOG.value:
        raise PolicyError("ridge_fit_summary.model_form=log_log is forbidden for prod decide")
    if mf != ModelForm.SEMI_LOG.value:
        raise PolicyError(f"ridge_fit_summary.model_form must be semi_log for prod decide (got {mf!r})")

    bp = rfs.get("best_params")
    if not isinstance(bp, dict):
        raise PolicyError("ridge_fit_summary.best_params must be a dict for prod decide")
    missing_bp = [k for k in _REQUIRED_BEST_PARAM_KEYS if k not in bp]
    if missing_bp:
        raise PolicyError(f"ridge_fit_summary.best_params missing keys for prod decide: {missing_bp}")

    tp = _transform_policy_from_report(extension_report)
    if not tp:
        raise PolicyError(
            "extension_report.transform_policy (or ridge_fit_summary.transform_policy) required for prod decide"
        )
    if str(tp.get("adstock")) != "geometric":
        raise PolicyError(f"transform_policy.adstock must be geometric for prod decide (got {tp.get('adstock')!r})")
    if str(tp.get("saturation")) != "hill":
        raise PolicyError(f"transform_policy.saturation must be hill for prod decide (got {tp.get('saturation')!r})")

    policy_version = str(tp.get("policy_version") or "")
    if policy_version and policy_version != TRANSFORM_POLICY_VERSION:
        raise PolicyError(
            f"transform_policy.policy_version mismatch: expected {TRANSFORM_POLICY_VERSION!r}, got {policy_version!r}"
        )

    decay = float(bp["decay"])
    hill_half = float(bp["hill_half"])
    hill_slope = float(bp["hill_slope"])
    if hill_half <= 0 or hill_slope <= 0 or not (0.0 < decay < 1.0):
        raise PolicyError(
            "ridge_fit_summary.best_params must have 0<decay<1 and hill_half,hill_slope>0 for prod decide"
        )

    if str(tp.get("framework")) and str(tp.get("framework")) != cfg.framework.value:
        raise PolicyError("transform_policy.framework does not match decide config framework")
    if str(tp.get("model_form")) and str(tp.get("model_form")) != cfg.model_form.value:
        raise PolicyError("transform_policy.model_form does not match decide config model_form")
    if str(tp.get("adstock")) != cfg.transforms.adstock:
        raise PolicyError(
            f"decide config transforms.adstock={cfg.transforms.adstock!r} disagrees with "
            f"training transform_policy.adstock={tp.get('adstock')!r}"
        )
    if str(tp.get("saturation")) != cfg.transforms.saturation:
        raise PolicyError(
            f"decide config transforms.saturation={cfg.transforms.saturation!r} disagrees with "
            f"training transform_policy.saturation={tp.get('saturation')!r}"
        )

    train_fp = extension_report.get("data_fingerprint") or extension_report.get("panel_fingerprint")
    lineage_fp = rfs.get("data_fingerprint") or rfs.get("panel_fingerprint")
    if not train_fp and not lineage_fp:
        raise PolicyError(
            "extension_report must include data_fingerprint (or panel_fingerprint) for prod decide lineage"
        )
