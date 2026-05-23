"""Explicit human-reviewed model promotion (no auto-promotion)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal

from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.data.fingerprint import FINGERPRINT_VERSION
from mmm.governance.model_release import ModelReleaseState
from mmm.governance.policy import PolicyError

ApprovalStatus = Literal["approved", "revoked", "expired"]


@dataclass(frozen=True)
class PromotionRecord:
    promotion_id: str
    run_id: str
    model_id: str
    artifact_uri: str
    data_fingerprint: dict[str, Any]
    config_fingerprint: str
    model_fingerprint: str
    seed_resolution: dict[str, Any]
    promoted_by: str
    promoted_at: str
    approval_status: ApprovalStatus
    approval_notes: str
    allowed_surfaces: tuple[str, ...]
    expiration_date: str | None
    rollback_of: str | None
    parent_promotion_id: str | None
    governance_summary: dict[str, Any]
    calibration_summary: dict[str, Any]
    unsupported_questions: tuple[str, ...]
    signature_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["allowed_surfaces"] = list(self.allowed_surfaces)
        d["unsupported_questions"] = list(self.unsupported_questions)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PromotionRecord:
        return cls(
            promotion_id=str(d["promotion_id"]),
            run_id=str(d["run_id"]),
            model_id=str(d["model_id"]),
            artifact_uri=str(d["artifact_uri"]),
            data_fingerprint=dict(d.get("data_fingerprint") or {}),
            config_fingerprint=str(d["config_fingerprint"]),
            model_fingerprint=str(d["model_fingerprint"]),
            seed_resolution=dict(d.get("seed_resolution") or {}),
            promoted_by=str(d.get("promoted_by", "")),
            promoted_at=str(d.get("promoted_at", "")),
            approval_status=d.get("approval_status", "approved"),  # type: ignore[arg-type]
            approval_notes=str(d.get("approval_notes", "")),
            allowed_surfaces=tuple(str(x) for x in (d.get("allowed_surfaces") or [])),
            expiration_date=str(d["expiration_date"]) if d.get("expiration_date") else None,
            rollback_of=str(d["rollback_of"]) if d.get("rollback_of") else None,
            parent_promotion_id=str(d["parent_promotion_id"]) if d.get("parent_promotion_id") else None,
            governance_summary=dict(d.get("governance_summary") or {}),
            calibration_summary=dict(d.get("calibration_summary") or {}),
            unsupported_questions=tuple(str(x) for x in (d.get("unsupported_questions") or [])),
            signature_hash=str(d.get("signature_hash", "")),
        )


def promotion_signature_hash(record_body: dict[str, Any]) -> str:
    """Stable hash over promotion payload excluding ``signature_hash``."""
    body = {k: v for k, v in record_body.items() if k != "signature_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def build_promotion_record(
    *,
    promotion_id: str,
    run_id: str,
    model_id: str,
    artifact_uri: str,
    data_fingerprint: dict[str, Any],
    config_fingerprint: str,
    model_fingerprint: str,
    seed_resolution: dict[str, Any],
    promoted_by: str,
    governance_summary: dict[str, Any],
    calibration_summary: dict[str, Any],
    unsupported_questions: list[str],
    allowed_surfaces: list[str] | None = None,
    approval_notes: str = "",
    expiration_date: str | None = None,
    rollback_of: str | None = None,
    parent_promotion_id: str | None = None,
) -> PromotionRecord:
    promoted_at = datetime.now(timezone.utc).isoformat()
    body: dict[str, Any] = {
        "promotion_id": promotion_id,
        "run_id": run_id,
        "model_id": model_id,
        "artifact_uri": artifact_uri,
        "data_fingerprint": data_fingerprint,
        "config_fingerprint": config_fingerprint,
        "model_fingerprint": model_fingerprint,
        "seed_resolution": seed_resolution,
        "promoted_by": promoted_by,
        "promoted_at": promoted_at,
        "approval_status": "approved",
        "approval_notes": approval_notes,
        "allowed_surfaces": allowed_surfaces or ["simulate", "optimize_budget"],
        "expiration_date": expiration_date,
        "rollback_of": rollback_of,
        "parent_promotion_id": parent_promotion_id,
        "governance_summary": governance_summary,
        "calibration_summary": calibration_summary,
        "unsupported_questions": unsupported_questions,
    }
    sig = promotion_signature_hash(body)
    return PromotionRecord(
        promotion_id=promotion_id,
        run_id=run_id,
        model_id=model_id,
        artifact_uri=artifact_uri,
        data_fingerprint=data_fingerprint,
        config_fingerprint=config_fingerprint,
        model_fingerprint=model_fingerprint,
        seed_resolution=seed_resolution,
        promoted_by=promoted_by,
        promoted_at=promoted_at,
        approval_status="approved",
        approval_notes=approval_notes,
        allowed_surfaces=tuple(allowed_surfaces or ["simulate", "optimize_budget"]),
        expiration_date=expiration_date,
        rollback_of=rollback_of,
        parent_promotion_id=parent_promotion_id,
        governance_summary=governance_summary,
        calibration_summary=calibration_summary,
        unsupported_questions=tuple(unsupported_questions),
        signature_hash=sig,
    )


def validate_promotion_eligibility(
    *,
    config: MMMConfig,
    extension_report: dict[str, Any],
    data_fingerprint: dict[str, Any],
    config_fingerprint: str,
) -> None:
    """Fail closed when promotion prerequisites are not met."""
    gov = extension_report.get("governance") or {}
    if not bool(gov.get("approved_for_optimization")):
        raise PolicyError("promotion requires governance.approved_for_optimization=true")
    mr = extension_report.get("model_release") or {}
    state = str(mr.get("state", ""))
    if state not in (ModelReleaseState.PLANNING_ALLOWED.value, ModelReleaseState.REPORTING_ALLOWED.value):
        raise PolicyError(f"promotion requires model_release planning_allowed or reporting_allowed (got {state!r})")
    fp_ver = str(data_fingerprint.get("fingerprint_version", ""))
    if fp_ver and fp_ver != FINGERPRINT_VERSION:
        raise PolicyError(f"promotion requires data fingerprint {FINGERPRINT_VERSION!r} (got {fp_ver!r})")
    if not config_fingerprint:
        raise PolicyError("promotion requires config_fingerprint")
    if config.run_environment == RunEnvironment.PROD:
        from mmm.governance.policy import prod_replay_evidence_ok

        cal = extension_report.get("calibration_summary")
        em = extension_report.get("experiment_matching")
        ev = extension_report.get("evidence_weighted_replay_summary")
        if isinstance(ev, dict) and ev.get("n_evidence_units_used", 0):
            pass
        else:
            cal_js = cal if isinstance(cal, dict) else None
            em_js = em if isinstance(em, dict) else None
            ok, msg = prod_replay_evidence_ok(cal_js, em_js)
            if not ok:
                raise PolicyError(f"promotion prod replay evidence gate failed: {msg}")


def assert_promotion_valid_for_decision(
    record: PromotionRecord,
    *,
    surface: str,
    data_fingerprint: dict[str, Any] | None,
    config_fingerprint: str | None,
) -> None:
    if record.approval_status != "approved":
        raise PolicyError(f"promotion {record.promotion_id!r} status is {record.approval_status!r}")
    if record.expiration_date:
        exp = date.fromisoformat(record.expiration_date[:10])
        if date.today() > exp:
            raise PolicyError(f"promotion {record.promotion_id!r} expired on {record.expiration_date}")
    if surface not in record.allowed_surfaces:
        raise PolicyError(
            f"surface {surface!r} not in promotion allowed_surfaces {list(record.allowed_surfaces)!r}"
        )
    if data_fingerprint is not None:
        rec_fp = record.data_fingerprint.get("combined_hash") or record.data_fingerprint.get("hash")
        cur_fp = data_fingerprint.get("combined_hash") or data_fingerprint.get("hash")
        if rec_fp and cur_fp and str(rec_fp) != str(cur_fp):
            raise PolicyError("promotion data_fingerprint mismatch")
    if config_fingerprint and record.config_fingerprint != config_fingerprint:
        raise PolicyError("promotion config_fingerprint mismatch")
    body = record.to_dict()
    if promotion_signature_hash(body) != record.signature_hash:
        raise PolicyError(f"promotion {record.promotion_id!r} signature_hash mismatch (record tampered)")
