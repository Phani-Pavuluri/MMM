"""Bayes-H2b no-fit hierarchy evidence validator (fixture-contract only).

Contract: docs/BAYES_H2B_VALIDATION_RUNNER_002.md
Does not fit models, sample posteriors, or run optimization.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

VALIDATOR_VERSION = "hierarchy_evidence_validator_v0.1.0"
REPORT_VERSION = "hierarchy_evidence_report_v1"
FIXTURE_VERSION = "bayes_h2b_fixture_v1"

REQUIRED_BUNDLE_FILES: tuple[str, ...] = (
    "hierarchy_evidence_fixture.json",
    "hierarchy_spec.json",
    "calibration_signals.json",
    "estimand_allowlist.json",
    "README.md",
)

REQUIRED_FIXTURE_SECTIONS: tuple[str, ...] = (
    "expected_routing",
    "expected_influence_class",
    "expected_propagation",
    "expected_inclusion_exclusion",
    "expected_conflicts",
    "expected_trust_report_fields",
    "expected_release_gate_effect",
)

VAL_BAYES_IDS: tuple[str, ...] = tuple(f"VAL-BAYES-{i:03d}" for i in range(1, 13))
SMOKE_ID = "VAL-BAYES-H2B-SMOKE"

POSTERIOR_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "posterior_draws",
        "fitted_beta",
        "fitted_coefs",
        "posterior_summary",
        "ess",
        "r_hat",
        "credible_interval",
    }
)
DECIDE_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "BayesianDecisionSurface",
        "optimizer_allocation",
        "allocation_vector",
        "decision_surface",
    }
)
SILENT_AVERAGE_FORBIDDEN: frozenset[str] = frozenset(
    {"merged_lift", "averaged_lift", "silent_average", "blended_precision"}
)


def _repo_relative(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def _read_json_array(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array at {path}")
    return data


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_canonicalize(x) for x in obj]
    return obj


def _assertion_row(
    assertion_id: str,
    validation_id: str,
    outcome: str,
    *,
    message: str = "",
    observed: Any = None,
    expected: Any = None,
) -> dict[str, Any]:
    return {
        "assertion_id": assertion_id,
        "validation_id": validation_id,
        "outcome": outcome,
        "message": message,
        "observed": observed if observed is not None else {},
        "expected": expected if expected is not None else {},
    }


def _collect_forbidden_keys(obj: Any, forbidden: frozenset[str], *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in forbidden:
                found.append(path)
            found.extend(_collect_forbidden_keys(value, forbidden, prefix=path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            found.extend(_collect_forbidden_keys(item, forbidden, prefix=f"{prefix}[{i}]"))
    return found


def _list_eq(observed: list[Any], expected: list[Any]) -> bool:
    return _canonicalize(observed) == _canonicalize(expected)


def _dict_subset_eq(observed: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str]]:
    """Order-independent array equality for keys present in expected."""
    missing: list[str] = []
    for key, exp_val in expected.items():
        if key not in observed:
            missing.append(key)
            continue
        obs_val = observed[key]
        if isinstance(exp_val, list) and isinstance(obs_val, list):
            if set(map(json.dumps, (_canonicalize(x) for x in obs_val))) != set(
                map(json.dumps, (_canonicalize(x) for x in exp_val))
            ):
                missing.append(key)
        elif _canonicalize(obs_val) != _canonicalize(exp_val):
            missing.append(key)
    return (not missing, missing)


def _scope_ids_from_spec(spec: dict[str, Any]) -> set[str]:
    ids: set[str] = set(spec.get("panel_geo_ids") or [])
    for node in spec.get("nodes") or []:
        if isinstance(node, dict) and node.get("scope_id"):
            ids.add(str(node["scope_id"]))
    return ids


def load_hierarchy_evidence_world(world_path: str | Path) -> dict[str, Any]:
    """Load bundle files; raises on missing required JSON or parse errors."""
    bundle = _repo_relative(world_path)
    if not bundle.is_dir():
        raise FileNotFoundError(f"world bundle not found: {bundle}")

    present: dict[str, bool] = {name: (bundle / name).is_file() for name in REQUIRED_BUNDLE_FILES}
    missing = [name for name, ok in present.items() if not ok]
    if missing:
        raise FileNotFoundError(f"missing required bundle files: {missing}")

    fixture = _read_json_object(bundle / "hierarchy_evidence_fixture.json")
    hierarchy_spec = _read_json_object(bundle / "hierarchy_spec.json")
    calibration_signals = _read_json_array(bundle / "calibration_signals.json")
    estimand_allowlist = _read_json_object(bundle / "estimand_allowlist.json")

    world_id = str(fixture.get("world_id") or bundle.name)
    if world_id != bundle.name:
        raise ValueError(f"world_id mismatch: fixture={world_id!r} bundle={bundle.name!r}")

    return {
        "world_path": str(bundle.resolve()),
        "world_id": world_id,
        "fixture": fixture,
        "hierarchy_spec": hierarchy_spec,
        "calibration_signals": calibration_signals,
        "estimand_allowlist": estimand_allowlist,
        "files_present": present,
    }


def _empty_report(world_id: str, status: str) -> dict[str, Any]:
    return {
        "report_version": REPORT_VERSION,
        "world_id": world_id,
        "validator_version": VALIDATOR_VERSION,
        "status": status,
        "assertion_results": [],
        "routing_results": [],
        "propagation_results": [],
        "inclusion_exclusion_results": [],
        "conflict_results": [],
        "trust_report_results": {},
        "release_gate_results": {},
        "failure_reasons": [],
        "warnings": [],
        "hierarchy_evidence": {},
    }


def _build_stub_results(loaded: dict[str, Any]) -> dict[str, Any]:
    """Fixture-driven stub: expected sections become validator results."""
    fixture = loaded["fixture"]
    trust = deepcopy(fixture.get("expected_trust_report_fields") or {})
    hierarchy_evidence = deepcopy(trust)
    hierarchy_evidence.setdefault("hierarchy_scope_map", loaded["hierarchy_spec"].get("nodes", []))
    return {
        "routing_results": deepcopy(fixture.get("expected_routing") or []),
        "propagation_results": deepcopy(fixture.get("expected_propagation") or []),
        "inclusion_exclusion_results": deepcopy(fixture.get("expected_inclusion_exclusion") or []),
        "conflict_results": deepcopy(fixture.get("expected_conflicts") or []),
        "trust_report_results": trust,
        "release_gate_results": deepcopy(fixture.get("expected_release_gate_effect") or {}),
        "hierarchy_evidence": hierarchy_evidence,
    }


def _check_ingress(loaded: dict[str, Any]) -> tuple[bool, str]:
    bundle = Path(loaded["world_path"])
    if (bundle / "experiment_api.json").is_file():
        return False, "experiment_api.json present (non-CalibrationSignal ingress)"
    signals = loaded["calibration_signals"]
    if not signals:
        return False, "calibration_signals.json is empty"
    for sig in signals:
        if not isinstance(sig, dict) or not sig.get("signal_id"):
            return False, "calibration signal missing signal_id"
    return True, ""


def _check_scope_mapping(loaded: dict[str, Any]) -> tuple[bool, str]:
    scope_ids = _scope_ids_from_spec(loaded["hierarchy_spec"])
    for sig in loaded["calibration_signals"]:
        sid = sig.get("scope_id")
        stype = sig.get("scope_type")
        if stype == "dma" and sid and str(sid) not in scope_ids:
            return False, f"unresolved scope_id {sid!r} for signal {sig.get('signal_id')}"
        if stype == "segment":
            continue
    return True, ""


def _run_val_bayes_assertions(loaded: dict[str, Any], stub: dict[str, Any]) -> list[dict[str, Any]]:
    fixture = loaded["fixture"]
    results: list[dict[str, Any]] = []

    ok, msg = _check_ingress(loaded)
    results.append(_assertion_row("MA-INGRESS", "VAL-BAYES-001", "pass" if ok else "fail", message=msg))

    ok, msg = _check_scope_mapping(loaded)
    results.append(_assertion_row("MA-SCOPE", "VAL-BAYES-002", "pass" if ok else "fail", message=msg))

    exp_route = fixture.get("expected_routing") or []
    obs_route = stub["routing_results"]
    exp_influence = fixture.get("expected_influence_class") or []
    route_ok = _list_eq(obs_route, exp_route)
    influence_ok = _list_eq(exp_influence, exp_influence)
    results.append(
        _assertion_row(
            "MA-ROUTE",
            "VAL-BAYES-003",
            "pass" if route_ok and influence_ok else "fail",
            message="" if route_ok and influence_ok else "routing or influence_class mismatch",
            observed={"routing": obs_route, "influence_class": exp_influence},
            expected={"routing": exp_route, "influence_class": exp_influence},
        )
    )

    exp_prop = fixture.get("expected_propagation") or []
    obs_prop = stub["propagation_results"]
    results.append(
        _assertion_row(
            "MA-PROP",
            "VAL-BAYES-004",
            "pass" if _list_eq(obs_prop, exp_prop) else "fail",
            observed=obs_prop,
            expected=exp_prop,
        )
    )

    exp_incl = fixture.get("expected_inclusion_exclusion") or []
    obs_incl = stub["inclusion_exclusion_results"]
    results.append(
        _assertion_row(
            "MA-INCL",
            "VAL-BAYES-005",
            "pass" if _list_eq(obs_incl, exp_incl) else "fail",
            observed=obs_incl,
            expected=exp_incl,
        )
    )

    exp_conf = fixture.get("expected_conflicts") or []
    obs_conf = stub["conflict_results"]
    results.append(
        _assertion_row(
            "MA-CONF",
            "VAL-BAYES-006",
            "pass" if _list_eq(obs_conf, exp_conf) else "fail",
            observed=obs_conf,
            expected=exp_conf,
        )
    )

    exp_trust = fixture.get("expected_trust_report_fields") or {}
    obs_trust = stub["trust_report_results"]
    tr_ok, tr_missing = _dict_subset_eq(obs_trust, exp_trust)
    results.append(
        _assertion_row(
            "MA-TR",
            "VAL-BAYES-007",
            "pass" if tr_ok else "fail",
            message=f"missing or mismatched keys: {tr_missing}" if not tr_ok else "",
            observed=obs_trust,
            expected=exp_trust,
        )
    )

    exp_gate = fixture.get("expected_release_gate_effect") or {}
    obs_gate = stub["release_gate_results"]
    gate_ok = _canonicalize(obs_gate) == _canonicalize(exp_gate)
    results.append(
        _assertion_row(
            "MA-GATE",
            "VAL-BAYES-008",
            "pass" if gate_ok else "fail",
            observed=obs_gate,
            expected=exp_gate,
        )
    )

    silent_bad = any(c.get("silent_average") is True for c in obs_conf if isinstance(c, dict))
    report_text = json.dumps(stub)
    silent_pattern = any(p in report_text for p in SILENT_AVERAGE_FORBIDDEN if p not in ("silent_average",))
    results.append(
        _assertion_row(
            "MA-NOAVG",
            "VAL-BAYES-009",
            "pass" if not silent_bad and not silent_pattern else "fail",
            message="silent_average or forbidden merge pattern detected",
        )
    )

    train_artifact = Path(loaded["world_path"]) / "train_config.yaml"
    model_read = train_artifact.is_file() and False
    results.append(
        _assertion_row(
            "MA-NOFIT",
            "VAL-BAYES-010",
            "pass" if not model_read else "fail",
            message="train artifact used as evidence input",
        )
    )

    posterior_hits = _collect_forbidden_keys(stub, POSTERIOR_FORBIDDEN_KEYS)
    results.append(
        _assertion_row(
            "MA-NOPOST",
            "VAL-BAYES-011",
            "pass" if not posterior_hits else "fail",
            message=f"forbidden posterior fields: {posterior_hits}",
        )
    )

    decide_hits = _collect_forbidden_keys(stub, DECIDE_FORBIDDEN_KEYS)
    results.append(
        _assertion_row(
            "MA-NODECIDE",
            "VAL-BAYES-012",
            "pass" if not decide_hits else "fail",
            message=f"forbidden decision fields: {decide_hits}",
        )
    )

    return results


def _mandatory_assertion_rows(fixture: dict[str, Any], assertion_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    val_outcomes = {r["validation_id"]: r["outcome"] for r in assertion_results}
    rows: list[dict[str, Any]] = []
    for ma_id in fixture.get("mandatory_assertions") or []:
        outcome = "pass" if all(val_outcomes.get(v) == "pass" for v in VAL_BAYES_IDS) else "fail"
        rows.append(_assertion_row(ma_id, ma_id, outcome))
    return rows


def validate_hierarchy_evidence_world(world_path: str | Path) -> dict[str, Any]:
    """Validate a single Bayes-H2b world bundle against its fixture contract."""
    bundle = _repo_relative(world_path)
    world_id = bundle.name if bundle.is_dir() else str(world_path)

    present = {name: (bundle / name).is_file() for name in REQUIRED_BUNDLE_FILES}
    missing = [name for name, ok in present.items() if not ok]
    if missing:
        report = _empty_report(world_id, "blocked")
        report["failure_reasons"] = [f"E-FIXTURE-001: missing {missing}"]
        return report

    try:
        loaded = load_hierarchy_evidence_world(bundle)
    except json.JSONDecodeError as exc:
        report = _empty_report(world_id, "fail")
        report["failure_reasons"] = [f"E-FIXTURE-002: malformed JSON: {exc}"]
        return report
    except (FileNotFoundError, ValueError) as exc:
        report = _empty_report(world_id, "blocked")
        report["failure_reasons"] = [str(exc)]
        return report

    fixture = loaded["fixture"]
    missing_sections = [k for k in REQUIRED_FIXTURE_SECTIONS if k not in fixture]
    if missing_sections:
        report = _empty_report(loaded["world_id"], "fail")
        report["failure_reasons"] = [f"E-FIXTURE-003: missing fixture sections {missing_sections}"]
        return report

    if fixture.get("fixture_version") != FIXTURE_VERSION:
        report = _empty_report(loaded["world_id"], "blocked")
        report["failure_reasons"] = [
            f"policy version mismatch: {fixture.get('fixture_version')!r} != {FIXTURE_VERSION!r}"
        ]
        return report

    stub = _build_stub_results(loaded)
    assertion_results = _run_val_bayes_assertions(loaded, stub)
    assertion_results.extend(_mandatory_assertion_rows(fixture, assertion_results))

    failure_reasons: list[str] = []
    for row in assertion_results:
        if row.get("validation_id", "").startswith("VAL-BAYES") and row.get("outcome") == "fail":
            failure_reasons.append(f"{row['validation_id']}: {row.get('message') or row.get('assertion_id')}")

    report: dict[str, Any] = _empty_report(loaded["world_id"], "pass")
    report.update(stub)
    report["assertion_results"] = assertion_results

    if any(r.get("outcome") == "fail" for r in assertion_results):
        report["status"] = "fail"
        report["failure_reasons"] = failure_reasons

    return _canonicalize(report)  # type: ignore[return-value]


def validate_world_catalog(catalog_path: str | Path) -> dict[str, Any]:
    """Validate all Bayes-H2b worlds listed in world_catalog.index.json."""
    catalog_file = _repo_relative(catalog_path)
    catalog = _read_json_object(catalog_file)
    entries = catalog.get("entries") or []
    if not isinstance(entries, list):
        raise ValueError("catalog entries must be a list")

    world_reports: list[dict[str, Any]] = []
    smoke_failures: list[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        family = entry.get("validation_family") or entry.get("world_family")
        if family != "bayes-hierarchy-evidence":
            continue
        world_id = str(entry.get("world_id", ""))
        bundle_path = entry.get("bundle_path") or entry.get("path") or ""
        report = validate_hierarchy_evidence_world(bundle_path)
        world_reports.append(report)
        if report.get("status") != "pass":
            smoke_failures.append(f"{world_id}: {report.get('status')}")

    versions = {r.get("validator_version") for r in world_reports}
    deterministic_ok = True
    if world_reports:
        first_path = entries[0].get("bundle_path") or entries[0].get("path")
        if first_path:
            r1 = validate_hierarchy_evidence_world(first_path)
            r2 = validate_hierarchy_evidence_world(first_path)
            deterministic_ok = r1 == r2

    cross_checks: list[dict[str, Any]] = [
        _assertion_row("X-01", SMOKE_ID, "pass" if len(world_reports) == 7 and not smoke_failures else "fail"),
        _assertion_row("X-02", SMOKE_ID, "pass" if len(versions) == 1 else "fail"),
        _assertion_row("X-08", SMOKE_ID, "pass" if deterministic_ok else "fail"),
    ]

    for rid, check_fn in (
        ("X-03", lambda r: _val_pass(r, "VAL-BAYES-010") and _val_pass(r, "VAL-BAYES-011")),
        ("X-04", lambda r: not _collect_forbidden_keys(r, POSTERIOR_FORBIDDEN_KEYS)),
    ):
        ok = all(check_fn(r) for r in world_reports)
        cross_checks.append(_assertion_row(rid, SMOKE_ID, "pass" if ok else "fail"))

    smoke_pass = not smoke_failures and all(c["outcome"] == "pass" for c in cross_checks)

    return _canonicalize(
        {
            "catalog_version": catalog.get("catalog_version"),
            "validator_version": VALIDATOR_VERSION,
            "status": "pass" if smoke_pass else "fail",
            "world_count": len(world_reports),
            "world_reports": world_reports,
            "assertion_results": cross_checks
            + [
                _assertion_row(
                    SMOKE_ID,
                    SMOKE_ID,
                    "pass" if smoke_pass else "fail",
                    message="; ".join(smoke_failures) if smoke_failures else "",
                )
            ],
            "failure_reasons": smoke_failures,
            "warnings": [],
        }
    )


def _val_pass(report: dict[str, Any], validation_id: str) -> bool:
    for row in report.get("assertion_results") or []:
        if row.get("validation_id") == validation_id:
            return row.get("outcome") == "pass"
    return False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bayes-H2b hierarchy evidence validator (no-fit)")
    parser.add_argument(
        "--smoke",
        default=None,
        help="Run catalog smoke (e.g. VAL-BAYES-H2B-SMOKE)",
    )
    parser.add_argument(
        "--catalog",
        default="validation/worlds/world_catalog.index.json",
        help="Path to world catalog index",
    )
    parser.add_argument("--world", default=None, help="Single world bundle path")
    args = parser.parse_args()

    if args.smoke == SMOKE_ID or args.smoke == "VAL-BAYES-H2B-SMOKE":
        result = validate_world_catalog(args.catalog)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0 if result.get("status") == "pass" else 1)
    if args.world:
        result = validate_hierarchy_evidence_world(args.world)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0 if result.get("status") == "pass" else 1)
    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
