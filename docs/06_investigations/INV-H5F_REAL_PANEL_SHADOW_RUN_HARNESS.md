# INV-H5F — Bayes-H5 Real-Panel Shadow-Run Execution Harness (Research Only)

**Investigation ID:** INV-H5F  
**Status:** **Harness complete** (research lane; production Bayes still blocked)  
**Date:** 2026-06-01  
**Prerequisites:** Bayes-H5e (`7d6e27c`) shadow-run protocol + schema  
**Protocol:** [INV-H5E](INV-H5E_REAL_PANEL_SHADOW_RUN_PROTOCOL.md) · [H5e schema](../05_validation/archives/BAYES_H5E_SHADOW_RUN_SCHEMA_20260601.json)  
**Code:** `mmm/research/bayes_h3_sandbox/h5_shadow_runner.py`  
**Dry-run artifact:** [BAYES_H5F_SHADOW_RUN_DRY_RUN_20260601.json](../05_validation/archives/BAYES_H5F_SHADOW_RUN_DRY_RUN_20260601.json)

---

## 1. Purpose

Implement a **research-only execution harness** that runs Bayes-H5 (`bayes_h5_sandbox_spec_v1`) on **immutable historical MMM panels** (or a synthetic fixture) and emits **H5e schema-compliant** shadow-run JSON artifacts.

This is **not** production integration: no prod TrustReport wiring, optimizer, DecisionSurface, recommendations, hard gates, or Ridge replacement.

---

## 2. What was implemented

| Component | Role |
|-----------|------|
| `h5_shadow_runner.py` | Validates lineage inputs, calls `run_sandbox_fit` only via sandbox entrypoint, maps H5d `trust_report_candidate_diagnostics`, writes JSON envelope + inner `shadow_run` |
| Fixture dry-run | `run_fixture_dry_run_shadow()` — deterministic toy panel; `artifact_type=dry_run_shadow_artifact` |
| CLI | `python -m mmm.research.bayes_h3_sandbox.h5_shadow_runner` with required `--panel-id`, `--dataset-snapshot-id`, `--transform-config` |
| Tests | `tests/research/test_bayes_h5_shadow_runner.py` |
| Dry-run archive | `BAYES_H5F_SHADOW_RUN_DRY_RUN_20260601.json` |

---

## 3. How to run

### Fixture dry-run (no private data)

```bash
poetry run python -m mmm.research.bayes_h3_sandbox.h5_shadow_runner \
  --fixture-dry-run \
  --fast-mcmc
```

Writes default path: `docs/05_validation/archives/BAYES_H5F_SHADOW_RUN_DRY_RUN_20260601.json`.

### Real historical panel (research authorization required)

```bash
poetry run python -m mmm.research.bayes_h3_sandbox.h5_shadow_runner \
  --panel-path /path/to/panel.csv \
  --panel-id <panel_id> \
  --dataset-snapshot-id <immutable_snapshot_id> \
  --transform-config /path/to/transform_config.json \
  --output-path docs/05_validation/archives/BAYES_H5F_SHADOW_RUN_<PANEL_ID>_20260601.json \
  --fast-mcmc
```

Use `artifact_type=real_panel_shadow_artifact` only when the panel is a documented historical client slice (not the synthetic fixture).

---

## 4. Required inputs

| Input | Required | Notes |
|-------|----------|-------|
| `dataset_snapshot_id` | Yes | Immutable snapshot reference |
| `panel_id` | Yes | Logical panel name |
| `transform_config` | Yes | Non-empty H5 registry object |
| `model_spec_version` | Yes (implicit) | Must be `bayes_h5_sandbox_spec_v1` |
| `enable_h5_sandbox` | Yes (implicit) | Must be `true` |
| `panel_path` or `panel_df` | Yes (except fixture dry-run) | CSV/Parquet |
| Production flags | Must stay **false** | Requesting `approved_for_prod=true` etc. fails closed |

---

## 5. Failure modes (fail closed)

- Missing `dataset_snapshot_id`, `panel_id`, or `transform_config`
- `enable_h5_sandbox=false` or wrong `model_spec_version`
- Requested production flags `true`
- H5e validation failure on inner `shadow_run`
- Forbidden fields on envelope (`decision_surface`, optimizer/recommendation keys)
- Panel path missing or unsupported format

---

## 6. Dry-run artifact summary

| Field | Value |
|-------|--------|
| `artifact_type` | `dry_run_shadow_artifact` |
| `dataset_snapshot_id` (inner) | `synthetic_fixture_only` |
| `panel_id` | `synthetic_h5_shadow_fixture` |
| `model_spec_version` | `bayes_h5_sandbox_spec_v1` |
| Production flags | All **false** |
| Evidence claim | **Does not** constitute real-panel evidence |

---

## 7. First real-panel run (H5g)

See [INV-H5G_FIRST_REAL_PANEL_SHADOW_RUN.md](INV-H5G_FIRST_REAL_PANEL_SHADOW_RUN.md) and [H5G_FIRST_REAL_PANEL_SHADOW_RUN_MANIFEST.md](H5G_FIRST_REAL_PANEL_SHADOW_RUN_MANIFEST.md).

---

## 8. What counts as real-panel evidence later

A shadow run counts as **real-panel evidence** only when **all** of the following hold:

1. `artifact_type=real_panel_shadow_artifact`
2. Documented immutable `dataset_snapshot_id` tied to a client historical panel (not `synthetic_fixture_only`)
3. Explicit `transform_config` aligned to the documented client spec (or intentional mismatch labeled)
4. Successful H5e validation + populated `trust_report_candidate_diagnostics` from an executed fit
5. Separate research authorization / ops checklist (outside this harness deliverable)

Dry-run and schema-only records **do not** satisfy (2) or (5).

---

## 9. Production boundary

| Allowed | Blocked |
|---------|---------|
| Research sandbox `run_sandbox_fit` | Production pipelines |
| H5d candidate diagnostics in artifact | Production TrustReport wiring |
| Ridge comparison metadata (diagnostic) | Optimizer / DecisionSurface / recommendations |
| Archives under `docs/05_validation/archives/` | `approved_for_prod`, `prod_decisioning_allowed`, hard gates |

**Production Bayes remains blocked.** Ridge production path unchanged.
