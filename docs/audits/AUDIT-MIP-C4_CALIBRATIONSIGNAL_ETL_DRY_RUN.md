# AUDIT-MIP-C4 — CalibrationSignal ETL Dry-Run Gate

**Audit ID:** AUDIT-MIP-C4  
**Date:** 2026-06-01  
**Scope:** Offline ETL dry-run: GeoX/CLS export → versioned C2 signals JSON → `mmm train` consumption proof  
**Prerequisites:** MIP-C3 @ `773192b`, MIP-C2 @ `dd3b36d`  
**Verdict:** **Pass (dry-run)** — repeatable artifact flow; train ingests context only; no live API or scheduling

---

## 1. Purpose

Prove a **repeatable dry-run ETL** can:

1. Read fixture/export JSON (GeoX + CLS)  
2. Write a governed **C2-compatible signals artifact** with lineage and disposition counts  
3. Be consumed by **`mmm train --calibration-signals-path`** without decision-path changes  

This is **not** production scheduling or live API integration.

---

## 2. ETL module

| Function | Role |
|----------|------|
| `load_geox_cls_export` | Load export; unwrap `export_bundle` fixture wrapper |
| `adapt_export_to_signals` | MIP-C3 `adapt_mixed_batch_export` |
| `build_etl_lineage` | Input hash, counts, boundary flags |
| `validate_signal_artifact` | C2 + production_flags checks |
| `write_signal_artifact` | Validated JSON write |
| `run_dry_run_etl` | End-to-end dry-run |

**Module:** `mmm/diagnostics/calibration_signal_etl.py`

---

## 3. CLI (dry-run)

```bash
poetry run python -m mmm.diagnostics.calibration_signal_etl \
  --input tests/fixtures/mip_calibration_signal_adapters/mixed_batch.json \
  --output docs/05_validation/archives/MIP_C4_DRY_RUN_CALIBRATION_SIGNALS_20260601.json
```

---

## 4. Artifact shape

| Field | Required |
|-------|----------|
| `artifact_type` | `calibration_signal_etl_dry_run` |
| `signals` | C2-ingestible CalibrationSignal[] |
| `etl_lineage` | `input_source`, `source_systems`, counts, boundary flags |
| `production_flags` | `approved_for_prod=false`, `decisioning_allowed=false` |

**Archive:** [MIP_C4_DRY_RUN_CALIBRATION_SIGNALS_20260601.json](../05_validation/archives/MIP_C4_DRY_RUN_CALIBRATION_SIGNALS_20260601.json)

---

## 5. Train consumption proof

```bash
poetry run mmm train examples/minimal_train.yaml \
  --calibration-signals-path docs/05_validation/archives/MIP_C4_DRY_RUN_CALIBRATION_SIGNALS_20260601.json
```

**Outcome:** `evidence_attachment_lineage.attempted=true`, `calibration_evidence_context` present, CLI calibration headline shown.

**Archive:** [MIP_C4_TRAIN_WITH_DRY_RUN_SIGNALS_20260601.json](../05_validation/archives/MIP_C4_TRAIN_WITH_DRY_RUN_SIGNALS_20260601.json)

**CLI fix (MIP-C4):** `mmm train` now uses `MMMTrainer(cfg)` so `--calibration-signals-path` is not dropped when reloading YAML.

---

## 6. Lineage and counts

Dry-run records:

- `records_seen`, `signals_written`  
- `blocked_count`, `inconclusive_count`, `stale_count`  
- `adapter_errors` (if any)  
- `input_content_sha256`  
- `context_only`, `optimizer_unchanged`, `decision_surface_unchanged`, `recommendations_unchanged`

---

## 7. Production boundaries

| Boundary | Status |
|----------|--------|
| Live GeoX/CLS API | Not called |
| Production scheduler | Not deployed |
| Ridge refit from signals | Not performed |
| Optimizer / DecisionSurface / recommendations | Unchanged |
| TrustReport | Not bypassed |
| Bayes-H5 | Research-only |

---

## 8. Verification

| Check | CI / artifact |
|-------|----------------|
| ETL writes valid artifact | `test_run_dry_run_etl_writes_artifact` |
| Disposition counts | `test_disposition_counts_in_lineage` |
| C2 ingest | `test_output_ingests_through_c2_path` |
| Train boundary | `test_train_boundary_consumes_dry_run_artifact` |
| CLI | `test_cli_module_dry_run` |
| Archives | materialized under `docs/05_validation/archives/` |

---

## 9. What remains (after MIP-C5)

- MIP-C5 drop-zone batch wrapper — ✅ [AUDIT-MIP-C5](AUDIT-MIP-C5_CALIBRATIONSIGNAL_SCHEDULED_ETL_WRAPPER.md)  
- Production cron/K8s deployment (roadmap audit recommended first)  
- Live GeoX/CLS API pull  
- Client channel mapping tables  
- TrustReport auto-population from ETL run metadata  
- Production evidence approval gates  

---

## 10. Related

- [AUDIT-MIP-C3](AUDIT-MIP-C3_GEOX_CLS_SIGNAL_ADAPTER_GATE.md)  
- [AUDIT-MIP-C2](AUDIT-MIP-C2_CALIBRATIONSIGNAL_TRAIN_BOUNDARY_WIRING.md)  
- [geox_cls_to_calibration_signal_adapter_contract.md](../05_validation/geox_cls_to_calibration_signal_adapter_contract.md)
