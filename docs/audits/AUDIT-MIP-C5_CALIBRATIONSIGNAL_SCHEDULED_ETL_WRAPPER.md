# AUDIT-MIP-C5 — CalibrationSignal Scheduled ETL Wrapper Gate

**Audit ID:** AUDIT-MIP-C5  
**Date:** 2026-06-01  
**Scope:** Drop-zone / scheduled-file ETL wrapper over MIP-C4 (no live API, no prod scheduler)  
**Prerequisites:** MIP-C4 @ `1f874a3`, MIP-C3 @ `773192b`, MIP-C2 @ `dd3b36d`  
**Verdict:** **Pass (drop-zone)** — batch ETL, manifest, C2 artifacts, train consumption proof

---

## 1. Purpose

Wrap MIP-C4 single-file ETL in a **repeatable drop-zone job** that:

1. Scans a configured input directory for export JSON (pattern/glob)  
2. Writes one **C2-compatible signals artifact** per export  
3. Emits a **job manifest** + Markdown summary with lineage and disposition counts  
4. Remains **non-decisioning** (no live API, no production cron deployment)

---

## 2. What directory/pattern was scanned?

| Field | Reference run |
|-------|----------------|
| **input_dir** | `tests/fixtures/mip_calibration_signal_adapters/` |
| **pattern** | `mixed_batch.json` |
| **run_id** | `MIP_C5_DRY_RUN_20260601` |
| **output_dir** | `docs/05_validation/archives/mip_c5_etl_outputs/` |

**CLI:**

```bash
poetry run python -m mmm.diagnostics.calibration_signal_etl_job \
  --input-dir tests/fixtures/mip_calibration_signal_adapters/ \
  --pattern mixed_batch.json \
  --output-dir docs/05_validation/archives/mip_c5_etl_outputs/ \
  --run-id MIP_C5_DRY_RUN_20260601
```

---

## 3. Which files were processed?

| Input | Output | Status |
|-------|--------|--------|
| `mixed_batch.json` | `MIP_C5_DRY_RUN_20260601_mixed_batch_signals.json` | Success (2 signals) |

Manifest: [MIP_C5_DRY_RUN_20260601_manifest.json](../05_validation/archives/mip_c5_etl_outputs/MIP_C5_DRY_RUN_20260601_manifest.json)  
Summary: [MIP_C5_DRY_RUN_20260601_summary.md](../05_validation/archives/mip_c5_etl_outputs/MIP_C5_DRY_RUN_20260601_summary.md)

---

## 4. What outputs were written?

- Per-file C2 artifact (`artifact_type=calibration_signal_etl_dry_run`)  
- Job manifest (`artifact_type=calibration_signal_etl_job_manifest`)  
- Operator Markdown summary  

---

## 5. Was live API used?

**No.** `live_api_used=false`, `production_scheduler=false` on manifest.

---

## 6. Were outputs C2-compatible?

**Yes.** `validate_signal_artifact` passes; `parse_calibration_signals_payload` accepts `signals` array; MIP-C2 ingest attaches `calibration_evidence_context`.

---

## 7. Did train consume one output successfully?

```bash
poetry run mmm train examples/minimal_train.yaml \
  --calibration-signals-path docs/05_validation/archives/mip_c5_etl_outputs/MIP_C5_DRY_RUN_20260601_mixed_batch_signals.json
```

**Proof archive:** [MIP_C5_TRAIN_WITH_SCHEDULED_ETL_SIGNALS_20260601.json](../05_validation/archives/MIP_C5_TRAIN_WITH_SCHEDULED_ETL_SIGNALS_20260601.json)  
`calibration_evidence_context_present=true`

---

## 8. Were decision boundaries preserved?

| Boundary | Status |
|----------|--------|
| Ridge refit from signals | Not performed |
| Optimizer / DecisionSurface / recommendations | Unchanged |
| TrustReport | Not bypassed |
| Bayes-H5 | Research-only |
| Manifest `context_only` | true |

---

## 9. Module surface

| Function | Role |
|----------|------|
| `discover_export_files` | Pattern/glob scan in drop-zone |
| `run_etl_for_file` | MIP-C4 ETL per export |
| `write_job_manifest` | Versioned manifest JSON |
| `validate_job_outputs` | Manifest + artifact validation |
| `summarize_job_run` | Markdown summary |
| `run_scheduled_etl_job` | Batch orchestration (`continue_on_error` default true) |

**Module:** `mmm/diagnostics/calibration_signal_etl_job.py`

---

## 10. What remains before live production scheduling?

| Item | Status |
|------|--------|
| Drop-zone wrapper (MIP-C5) | ✅ |
| MIP-C4 single-file ETL | ✅ |
| MIP-C3 adapter | ✅ |
| MIP-C2 train ingest | ✅ |
| Production cron/K8s/Airflow deployment | **Not in scope** |
| Live GeoX/CLS API pull | **Blocked** |
| Roadmap audit before live scheduling | **Recommended** (operator bias) |
| Client channel mapping + TrustReport auto-fill | Governance / ETL owner |

---

## 11. Verification

| Check | CI |
|-------|-----|
| Discover / ignore patterns | `test_discover_*` |
| Per-file C2 output | `test_run_etl_for_file_*` |
| Manifest + summary | `test_scheduled_job_*` |
| Continue on error | `test_continue_on_error_*` |
| C2 ingest | `test_output_ingests_through_c2` |
| CLI | `test_cli_job_module` |

---

## 12. Related

- [AUDIT-MIP-C4](AUDIT-MIP-C4_CALIBRATIONSIGNAL_ETL_DRY_RUN.md)  
- [AUDIT-MIP-C3](AUDIT-MIP-C3_GEOX_CLS_SIGNAL_ADAPTER_GATE.md)  
- [geox_cls_to_calibration_signal_adapter_contract.md](../05_validation/geox_cls_to_calibration_signal_adapter_contract.md)
