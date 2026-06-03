# AUDIT-MIP-C6 — MIP Integration Readiness Checkpoint

**Audit ID:** AUDIT-MIP-C6  
**Date:** 2026-05-22  
**Scope:** Roadmap audit — decide whether to proceed to C6 production scheduler governance, pause MIP integration, or return to GeoX estimator/inference OC work  
**Prerequisites:** MIP-C5 @ `c7b3ab6` (file-based GeoX/CLS → CalibrationSignal → MMM diagnostic bridge complete through drop-zone ETL)  
**Verdict:** **`continue_with_pause_before_live_scheduler`** — bridge is integration-ready for governed file/drop-zone use; **do not** deploy production cron, live APIs, or decision-path promotion until C6 prerequisites and upstream GeoX OC are satisfied

---

## 1. Purpose

This checkpoint answers three questions after MIP-C1–C5:

1. Is the **file-based diagnostic bridge** complete and safe for continued governed use?  
2. What is **missing** before C6 production scheduler governance and live GeoX/CLS pull?  
3. Which **next lane** should the program take: C6 governance, GeoX OC, H11b diagnostics, or Bayes-H5 transform alignment?

**Audit-only.** No live APIs, production cron, Ridge refit changes, optimizer/DecisionSurface changes, recommendation emission, or Bayes-H5 production promotion.

**Core principle (unchanged):** External signals inform **interpretation**; they do **not** change decision paths.

---

## 2. Completed bridge summary (C1–C5 + H7–H11)

### End-to-end flow

```text
GeoX/CLS export JSON (file)
  → MIP-C3 adapter (calibration_signal_adapters.py)
  → MIP-C4 single-file ETL (calibration_signal_etl.py)
  → MIP-C5 drop-zone batch job (calibration_signal_etl_job.py)
  → MIP-C2 train ingest (calibration_signal_ingestion.py + extension_runner)
  → MIP-C1 context attach (calibration_signal_attachment.py → calibration_evidence_context)
  → H7–H9 Ridge diagnostic stack + H8 operator summary (ridge_diagnostic_summary.py)
  → H10 E2E audit + H11 real-bundle hardening (ridge_real_bundle_hardening.py)
```

### Milestone table

| Milestone | Commit (ref) | Deliverable | Decision-path impact |
|-----------|----------------|-------------|----------------------|
| **MIP-C1** | `df54dd1` | Attachment contract; `calibration_evidence_context`; alignment/conflict/stale policy; fixture tests | None — context only |
| **MIP-C2** | `dd3b36d` | Train/extension ingest; YAML `calibration_signals_path`; CLI `--calibration-signals-path`; `evidence_attachment_lineage` | None — fit unchanged |
| **MIP-C3** | `773192b` | GeoX/CLS export adapters; adapter contract; export-only, no live API | None |
| **MIP-C4** | `1f874a3` | Dry-run ETL CLI; `calibration_signal_etl_dry_run` artifact; train consumption proof | None — `context_only`, `approved_for_prod=false` |
| **MIP-C5** | `c7b3ab6` | Drop-zone batch wrapper; job manifest; `continue_on_error`; reference archives under `mip_c5_etl_outputs/` | None — no prod scheduler |
| **H7** | (H7 impl) | `ridge_diagnostics.py`; production diagnostic contract | Metadata only |
| **H8** | (H8 impl) | Operator JSON/MD/CLI summary surfacing | UX only |
| **H9** | (H9 impl) | Severity/eligibility policy | Policy only |
| **H10** | `3ed159d` | E2E audit gate on H6 synthetic worlds | Verification |
| **H11** | `9d20c72` | Real-bundle hardening on benchmark geo panel; lineage on illustrative bundle | Diagnostics only |

### Key artifacts and contracts

| Layer | Doc / module |
|-------|----------------|
| C1 attachment | [calibration_signal_mmm_diagnostic_attachment_contract.md](../05_validation/calibration_signal_mmm_diagnostic_attachment_contract.md); `calibration_signal_attachment.py` |
| C2 ingest | [AUDIT-MIP-C2](AUDIT-MIP-C2_CALIBRATIONSIGNAL_TRAIN_BOUNDARY_WIRING.md); `calibration_signal_ingestion.py` |
| C3 adapter | [geox_cls_to_calibration_signal_adapter_contract.md](../05_validation/geox_cls_to_calibration_signal_adapter_contract.md); `calibration_signal_adapters.py` |
| C4 ETL | [AUDIT-MIP-C4](AUDIT-MIP-C4_CALIBRATIONSIGNAL_ETL_DRY_RUN.md); `calibration_signal_etl.py` |
| C5 batch | [AUDIT-MIP-C5](AUDIT-MIP-C5_CALIBRATIONSIGNAL_SCHEDULED_ETL_WRAPPER.md); `calibration_signal_etl_job.py` |
| Ridge stack | [ridge_production_diagnostics_contract.md](../05_validation/ridge_production_diagnostics_contract.md); [AUDIT-H10](AUDIT-H10_RIDGE_DIAGNOSTIC_E2E_GATE.md); [INV-H11](../06_investigations/INV-H11_REAL_BUNDLE_RIDGE_DIAGNOSTIC_HARDENING.md) |

### Train path note (MIP-C4 fix)

`mmm train` uses **`MMMTrainer(cfg)`** after config load so `--calibration-signals-path` is not dropped when YAML is re-read — required for C4/C5 consumption proofs.

---

## 3. Readiness dimensions

| Dimension | Status | Evidence / notes |
|-----------|--------|------------------|
| **Schema readiness** | **Ready (file bridge)** | C1 contract + C3 adapter contract + `validate_signal_artifact` / `parse_calibration_signals_payload`; ETL artifacts versioned with `artifact_type` |
| **Lineage readiness** | **Ready (train + ETL)** | MIP-C2 `evidence_attachment_lineage`; C4/C5 manifests record source paths, disposition, `context_only`; H11 embeds diagnostic lineage on real bundle |
| **Operator visibility** | **Ready (diagnostics)** | H8 summary MD/CLI; C5 job Markdown summary; calibration context blocks in train bundle; severity policy H9 |
| **Failure handling** | **Ready (batch/file)** | C5 `continue_on_error` default; per-file failure in manifest; ingest fail-closed on invalid artifact at train boundary |
| **Malformed input handling** | **Ready (fixtures + tests)** | Adapter/ETL tests for invalid JSON, wrapper vs export shape; job records failed files without aborting batch (when configured) |
| **Conflict / stale / mismatch** | **Ready (attachment policy)** | C1 `attach_calibration_evidence_context` evaluates alignment, stale, estimand/scope mismatch; forbidden claims additive; TrustReport-only paths documented |
| **TrustReport boundary** | **Preserved** | Signals do not bypass TrustReport; directional conflict and inconclusive CLS → human review; no auto-promotion from diagnostics |
| **Production scheduler readiness** | **Not ready** | C5 is CLI/drop-zone proof only; no ownership, runtime env, secrets, retention, monitoring, or idempotency SLO |
| **Live API readiness** | **Blocked** | No GeoX/CLS pull in repo; `live_api_used=false` on manifests; explicit signoff required before any live integration |
| **Decision-path safety** | **Pass** | No Ridge refit from signals; no optimizer/DecisionSurface/recommendation fields; Bayes-H5 research-only; `production_flags.approved_for_prod=false` on ETL/job artifacts |

---

## 4. Verdict

| Field | Value |
|-------|-------|
| **Verdict code** | `continue_with_pause_before_live_scheduler` |
| **Bridge (C1–C5 + H7–H11)** | **Complete** for governed file-based diagnostic context |
| **C6 production scheduler** | **Deferred** until prerequisites below and explicit operational need |
| **Live APIs** | **Blocked** until C6 governance + TrustReport/live-API signoff |
| **MIP integration pause?** | **Partial pause** — do not advance live scheduling/APIs; file bridge may continue in dry-run/drop-zone mode |

**Interpretation:** Proceed with the **integration program** in the sense that the diagnostic bridge is validated and maintainable, but **pause** before production cron/K8s/Airflow deployment and before live GeoX/CLS pull. Upstream export quality and estimator OC should lead unless there is an immediate operational mandate for scheduled ingest.

---

## 5. Remaining C6 prerequisites (production scheduler governance)

| Prerequisite | Why it blocks C6 |
|--------------|------------------|
| **Production ownership** | Named on-call + ETL owner for drop-zone, train ingest failures, and manifest review |
| **Scheduler runtime environment** | Approved host (cron/K8s/Airflow/etc.), image/poetry pin, job identity, run windows |
| **Secrets / access policy** | Drop-zone read, output write, train artifact paths — least privilege, rotation |
| **Drop-zone retention policy** | Input export TTL, processed archive, failed quarantine, PII/redaction rules |
| **Monitoring / alerting** | Job success rate, zero-signal runs, adapter validation failures, train attach failures |
| **Retry / idempotency** | Deterministic `run_id`, safe re-run on same export, dedup by export fingerprint |
| **Data freshness SLA** | Max age for signals at train time; align with C1 `freshness_status=stale` policy |
| **Audit log retention** | Manifest + summary retention matching compliance; link to train bundle lineage |
| **TrustReport review path** | Escalation when `directional_conflict` or TrustReport-only flags appear in production manifests |
| **Signoff before live APIs** | Separate gate: API credentials, rate limits, export schema drift monitoring — not in MMM repo today |

---

## 6. Recommended next lane

| Option | Lane | When to choose |
|--------|------|----------------|
| **A** | C6 production scheduler governance | Immediate operational need for scheduled drop-zone → train ingest in a governed prod environment |
| **B** | **GeoX estimator / inference OC work** (unified MIP / GeoX program) | **Recommended default** — improve export quality, estimand clarity, and inference ops **before** automating pull/schedule |
| **C** | H11b real-bundle diagnostic coverage | Need triangulation panel (`MMM-TRIANGULATION-GEO-PANEL-V1`) + sparse radio + calibration stub on Ridge path |
| **D** | Bayes-H5 transform alignment | Research-only posterior/transform consistency; **not** production promotion |

**Recommendation:** **Option B** unless there is an immediate operational need for Option A.

**Rationale:**

- MMM-side bridge is **proven through C5** with decision boundaries intact.  
- Weak links are **upstream** (export shape drift, estimator OC, channel mapping) and **governance** (scheduler, secrets, TrustReport ops) — not Ridge attachment mechanics.  
- GeoX estimator/inference OC lives primarily in the **unified MIP / GeoX** program ([platform_roadmap.md](../05_validation/platform_roadmap.md); [ACCIDENTAL_GEOX_TRACK_D_PASTE_QUARANTINE.md](../ACCIDENTAL_GEOX_TRACK_D_PASTE_QUARANTINE.md)).  
- H11b and Bayes-H5 remain valuable but secondary to export/estimator trust for integration.

---

## 7. Explicit non-goals (this checkpoint)

- Live GeoX/CLS APIs in MMM  
- Production cron/K8s/Airflow deployment  
- Ridge coefficient override or refit from CalibrationSignal  
- Optimizer, DecisionSurface, or budget **recommendations** from signals  
- Bayes-H5 production promotion  
- TrustReport auto-fill from ETL without human review path  

---

## 8. Verification references

| Check | Location |
|-------|----------|
| C1 attachment contract | `tests/mip/test_calibration_signal_mmm_attachment_contract.py` |
| C2 train boundary | `tests/mip/test_calibration_signal_train_boundary_ingestion.py` |
| C3 adapters | `tests/mip/test_calibration_signal_adapters.py` |
| C4 ETL dry-run | `tests/mip/test_calibration_signal_etl_dry_run.py` |
| C5 scheduled job | `tests/mip/test_calibration_signal_etl_job.py` |
| H10 E2E | `tests/diagnostics/test_ridge_diagnostic_e2e_audit.py` |
| H11 real bundle | `tests/diagnostics/test_ridge_diagnostics_real_bundle_compat.py` |

---

## 9. Related audits

- [AUDIT-MIP-C1](AUDIT-MIP-C1_CALIBRATIONSIGNAL_MMM_INTEGRATION_GATE.md) through [AUDIT-MIP-C5](AUDIT-MIP-C5_CALIBRATIONSIGNAL_SCHEDULED_ETL_WRAPPER.md)  
- [AUDIT-H10](AUDIT-H10_RIDGE_DIAGNOSTIC_E2E_GATE.md)  
- [INV-H11](../06_investigations/INV-H11_REAL_BUNDLE_RIDGE_DIAGNOSTIC_HARDENING.md)  
- [ROADMAP_ALIGNMENT_REGISTRY](../ROADMAP_ALIGNMENT_REGISTRY.md) — MIP-C6 row  
