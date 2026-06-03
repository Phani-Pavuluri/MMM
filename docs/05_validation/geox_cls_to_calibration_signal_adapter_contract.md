# GeoX / CLS → CalibrationSignal adapter contract (MIP-C3)

**Status:** Accepted (adapter contract — no live APIs)  
**Date:** 2026-06-01  
**Audit:** [AUDIT-MIP-C3](../audits/AUDIT-MIP-C3_GEOX_CLS_SIGNAL_ADAPTER_GATE.md)  
**Implementation:** `mmm/diagnostics/calibration_signal_adapters.py`  
**Downstream:** [MIP-C2 train-boundary ingestion](calibration_signal_mmm_diagnostic_attachment_contract.md) — same JSON shape as `signals` file

## Purpose

Define how **offline GeoX and CLS exports** convert to **CalibrationSignal** JSON for `mmm train --calibration-signals-path` — without live API calls, Ridge refit, or decision-path changes.

---

## Architecture

```text
GeoX export CSV/JSON  ──┐
                        ├── adapter (MIP-C3) ──► CalibrationSignal[] ──► C2 ingest ──► Ridge diagnostics context
CLS readout JSON     ──┘
```

**Not in scope:** live GeoX/CLS API clients, production evidence approval, Bayes-H5 promotion.

---

## GeoX required input fields

| Export field (aliases accepted) | Required | Maps to CalibrationSignal |
|---------------------------------|----------|---------------------------|
| `geox_experiment_id` / `experiment_id` | Yes | `experiment_id`, `signal_id` prefix |
| `channel` / `media_channel` | Yes | `channel` |
| `incremental_lift` / `lift` / `effect_estimate` | Yes | `effect_estimate` |
| `incremental_lift_se` / `lift_se` / `standard_error` | For `eligible` | `standard_error` |
| `dma_ids` / `geo_ids` | Recommended | `geo_scope.kind=dma` |
| `window_start`, `window_end` | Recommended | `time_window` |
| `estimand_type` / `estimand_id` | Recommended | `estimand_id` (alias table) |
| `geox_export_version` / `export_version` | Recommended | `measurement_instrument_id` |
| `evidence_as_of` / `export_timestamp` | Recommended | `freshness_status` |

Optional flags: `estimand_mismatch_flag` → `adapter_notes` (TrustReport boundary preserved).

---

## CLS required input fields

| Export field (aliases accepted) | Required | Maps to CalibrationSignal |
|---------------------------------|----------|---------------------------|
| `cls_study_id` / `study_id` | Yes | `study_id`, `signal_id` prefix |
| `channel` / `media_channel` | Yes | `channel` |
| `point_estimate` / `effect_estimate` | Yes | `effect_estimate` |
| `standard_error` | For `eligible` | `standard_error` |
| `readout_status` / `cls_readout_status` | Recommended | `eligibility_status` |
| `cls_readout_version` / `readout_id` | Recommended | `measurement_instrument_id` |
| `as_of_date` / `readout_date` | Recommended | `freshness_status` |
| `estimand_id` / `kpi_type` | Recommended | `estimand_id` |
| `geo_scope` / `geo_scope_kind` | Optional | `geo_scope` |

---

## Output CalibrationSignal schema

Must satisfy [MIP-C1/C2 attachment contract](calibration_signal_mmm_diagnostic_attachment_contract.md) ingest rules:

- `signal_id`, `source_system`, `source_modality`, `channel`
- `effect_estimate`, `freshness_status`, `eligibility_status`
- Plus `adapter_metadata` block (MIP-C3):

| Field | Description |
|-------|-------------|
| `adapter_version` | `mip_geox_cls_calibration_signal_adapter_v1` |
| `source_lineage` | Original export ids (experiment, readout, export version) |
| `adapter_notes` | Uncertainty / estimand / mismatch notes |
| `trust_report_boundary` | Context-only; TrustReport governs promotion |

---

## Mapping rules

### Estimand mapping

| Export value | Platform `estimand_id` |
|--------------|------------------------|
| `incremental_sales`, `incremental_roi`, `sales_lift` | `incremental_sales` |
| `brand_lift`, `awareness_lift` | `brand_lift` |
| Unknown string | Passed through + `estimand_unmapped` note |

Mismatch with MMM run estimand is evaluated at **MIP-C1 attach** time (`scope_mismatch`), not overridden by adapter.

### Measurement instrument

| Source | `measurement_instrument_id` |
|--------|----------------------------|
| GeoX | `geox_export_version` or `geox_v3` default |
| CLS | `cls_readout_version` or `cls_readout_v1` default |

### Channel mapping

Export `channel` / `media_channel` must match MMM `data.channel_columns` keys — **no silent rename** in adapter v1; ETL owner documents mapping table externally.

### Geo / time window

- GeoX: `dma_ids` → `{kind: dma, ids: [...]}`; else `national`
- CLS: `geo_scope` object or `geo_scope_kind`
- Windows: `window_start` + `window_end` → `{start, end}` ISO strings

### Uncertainty handling

| Condition | Adapter `eligibility_status` | C1/C2 behavior |
|-----------|------------------------------|----------------|
| SE or interval present | `eligible` (unless readout blocks) | May attach as diagnostic context |
| Missing SE | `blocked` | `trust_report_only` at attach |
| Interval only | `eligible` if interval valid | SE optional if interval present |

### Freshness handling

| Condition | `freshness_status` |
|-----------|-------------------|
| `evidence_as_of` older than 180 days | `stale` |
| `stale_flag=true` (CLS) | `stale` |
| Recent as-of | `fresh` |
| Missing as-of | `unknown` |

### Eligibility handling

| Source signal | `eligibility_status` |
|---------------|---------------------|
| CLS `readout_status=inconclusive` | `inconclusive` |
| Missing uncertainty | `blocked` |
| GeoX default | `eligible` when SE present |

---

## Failure modes (fail-closed)

| Failure | Behavior |
|---------|----------|
| Invalid batch row type | Skip row; `attachment_errors` at C2 if ingested |
| Missing `signal_id` after adapt | `validate_adapter_output` fails |
| Missing `channel` | Validation fails |
| Forbidden decision fields on signal | Validation fails |
| Malformed JSON export | C2 load errors; no attach |
| Live API not implemented | **N/A** — adapter is export-only |

---

## API (adapter module)

```python
geox_record_to_calibration_signal(record) -> dict
cls_record_to_calibration_signal(record) -> dict
normalize_calibration_signal_batch(records, source_system) -> (signals, errors)
validate_adapter_output(signal) -> (ok, errors)
adapt_mixed_batch_export(data) -> (signals, errors, lineage)
```

### C2 file shape (after adaptation)

```json
{
  "source_ref": "geox-cls-etl-run-id",
  "signals": [ { "...": "CalibrationSignal" } ]
}
```

---

## Tests and fixtures

| Fixtures | `tests/fixtures/mip_calibration_signal_adapters/` |
| Tests | `tests/mip/test_calibration_signal_adapters.py` |
| Archive | `docs/05_validation/archives/MIP_C3_ADAPTED_GEOX_CLS_SIGNALS_20260601.json` |

---

## Related

- [AUDIT-MIP-C2](../audits/AUDIT-MIP-C2_CALIBRATIONSIGNAL_TRAIN_BOUNDARY_WIRING.md)
- [Bayes-H2 CalibrationSignal ADR](bayes_h2_calibration_signal_mapping_adr.md)
