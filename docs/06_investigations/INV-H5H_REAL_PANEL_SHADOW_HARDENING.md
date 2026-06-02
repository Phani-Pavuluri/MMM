# INV-H5H — Bayes-H5 Real-Panel Shadow-Run Hardening

**Investigation ID:** INV-H5H  
**Status:** **Complete (research lane)**  
**Date:** 2026-06-01  
**Prerequisites:** Bayes-H5g first real-panel shadow (`4115c5b`)  
**Artifact:** [BAYES_H5H_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json](../05_validation/archives/BAYES_H5H_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json)  
**Code:** `h5_shadow_runner.py`, `h5_trust_diagnostics.py`, `h5_transforms.py`, `model.py`

---

## 1. What did H5g reveal?

| Issue | H5g observation |
|-------|-----------------|
| Panel schema | Harness assumed `geo_id`/`week`/`y`; real panels use `week_start_date`/`revenue` |
| Transform warnings | `h5:transform_mismatch:adstock` fired on real panel because generative kind was `shadow_panel` stub |
| Convergence | Fast MCMC: `rhat_max=2.09`, 9 divergences → not usable evidence |
| Ridge | Diagnostic contrast possible (WMAPE ~13.6% in-sample) |
| GeoX/CLS | Not available on sample panel |

H5g proved the **harness works**; it did **not** prove the model is ready.

---

## 2. What diagnostic semantics were corrected?

### Real panels (`panel_context=real_panel`)

- Generative transform reported as **`unknown`** (no synthetic truth).
- `transform_mismatch_detected=false` unless `transform_mismatch_mode=intentional_mismatch`.
- Warning codes use **assumption** taxonomy:
  - `h5:transform_assumption:identity` (per declared channel transforms)
  - `h5:transform_unknown:real_panel`
- **`h5:transform_mismatch:*` removed** from default real-panel runs.

### Synthetic worlds / dry-run fixture

- Unchanged: generative truth known; `h5:transform_mismatch:*` when mismatch detected.
- Recovery worlds still use `h5_validation_worlds` + pilot mapping.

### Convergence (report-only)

| Class | Rule |
|-------|------|
| `converged_diagnostic_only` | `rhat_max <= 1.05` and `divergence_count == 0` |
| `weak_convergence` | `rhat_max <= 1.10` and `divergence_count <= 5` |
| `failed_convergence` | otherwise |

Warnings: `h5:convergence:failed` / `h5:convergence:weak`, plus `h5:evidence:blocked` when not converged.

`evidence_promotion_allowed` is **false** unless `converged_diagnostic_only`.

### Artifact fields

- `real_panel_diagnostics` block with `sampler_diagnostics`, `convergence_status`, `evidence_promotion_allowed`.
- CLI `--extended-mcmc` → 600/600/4 chains, `target_accept=0.95` (H5c profile).

---

## 3. Did extended MCMC improve convergence?

**No** on `examples/sample_panel.csv` (extended rerun in H5h):

| Profile | rhat_max | divergences | convergence_status |
|---------|----------|-------------|-------------------|
| Fast (H5g) | 2.09 | 9 | `failed_convergence` |
| Extended (H5h) | 1.69 | 74 | `failed_convergence` |

Extended sampling reduced R-hat somewhat but **increased** divergences. The sample panel remains **not usable** as shadow evidence until sampler/panel geometry issues are addressed (reparameterization, scaling, or richer geo structure — out of scope for H5h).

---

## 4. Is the sample panel usable as evidence?

**No.** H5h artifact records:

- `convergence_status`: `failed_convergence`
- `evidence_promotion_allowed`: `false`
- `h5:evidence:blocked` in warning codes

It remains a **harness and diagnostics regression panel**, not decision-grade or multi-panel campaign evidence.

---

## 5. What is required before running more real panels?

1. **Convergence** — achieve `converged_diagnostic_only` or documented `weak_convergence` on a pilot panel before batching.
2. **Transform policy** — document declared assumptions vs prod Ridge FE; optional intentional_mismatch probes only when explicit.
3. **Sampler** — investigate divergences on small-geo panels (3 geos, identity transforms).
4. **GeoX/CLS stubs** — attach experiment evidence when available.
5. **Automated Ridge contrast** — optional harness hook (still diagnostic-only).
6. **Authorization** — per-panel manifest + immutable `dataset_snapshot_id` (H5g pattern).

**Do not** run broad real-panel batches until the above are met.

---

## 6. Production boundary

| Item | Status |
|------|--------|
| Production Bayes | **Blocked** |
| Prod TrustReport / optimizer / DecisionSurface | **Not wired** |
| Shadow output | **Report-only**; `h5:evidence:blocked` when convergence fails |
| Ridge production path | **Unchanged** |

---

## Re-run (extended profile)

```bash
poetry run python -m mmm.research.bayes_h3_sandbox.h5_shadow_runner \
  --panel-path examples/sample_panel.csv \
  --panel-id examples_mmm_sample_panel_v1 \
  --dataset-snapshot-id mmm-examples-sample-panel-frozen-2022 \
  --transform-config docs/06_investigations/h5g_sample_panel_transform_config.json \
  --output-path docs/05_validation/archives/BAYES_H5H_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json \
  --extended-mcmc
```
