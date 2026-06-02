# INV-H5O — Bayes-H5 Second Real-Panel Shadow Run

**Investigation ID:** INV-H5O  
**Status:** complete (research lane)  
**Date:** 2026-06-01  
**Prerequisites:** H5n @ `bf98798`  
**Manifest:** [H5O_SECOND_REAL_PANEL_SHADOW_RUN_MANIFEST.md](H5O_SECOND_REAL_PANEL_SHADOW_RUN_MANIFEST.md)

## Which panel was selected and why?

**Panel:** `examples/benchmark_geo_panel_v1.csv` (`examples_mmm_benchmark_geo_panel_v1`)

Selected as the **only authorized second panel** for H5o: distinct from the H5g–H5m `sample_panel.csv` pilot, frozen in-repo, public, sufficient rows for extended MCMC (208 rows, 4 geos), and no live business decision attached. It is a deterministic benchmark CSV (not an H4 recovery world, not client-confidential).

## What did the recommender diagnose?

| Diagnostic | Result |
|------------|--------|
| max \|ρ\| | 0.061 |
| High collinearity | false |
| Collinear groups | none |
| Sparsity | all channels with variation |
| Prior sample-panel experiments | **not** applied (panel-specific diagnostics only) |

Recommendation artifact: [BAYES_H5O_SHADOW_POLICY_RECOMMENDATION_…](../05_validation/archives/BAYES_H5O_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json)

## What policy did it recommend?

| Field | Value |
|-------|--------|
| Channel | **keep_all_channels** (search, social, tv) |
| Geometry | NC full hierarchy, learned τ, `sigma_floor=0.05` |
| Prescale | z-score media + z-score log outcome |
| Sampler | extended MCMC 600/600/4, `target_accept=0.95` |
| Status | `recommended` |

Contrasts with H5m on sample panel: **drop tv** under social–tv collinearity (~0.99).

## Were any channels dropped or composited?

**No.** `channel_policy.mode=keep_all_channels`, `no_silent_dropping=true`. All three media columns retained.

## What claims became forbidden?

- No production Bayes, optimizer, DecisionSurface, or budget recommendations
- No Ridge replacement
- Keep-all weak-ID: forbid channel-level budget or causal decision use
- Shadow recommendations are not business decisions

## Did the frozen policy validate?

**Yes.** `validate_shadow_policy` passed on [h5o_examples_mmm_benchmark_geo_panel_v1_shadow_policy.json](h5o_examples_mmm_benchmark_geo_panel_v1_shadow_policy.json) (`policy_id=bayes_h5o_examples_mmm_benchmark_geo_panel_v1_shadow_policy_v1`).

## Did the shadow run converge?

**Yes** (this run):

| Metric | Value |
|--------|--------|
| convergence_status | `converged_diagnostic_only` |
| rhat_max | 1.0 |
| divergences | 0 |
| evidence_promotion_allowed | true (research only) |

Artifact: [BAYES_H5O_SHADOW_RUN_…](../05_validation/archives/BAYES_H5O_SHADOW_RUN_EXAMPLES_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json)

```bash
poetry run python -m mmm.research.bayes_h3_sandbox.h5_shadow_runner \
  --policy-path docs/06_investigations/h5o_examples_mmm_benchmark_geo_panel_v1_shadow_policy.json \
  --output-path docs/05_validation/archives/BAYES_H5O_SHADOW_RUN_EXAMPLES_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json
```

## Ridge comparison

**Not available** for this benchmark panel (Ridge diagnostic arm not executed).

## GeoX/CLS / calibration evidence

**Not available** (`calibration_evidence_available=false`).

## Is this panel usable as research shadow evidence?

**Yes — with scope limits:**

- Proves the **H5n → freeze → `--policy-path` replay** workflow generalizes beyond `sample_panel.csv`.
- Demonstrates **keep-all** path under low collinearity vs **governed drop** on the collinear pilot.
- Does **not** constitute production Bayes acceptance or client sign-off.

## What must happen before more panels?

1. Document each new panel with manifest + recommender artifact (stop on `do_not_run`).
2. Do **not** batch panels — one panel per milestone unless explicitly authorized.
3. Prefer panels with Ridge/GeoX contrast when available.
4. Production Bayes remains blocked behind Promotion Gate.

## Production boundary

**Blocked.** All production flags false; no optimizer / DecisionSurface / recommendations / prod TrustReport / Ridge replacement.
