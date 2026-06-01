# INV-H4-001 — Sparse partial-pooling shrinkage behavior

| Field | Value |
|-------|--------|
| **Investigation ID** | INV-H4-001 |
| **Title** | Why `WORLD-BAYES-H4-SPARSE-GEO` does not show expected shrinkage toward \(\mu_c\) |
| **Status** | **Open** — metric/indexing resolved; world/prior disposition pending (blocks H4c) |
| **Track** | Bayes-H4 research recovery — research sandbox only |
| **Related** | INV-071 · [bayes_h4_recovery_worlds_adr.md](../05_validation/bayes_h4_recovery_worlds_adr.md) · H4a/H4b pilot JSON |
| **Implementation** | `sparse_shrinkage_metrics.py` · `sparse_pooling_investigation.py` · `recovery_runner.py` |

---

## 1. Observed H4a / H4b results

| Pilot | Sampler | `shrinkage_ratio_sparse` (legacy vs **true** \(\mu_c\)) |
|-------|---------|--------------------------------------------------------|
| H4a threshold | fast (200/200, 2 chains) | **≈ 2.57** |
| H4b repeated seed 4400 | extended (600/600, 4 chains) | **≈ 2.57** |
| H4b repeated seed 4401 | extended | **≈ 2.65** |
| H4b repeated seed 4402 | extended | **≈ 2.73** |

**H4b classification:** `likely_model_prior_or_world_design` — not resolved by longer sampling.

**Other signals (healthy):**

- Conflict world: `conflict_warning_pass_rate = 1.0`
- All production authorization flags remain **false**
- Posterior \(\hat{R}\) on toy panels generally acceptable in pilots

---

## 2. Expected behavior

On `WORLD-BAYES-H4-SPARSE-GEO`:

- Generative sparse geo `dma_sparse` has **3 weeks** and an outlier \(\beta_{g,\text{tv}} = 0.85\) vs \(\mu_{\text{tv}} = 0.30\).
- H3 MVP partial pooling: \(\beta_{g,c} = \mu_c + z_{g,c}\,\tau_c\).
- **Diagnostic expectation:** posterior \(\mathbb{E}[\beta_{g,c} \mid y]\) for the sparse geo should move **closer to the pooling center** than the generative outlier was — i.e. shrinkage toward \(\mathbb{E}[\mu_c \mid y]\), not necessarily toward **true** \(\mu_c^\*\).

ADR provisional check (report-only): legacy ratio **&lt; 1** when comparing distances to **true** \(\mu_c\).

---

## 3. Actual behavior

**H4a/H4b (legacy metric only):** \(\|\hat\beta - \mu^\*\| / \|\beta^\* - \mu^\*\|\) stayed **&gt; 1** (≈ 2.5–2.7) across fast and extended MCMC — this compared pooling to **unknown-in-practice** true \(\mu^\*\).

**After INV-H4-001 metric fix (primary vs \(\hat\mu_c\)):** on the same baseline sparse world with fast MCMC, primary `shrinkage_ratio_sparse` **≈ 0.55 &lt; 1** — posterior \(\hat\beta\) is **closer to learned \(\hat\mu_c\)** than the generative outlier was to \(\hat\mu_c\). Legacy ratio on that refit remains **≈ 2.57** (recovery diagnostic only).

**Caveat:** toy-panel \(\hat\mu_c, \hat\beta\) can be poorly calibrated vs generative truth (small N, semi-log MVP); legacy and `mu_c_mae` still matter for **parameter recovery**, separate from **pooling-toward-\(\hat\mu\)** checks.

---

## 4. Hypotheses

| ID | Hypothesis | How tested |
|----|------------|------------|
| H1 | **Metric reference error** — ratio vs true \(\mu_c\) misstates pooling (target is posterior \(\mu_c\)) | Primary vs legacy ratio in `sparse_shrinkage_metrics.py` |
| H2 | **Posterior geo index mismatch** — \(\beta_{g,c}\) mapped to wrong DMA | `beta_geo_index_order` in fit artifact + `validate_posterior_index_mapping` |
| H3 | **Likelihood dominates pooling** — 3 sparse weeks support high local \(\beta\) | Variant `sparse_more_weeks` vs baseline |
| H4 | **Outlier too extreme** — partial pool still far from \(\mu\) | Variant `sparse_outlier_moderate` |
| H5 | **Prior too weak** — \(\tau_c\) prior allows geo-specific fit | Variant `sparse_stronger_tau_prior` (`tau_channel_prior_sigma=0.15`) |
| H6 | **False alarm world** — no outlier should not warn | Variant `sparse_no_outlier` |
| H7 | **Sampling instability** | Ruled down by H4b multi-seed extended runs |

---

## 5. Tests performed

| Test | Type | Purpose |
|------|------|---------|
| `test_bayes_h4_sparse_pooling_investigation.py` | fast | Toy posterior shrinkage algebra; decomposition schema; index mapping |
| `test_bayes_h4_recovery_worlds.py` | slow (PyMC) | Live sparse world emits decomposition |
| `sparse_pooling_investigation.run_sparse_pooling_investigation` | slow (optional) | Baseline + 4 diagnostic variants |
| H4b repeated pilot (committed) | slow (done) | Extended MCMC × 3 seeds |

**Metric audit (code):**

- **Legacy (H4a/H4b):** `shrinkage_ratio_sparse_vs_true_mu` = mean\(_{g\in\text{sparse},c}\) \(|\hat\beta_{g,c}-\mu^\*_c| / |\beta^\*_{g,c}-\mu^\*_c|\).
- **Primary (INV-H4-001):** `shrinkage_ratio_sparse` = same with **\(\hat\mu_c\)** as pool center.
- **Ratio &lt; 1** means posterior \(\hat\beta\) is closer to the pool center than the generative outlier was.

**Posterior indexing:** `model.fit_h3_sandbox_hierarchical` records `beta_geo_index_order` (sorted panel geo order) and `channel_index_order`; `recovery_runner` maps `beta_geo_channel_mean[str(geo_idx)][channel]`.

For current sparse world, sorted geo order matches `spec.geo_order` — **H2 unlikely** for baseline world.

---

## 6. Conclusion (after INV-H4-001b variant sweep)

| Finding | Status |
|---------|--------|
| **1. Metric issue** | **Resolved** — H4a/H4b sparse “failure” was **overstated** on legacy vs true \(\mu^\*\); primary vs \(\hat\mu_c\) shows pooling on baseline (≈ 0.55). Keep legacy as recovery diagnostic only. |
| **2. Indexing risk** | **Mitigated** — `beta_geo_index_order` / `channel_index_order` recorded; baseline `index_order_matches_spec: true`. |
| **3. Model / world design** | **Still open** — choose among revise sparse world, revise \(\tau\) prior, revise acceptance metrics, or document MVP pooling limits. |

**Roadmap:** Do **not** start **H4c** until disposition on (3). Next gate: **INV-H4-001b** sweep recorded → leadership pick **A/B/C/D** (below).

**Disposition:** Keep **INV-071** and **INV-H4-001** open for ADR threshold wording and sparse-world redesign. **H4c blocked.**

---

## 7. Governance disposition

| Flag | Value |
|------|--------|
| `research_only` | **true** |
| `approved_for_prod` | **false** |
| `prod_decisioning_allowed` | **false** |
| `production_promotion` | **false** |
| Ridge production path | **unchanged** |
| Optimizer / DecisionSurface / recommendations | **not connected** |

---

## 8. INV-H4-001b — Sparse variant sweep (2026-06-01)

**Artifact:** [archives/BAYES_H4_SPARSE_VARIANT_SWEEP_20260601.json](../05_validation/archives/BAYES_H4_SPARSE_VARIANT_SWEEP_20260601.json)  
**Command:** `run_sparse_pooling_investigation()` (fast MCMC: draws/tune 200, chains 2)

| Variant | Primary `shrinkage_ratio_sparse` (vs \(\hat\mu_c\)) | Legacy vs true \(\mu^\*\) | Classification |
|---------|---------------------------------------------------|---------------------------|----------------|
| **baseline** `WORLD-BAYES-H4-SPARSE-GEO` | **0.55** | 2.57 | Pooling toward \(\hat\mu\) present; legacy alarm was metric artifact |
| `sparse_no_outlier` | **0.21** | 32.9* | **Pass** — no false primary alarm; *legacy unstable (tiny \(\|\beta^\*-\mu^\*\|\)) |
| `sparse_outlier_moderate` | **0.27** | 3.20 | **Pass** — moderate outlier still pools toward \(\hat\mu\) |
| `sparse_stronger_tau_prior` | **0.28** | 1.90 | **τ prior helps legacy alignment** with true \(\mu^\*\); primary already &lt; 1 |
| `sparse_more_weeks` | **1.12** | 0.62 | **More weeks → less pooling toward \(\hat\mu\)** (search channel expands); legacy improves because \(\hat\mu\) tracks truth better |

### Variant interpretations

1. **`sparse_no_outlier`** — Primary &lt; 1; metric/model not suspect on pooling check. Do not use legacy ratio when true \(\beta \approx \mu^\*\) (denominator → 0).
2. **`sparse_outlier_moderate`** — Primary &lt; 1; partial pooling operates under moderate outlier.
3. **`sparse_stronger_tau_prior`** — Stronger \(\tau\) prior pulls legacy ratio down (1.90 vs 2.57); supports **B. revise τ prior** for production-adjacent research, not required for primary pooling signal.
4. **`sparse_more_weeks`** — Primary &gt; 1: additional sparse-geo data allows geo-specific fit away from \(\hat\mu\) on at least one channel; original **3-week** design was sparse/unstable for the **primary** pooling metric. Supports **A. revise sparse world** (weeks / outlier strength) before H4c.

### Re-read of H4a/H4b

Extended multi-seed pilots used the **legacy** metric only. Under the **primary** metric, baseline sparse pooling toward \(\hat\mu_c\) is **not** disproven by H4b; the open work is **world design**, **μ/β recovery on toy panels**, and **ADR acceptance wording** — not “pooling absent.”

---

## 9. Disposition options (pick one before H4c)

| Option | Action |
|--------|--------|
| **A. Revise sparse world design** | Reduce outlier gap and/or increase sparse weeks; re-commit pilot JSON |
| **B. Revise \(\tau\) prior** | Research default `tau_channel_prior_sigma` (diagnostic variants suggest benefit for legacy recovery) |
| **C. Revise acceptance metric** | ADR: primary vs \(\hat\mu_c\); legacy vs \(\mu^\*\) report-only; degenerate denominator guards |
| **D. Document limitation** | MVP partial pooling is diagnostic-only; sparse extreme worlds may not shrink toward \(\mu^\*\) |

**Recommended default:** **C** (already implemented) + **A** (tune official `WORLD-BAYES-H4-SPARSE-GEO` for stable dual-metric behavior) before H4c.

---

## 10. Next steps

1. ADR/registry: mark primary metric authoritative; H4c blocked pending **A or D** decision.
2. Re-run H4b repeated pilot with primary metric if extended JSON should be updated (optional).
3. Close INV-H4-001 when disposition **A/B/C/D** is recorded in ADR §11.
