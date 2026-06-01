# INV-H4-001 — Sparse partial-pooling shrinkage behavior

| Field | Value |
|-------|--------|
| **Investigation ID** | INV-H4-001 |
| **Title** | Why `WORLD-BAYES-H4-SPARSE-GEO` does not show expected shrinkage toward \(\mu_c\) |
| **Status** | **Open** (blocks Bayes-H4c extended worlds) |
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

- Legacy metric \(\|\hat\beta - \mu^\*\| / \|\beta^\* - \mu^\*\|\) stays **&gt; 1** (often **≈ 2.5–2.7**) across fast and extended MCMC.
- Posterior \(\hat\beta_{g,c}\) for sparse TV remains **near or beyond** the generative outlier (not pulled toward \(\mu^\*\)).
- Primary metric (vs **posterior** \(\hat\mu_c\)) must be recomputed on new fits; decomposition is emitted per run in `sparse_shrinkage_decomposition`.

**Interpretation (provisional):** the sandbox is **not failing to converge**; it is **not exhibiting the ADR’s naive shrinkage check** on this world. That may be correct Bayesian behavior (likelihood dominates with sparse weeks), a **metric/reference issue**, and/or **weak \(\tau_c\) prior** relative to likelihood.

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

## 6. Conclusion (current)

| Question | Answer |
|----------|--------|
| Fast-MCMC artifact only? | **No** — H4b extended multi-seed still &gt; 1 on legacy metric |
| Index swap bug? | **Unlikely** on baseline (orders align); guard added for future worlds |
| Metric definition? | **Partially** — legacy vs true \(\mu_c\) is a stricter/misaligned check; primary metric uses posterior \(\hat\mu_c\) |
| Model / prior / design? | **Most likely** — sparse weeks + outlier \(\beta\) + default \(\tau_c\) prior allow geo-specific posterior near MLE; **investigate with diagnostic variants** |

**Disposition:** Keep **INV-071** and **INV-H4-001** open. Do **not** promote Bayesian output or start **H4c** extended worlds until sparse pooling behavior is understood or ADR acceptance metrics are revised to match partial-pooling semantics.

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

## 8. Next steps

1. Run `run_sparse_pooling_investigation()` (slow) and record variant ratios in this doc or a small JSON appendix.
2. If primary ratio &lt; 1 but legacy &gt; 1, update ADR §6 to document both metrics.
3. If all variants fail shrinkage, revise MVP prior / world design before H4c.
4. Close INV-H4-001 only when behavior is explained **or** acceptance criteria are explicitly revised.
