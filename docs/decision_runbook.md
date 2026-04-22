# MMM Decision Runbook

## 1. Purpose

This document defines how to safely use the MMM system for decision-making.

---

## 2. Decision vs Research Modes

| Topic | Decision-safe (Production) | Research |
|--------|----------------------------|----------|
| **Environment** | `run_environment=prod` | `run_environment=research` (or non-prod) |
| **Planning path** | Full-panel **μ simulation** only (`planner_mode=full_model`) | May use `planner_mode=curve_local` for exploration only |
| **Curves** | Diagnostics / warm starts only — **not** the budget truth | Curves allowed for analysis |
| **Calibration** | **Replay** with explicit estimand, frames, lift scale, and (when configured) **approved experiment registry** | Research-only coefficient mismatch utilities exist under `mmm.research_legacy` — not used in training or prod |
| **APIs** | `allow_unsafe_decision_apis=false` (config default; required for decision-safe prod posture) | Unsafe / experimental APIs allowed only when explicitly set |
| **Governance** | Scorecard + gates must pass for optimization approval | Advisory or overridden for exploration |

**Rules**

- In production, treat **only** full-model simulation outputs as eligible for allocation decisions.
- In research, anything labeled diagnostic or experimental is **not** portable to production without re-validation under production rules.

---

## 2a. Production policy (code-enforced)

These rules are validated when loading YAML (`MMMConfig`); do not rely on operators to remember them.

- **`allow_unsafe_decision_apis`**: **must be `false`** in `run_environment=prod`. Unsafe / legacy surfaces (e.g. curve-bundle optimizers) are not available in prod.
- **`extensions.optimization_gates.enabled`**: **must be `true`** in prod. Budget optimization without gates is rejected.
- **Bayesian budget / planning in prod**: **Optimization approval is disabled** for Bayesian models in prod until a future validated implementation re-enables it (governance always sets `approved_for_optimization` false with an explicit note). **CLI `optimize-budget` also rejects** `framework=bayesian` in prod.
- **Bayesian prod training**: **`bayesian.posterior_predictive_draws` must be > 0** in prod (PPC path required by config).

---

## 3. Canonical Economic Definition (CRITICAL)

**Definition (only decision quantity)**

- **Δμ = μ(candidate plan) − μ(baseline plan)**  
  Both **μ** values are computed with the **same** model, on the **same** modeling scale, using the **declared** baseline and candidate spend (and control overlays when supplied).

**Rules**

- **Δμ is the only quantity** that budget simulation and full-model optimization are defined to optimize or report as the decision objective.
- Curves, decompositions, and coefficient tables **do not** redefine Δμ; they are supporting diagnostics unless explicitly mapped (by you) to the same estimand — the system does not treat them as equivalent.
- **Baseline = BAU** unless you set another baseline type and accept the disclosure implications.
- Any non-BAU baseline must be **explicitly labeled** in inputs and in downstream reporting (`baseline_type`, `baseline_definition`, metadata). Do not silently swap baselines between runs.

---

## 4. Simulation Contract

**What `simulate()` (full-model path) does**

- Builds predictions using the **full** fitted model path (media transforms, controls, seasonality as modeled).
- Applies a **counterfactual spend plan** (candidate) vs the **declared baseline plan**.
- Returns **Δμ** (and related KPI summaries) **relative to that baseline**, with explicit aggregation semantics (e.g. mean μ over panel rows — see result fields such as `aggregation_semantics`).

**Default assumptions (unless you override in API)**

- Spend path is **constant per channel over time** unless a different path is explicitly specified.
- **Controls** follow the scenario: defaults tie to **observed** behavior unless you supply an explicit controls / overlay plan.
- **Geo and time** aggregation follows the contract encoded in the simulation result (do not assume a different level than returned).

**What full-model simulation does *not* do**

- Does **not** forecast **unobserved** macro shocks, competitor actions, or structural breaks.
- Does **not** assert **causal** validity of Δμ outside a **replay-aligned** calibration program with matching estimand.
- Does **not** remove **extrapolation risk** when candidate spends lie far outside the training support.

---

## 5. Calibration Contract (VERY IMPORTANT)

**What calibration compares**

- **Observed experiment lift** (on the declared KPI and aggregation)  
  **vs**  
- **Model-implied lift** from applying **observed vs counterfactual** spend frames (or equivalent replay construction) through the **same** prediction pipeline used for decisions.

**Must match exactly**

- **Geo scope**
- **Time window**
- **KPI column / scale**
- **Aggregation** (mean vs sum, horizon, etc.)
- **Lift scale** (e.g. mean KPI-level delta) — must align with replay estimand JSON and training target

**Hard rules**

- **Raw media coefficients are not incremental experiment lift.** Never interpret calibration loss as “lift error on coefficients.”
- **Calibration loss is only meaningful** when the experiment **estimand** matches the replay **estimand** and economics metadata.
- **Mismatched estimand** ⇒ calibration metrics are **invalid** for go/no-go — do not use them to approve budgets.

**Replay semantics (Ridge canonical calibration path)**

- Replay compares **observed experiment lift** to **model predictions on replay spend frames** (`predict_fn` on observed vs counterfactual panels).
- That **`predict_fn` is the same canonical μ path** as Ridge training / `predict()` (design matrix + fitted coefficients), not a separate or hidden formula.
- **Coefficients are not calibration targets** and are not scored against experiments in training or prod paths.

---

## 6. Optimization Safety Rules

**Facts**

- The optimizer objective is framed on **Δμ** from full-panel simulation (not on curve-local shortcuts when operating in decision mode).
- **`optimizer_success` is not `decision_safe`.** A converged solve can still be unsafe.

**Decision-safe requires *all* applicable items**

- **Governance**: reporting / optimization flags and gates consistent with your runbook threshold (including Bayesian diagnostics when applicable).
- **Calibration**: within policy limits when calibration is active for decisions (replay chi² / governance notes as configured).
- **No dangerous extrapolation**: candidate plans inside defensible spend support; document exceptions.
- **Stable solution**: re-solves or stability checks do not show large allocation churn under small perturbations when stability gating is enabled.

**Instability**

- Multiple local optima and **sensitivity to small spend perturbations** are normal numerical risks. If stability diagnostics fail, **do not** treat the allocation as decision-safe.

---

## 7. Uncertainty Policy

**Ridge**

- **Point estimates** for decision-facing Δμ unless you have explicitly enabled and validated posterior draw paths in a non-prod workflow.
- **No decision-safe monetary confidence intervals** in production Ridge (`ridge_forbids_precise_monetary_ci` policy). Do not ship “95% revenue CI” style claims from Ridge prod.

**Bayesian**

- Intervals / draw-based risk are **only** supportable when **posterior diagnostics** and (when required) **PPC / decision inference gates** pass under your configuration.
- If diagnostics or PPC fail governance, treat Bayesian outputs as **experimental** for decisions.

**General**

- **Never** treat uncertainty bands as **guaranteed** bounds on realized business outcomes unless the full validation chain (data, model, estimand, PPC where required) explicitly supports that claim.

---

## 7a. Posterior and risk-aware budget optimization

**Certified semantics (do not mis-describe)**

- **Posterior / risk-aware** budget optimization (`optimize_budget_risk_aware`, draw-based objectives) uses **precomputed Δμ draws** from `delta_mu_draws_linear_ridge` / `delta_mu_draws_hierarchical_geo_beta` at each candidate spend vector.
- It is a **performance-certified approximation layer**: the optimizer does **not** re-run full NUTS each SLSQP step, and you must **not** describe it as “the optimizer independently samples the full simulator every iteration.”
- Draws and point **Δμ** from `mmm.planning.decision_simulate.simulate` share the **same μ construction and canonical Δμ contract** (same design path; risk code composes objectives from the draw distribution of that Δμ).

**Artifact honesty**

- Outputs from this path include metadata such as `posterior_planning_mode: "draw_based_approximation"`, `economics_contract_version`, `draw_source_artifact`, and **`decision_safe: false` in prod** until the product explicitly certifies otherwise.

---

## 8. Panel Data QA Requirements

**Required properties before trusting any decision**

- **Keys**: **No duplicate (geo, week)** rows in the modeling panel.
- **Coverage**: Understand and document **missing (geo, week)** cells vs a full grid; large gaps are a **warning** at minimum.
- **Spend**: **No negative** channel spend; all-zero rows across all channels should be rare and explained.
- **Continuity**: **Extreme spikes** in `log1p(spend)` vs the rest of the panel should be reviewed (automated QA may **warn**).
- **KPI**: Target column is **consistent** with the business definition used in calibration and reporting.

**Production behavior**

- **Block-level** panel QA issues (e.g. duplicate geo-week) **must block** production training when `extensions.panel_qa.prod_block_severity=block` (recommended for prod).
- **Warn-level** issues in prod should **downgrade** what you treat as planning-safe until resolved or explicitly accepted with sign-off.

---

## 9. Experiment Usage Rules

Every experiment used for calibration must declare (and you must verify):

- **Geo** (or geo scope rule)
- **Time window**
- **KPI** and **units**
- **Aggregation** aligned to replay JSON

**Production**

- When `calibration.require_approved_experiment_registry` is enabled, **every replay unit** must carry an **`experiment_id`** present in the durable registry with **`approved`** status.
- **Unapproved or unknown** experiment IDs **block** replay usage in that configuration.

**Estimand mismatch**

- If any of geo / time / KPI / aggregation / lift scale disagrees between experiment and model replay, **invalidate** calibration conclusions for that unit.

---

## 10. Diagnostic vs Decision Outputs

| Class | Examples | May be used for budgeting? |
|--------|----------|----------------------------|
| **Diagnostic** | Response **curves**, curve stress tests, identifiability, decomposition (often log / modeling scale), sensitivity sweeps | **No** |
| **Decision** | **`simulate()`** on full model, **optimize-budget** via full-model simulation, **replay-aligned** calibration metrics | **Yes** (subject to governance + QA + stability) |

**Rule**

- **Diagnostic outputs must not be used for budgeting decisions**, even if they look like “ROI” or “response.” Only **full-model Δμ** paths qualify.

---

## 11. Things You Must Never Do (CRITICAL SECTION)

- Use **curve** or curve-bundle outputs as the **primary** basis for budget decisions.
- **Extrapolate** far outside observed spend support and still claim decision safety.
- Treat **decomposition** or attribution-style splits as **dollar truth** for P&L.
- Treat a **low calibration loss** as **proof of causality** or incremental validity without estimand alignment.
- Assume **`optimizer_success` ⇒ safe allocation** without governance, calibration, and stability checks.
- Use **Bayesian** uncertainty in decisions **without** checking diagnostics and required PPC gates.
- **Ignore or relabel baseline** when interpreting **Δμ** across runs or slides.

---

## 12. Decision Checklist

Before using model outputs for allocation or external commitments:

- [ ] **Data** passed panel QA at appropriate severity for prod.
- [ ] **Baseline** is explicit (default BAU or labeled alternative).
- [ ] **Simulation contract** read: aggregation, spend path, controls, horizon.
- [ ] **Calibration** (if used) is replay-aligned with matching estimand and approved registry when required.
- [ ] **Governance** flags and gates match the intended decision (reporting vs optimization).
- [ ] **Optimization stability** checked when using optimized allocations.
- [ ] **Outputs labeled** exact vs approximate, **decision_safe** vs diagnostic, modeling scale vs KPI level — and Ridge prod **excludes** forbidden monetary CI claims.
- [ ] **Production policy** (§2a): unsafe APIs off, gates on, Bayesian planning posture understood.
- [ ] **Draw-based optimization** (§7a): if using risk/posterior optimizers, read `posterior_planning_metadata` and do not mis-describe as full-simulator-per-step sampling.
