# MMM v1.0.0 release notes

**Release date:** 2026-05-22  
**Tag:** `v1.0.0`

This document describes the first **production-ready** release scope for the geo-level MMM package. It is written to be explicit about what is safe for budget decisions and what remains research or diagnostic only.

---

## 1. Summary

v1.0.0 **freezes the production-safe path** for governed Ridge marketing mix modeling and budget decisions:

| Layer | v1 production scope |
|-------|---------------------|
| Framework | Ridge + Bayesian optimization (Ridge BO) |
| Model form | `semi_log` only for prod decide |
| Transforms | Geometric adstock + Hill saturation |
| Estimand | Full-panel Δμ via `mmm.planning.decision_simulate` |
| Decision APIs | `mmm decide simulate`, `mmm decide optimize-budget` |
| Safety | Fail-closed governance, artifact contracts, fingerprint matching |

This release **establishes** the governed Ridge semi-log full-panel Δμ decision path. It does **not** claim that every feature in the repository is production-ready.

**Production budget decisions must use `mmm decide`.** Training outputs, curves, decomposition charts, robust optimization research runs, uncertainty propagation summaries, Bayesian posterior outputs, and train-time decision stress reports may inform review—they are **not** substitutes for a persisted decision bundle produced through the decide path.

**What this release is not:** proof of causal incrementality, automatic experiment design, or “big-tech” autonomous MMM operations. Certification and readiness gates validate **contracts and internal consistency**, not that observational MMM will beat holdout in your market.

---

## 2. Production-safe scope

The following are intended for **`run_environment: prod`** when configured with the canonical modeling contract and governance flags documented in [prod_safety_checklist.md](prod_safety_checklist.md) and [decision_runbook.md](../03_planning/decision_runbook.md).

| Capability | Notes |
|------------|--------|
| **Ridge + BO** | Primary production training and planning framework |
| **`semi_log` model form** | Only model form supported for prod `mmm decide` |
| **Geometric adstock** | Canonical prod adstock |
| **Hill saturation** | Canonical prod saturation |
| **Full-panel Δμ** | Decision estimand; same panel and transforms as fit |
| **`mmm decide simulate`** | Scenario Δμ vs BAU with decision bundle |
| **`mmm decide optimize-budget`** | Constrained media allocation on full-panel simulation |
| **Production artifact validation** | Decision tier, semantics, economics metadata |
| **Fingerprint matching** | Train→decide panel and config lineage |
| **Replay calibration governance** | Evidence registry, weighted replay, prod replay gate |
| **Optimizer certification** | Synthetic-surface checks (auto-enabled on prod train) |
| **Synthetic certification** | Exact DGP numerical consistency (runtime + CI shared) |
| **Production readiness report** | Rollup `approved_for_prod` and blocked reasons |
| **Promotion workflow** | Optional `require_promoted_model_for_prod_decision` |
| **Decision trace** | `decision_trace.json` alongside prod `--out` artifacts |
| **Calibration freshness / drift readiness** | Stale calibration and coefficient drift signals |
| **Model release state** | `planning_allowed` required for prod decide |

Prod-bound config template: [`examples/prod_train_template.yaml`](../../examples/prod_train_template.yaml).

---

## 3. Not production-supported

Do **not** treat the following as production budget truth or supported prod decide paths in v1.0.0:

| Item | Status in v1.0.0 |
|------|------------------|
| **Bayesian production decisioning** | `mmm decide optimize-budget` **blocked** in prod for `framework=bayesian` |
| **`log_log` production decisioning** | **Blocked** in prod |
| **Weibull / logistic / log transform production path** | Not canonical prod stack; blocked unless explicitly validated |
| **Robust optimization as prod allocator** | **Research-only** extension |
| **Causal lift guarantees from observational MMM** | **Not claimed** — experiments inform calibration, not proof |
| **Auto-retraining** | **Not included** |
| **Auto-promotion** | **Not included** — promotion is explicit operator workflow |
| **Agentic orchestration** | **Not included** |
| **State-space / time-varying coefficients** | **Not production-supported** |
| **Response curves as primary allocator** | Diagnostic / explanatory |
| **Decomposition as budget truth** | Diagnostic |
| **Uncertainty propagation as decision distribution** | Research / diagnostic report |
| **Train-time decision stress at decide time** | `stress_scope` is `train_time` — not recomputed on `mmm decide` |
| **Continuous validation / decision validation extensions** | Research / monitoring hooks |

Research features may run in `run_environment: research` with appropriate disclosures. They must not be described as production-certified without separate organizational sign-off.

---

## 4. Major capabilities

### Decision contracts

- **Full-panel Δμ estimand** — counterfactual revenue change on the training panel structure, not curve extrapolation alone.
- **Decision-safe contract** — `decision_safe`, governance gates, and unsupported-question disclosures on prod surfaces.
- **Artifact tier validation** — incomplete lineage, semantics, or economics metadata **aborts** prod decide.
- **Train→decide fingerprint matching** — panel and config fingerprints must align unless an explicit waiver policy applies.

See [decision_artifact_contract.md](decision_artifact_contract.md), [artifact_schema.md](artifact_schema.md).

### Replay and calibration

- **Evidence registry** — structured experiment observations for matching and weighting.
- **Weighted replay** — integrates replay units into calibration objectives where configured.
- **Replay prod gate** — blocks or warns when replay evidence is missing or inconsistent in prod paths.
- **Full-panel replay transform with estimand mask** — replay aligned to panel geometry and estimand semantics.
- **Fold-aligned replay mode** — `replay_refit_mode: fold_aligned` for CV-aligned replay refit.
- **Replay generalization gap reporting** — severity surfaced in calibration summary and readiness.

See [calibration](../02_concepts/calibration.md), [decision_runbook.md](../03_planning/decision_runbook.md).

### Governance

- **Production readiness report** — `approved_for_prod`, `blocked_reasons`, optimizer/synthetic/repro status.
- **Promotion workflow** — optional promoted-run validation at decide time.
- **Model release state** — e.g. `planning_allowed` vs `research_only`.
- **Decision trace** — audit JSON for prod decisions.
- **Calibration freshness** — stale calibration warnings and drift review hooks.
- **Coefficient drift readiness** — calibration readiness report ties to planning gates.

See [production_readiness.md](production_readiness.md), [promotion_workflow.md](promotion_workflow.md), [decision_trace.md](decision_trace.md), [calibration_freshness.md](calibration_freshness.md).

### Certification

| Report | Role |
|--------|------|
| **Synthetic certification** | Exact DGP checks (single `CHECK_REGISTRY` for runtime + CI) |
| **Optimizer certification** | Grid optimum / directional fallback modes on synthetic surfaces |
| **Reproducibility certification** | Independent-run evidence when `reference_run_path` set |
| **Performance certification** | Optional timing / scale scenarios |
| **Decision stress** | Train-time behavioral or signal-only stress (`stress_scope`) |
| **Nightly validation** | CI workflow runs certification suites |

See [synthetic_certification.md](synthetic_certification.md), [optimizer_certification.md](optimizer_certification.md), [decision_stress.md](decision_stress.md).

### Research / diagnostic (not budget truth)

- **Bayesian experiment likelihood** — research-only likelihood path; does not enable prod Bayesian optimize.
- **Bayesian hierarchy** — partial pooling research paths.
- **Robust optimization research** — scenario sets over coefficients; not prod allocator.
- **Uncertainty propagation report** — buckets and legacy decomposition; diagnostic.
- **Continuous validation** — monitoring extension hooks.
- **Decision validation** — research extension for decision-path checks.

See [decision_vs_research.md](../02_concepts/decision_vs_research.md), [robust_optimization_research.md](../02_concepts/robust_optimization_research.md).

---

## 5. Safety guarantees

v1.0.0 enforces the following on **`run_environment: prod`** (subject to configured governance flags):

| Guarantee | Behavior |
|-----------|----------|
| **Prod `log_log` blocked** | `assert_prod_decision_not_log_log` on decide paths |
| **Bayesian prod decisioning blocked** | `mmm decide optimize-budget` raises for Bayesian in prod |
| **Unsupported transforms blocked** | Canonical media stack validation for prod modeling/decide |
| **Stale / incomplete decision artifacts rejected** | Semantic contract and business-surface validation fail closed |
| **Panel fingerprint mismatch rejected** | Unless explicit waiver policy; severe warning on payload |
| **Full-panel replay refit in prod** | Requires waiver / policy acknowledgment where configured |
| **Failed production readiness** | **Severe warning** always on prod decide JSON when `approved_for_prod=false` |
| **Strict certification gate** | `governance.require_production_certification: true` → **PolicyError** if not approved |
| **Unsafe decision APIs** | `allow_unsafe_decision_apis: false` mandatory in prod YAML |
| **CV in prod** | Explicit calendar strategy; `cv.mode=auto` forbidden in prod |

Warnings vs hard gates are documented in [production_readiness.md](production_readiness.md).

---

## 6. Known limitations

Read these before treating v1.0.0 as “fully validated in production”:

1. **Does not prove causal incrementality.** Observational MMM plus replay calibration improves discipline; it does not replace randomized lift where that is the estimand.
2. **Replay calibration depends on experiment quality.** Poor matches, stale units, or weak evidence propagate into warnings and readiness blocks—not into automatic correction.
3. **Optimizer certification uses synthetic surfaces.** `certification_mode=directional_fallback` proves channel dominance on controlled DGPs, not exact optimum recovery on real panels.
4. **Production readiness is a contract/gate, not causal proof.** `approved_for_prod` means certification rollups passed configured rules, not that spend will outperform holdout.
5. **Robust optimization remains research-only.** Do not wire robust scenario allocators to prod budget systems.
6. **Bayesian outputs remain research-only** for production budget decisions in v1.0.0.
7. **Synthetic certification validates known DGPs** — internal numerical consistency, not all real-world panel behavior (e.g. collinearity edge cases may remain integration-only tests).
8. **Decision stress is train-time.** It is not recomputed at `mmm decide`; do not treat it as decide-time validation.
9. **Reproducibility self-cert alone is insufficient** for strict readiness; use `reference_run_path` for independent-run evidence when required.

---

## 7. Migration notes

### Runtime

- **Python 3.11+ required** (`requires-python` in `pyproject.toml`).
- **Python 3.10 dropped** — upgrade CI and local venvs before adopting v1.0.0.

### Modeling contract

- **Production canonical path:** `framework: ridge_bo`, `model_form: semi_log`, geometric adstock, Hill saturation.
- **`log_log` is research-only** for production decisioning.
- Set `prod_canonical_modeling_contract_id` and `objective.normalization_profile: strict_prod` in prod YAML (see [config_yaml.md](../01_getting_started/config_yaml.md)).

### Decide behavior changes

- Decide paths may **fail** on fingerprint mismatch, incomplete `ridge_fit_summary`, or artifact contract violations—this is intentional fail-closed behavior.
- **`governance.require_production_certification: true`** — prod decide fails closed when `production_readiness_report.approved_for_prod` is false; default `false` still emits severe warnings.
- **`governance.require_promoted_model_for_prod_decision: true`** — prod decide requires valid promotion record and fingerprint alignment.

### Certification on train

- Prod trains auto-run **optimizer certification** (`run_environment: prod` or `extensions.optimizer_certification.enabled: true`).
- Every train emits synthetic, stress, and production readiness reports on the extension report.

---

## 8. Example production checklist

Short operator checklist for a prod budget decision:

1. **Train** with `run_environment: prod`, `semi_log`, geometric adstock, Hill saturation, explicit calendar CV, and `examples/prod_train_template.yaml` (or equivalent).
2. Confirm **replay evidence** passes the replay prod gate and calibration readiness is current.
3. Review **`production_readiness_report`**: `approved_for_prod: true`, no unexpected `blocked_reasons`; note warnings (e.g. directional optimizer cert).
4. **Promote** the run if `require_promoted_model_for_prod_decision` is enabled.
5. Run **`mmm decide simulate`** or **`mmm decide optimize-budget`** with `--extension-report` and `--out` (prod requires persisted artifact).
6. Save **`decision_bundle`** and **`decision_trace.json`** from the output directory.
7. Review **`unsupported_questions`**, **`decision_fingerprint_warnings`**, and production readiness warnings on the payload—do not ship decisions with unexplained severe warnings.

See [operator_runbook.md](operator_runbook.md) and [prod_safety_checklist.md](prod_safety_checklist.md).

---

## 9. Validation run

Commands used for the v1.0.0 release candidate:

```bash
ruff check mmm tests
pytest tests/ -q -m "not slow"
```

**CI:** GitHub Actions on Python **3.11** and **3.12** (unit, docs, artifact, and reproducibility jobs).

**Nightly:** extended certification suites (synthetic, optimizer, decision stress, production readiness) via `.github/workflows/nightly.yml`.

For a full slow suite locally:

```bash
pytest tests/ -q
```

---

## Related documentation

| Topic | Document |
|-------|----------|
| Prod decide runbook | [decision_runbook.md](../03_planning/decision_runbook.md) |
| Config reference | [config_yaml.md](../01_getting_started/config_yaml.md) |
| Decision vs research | [decision_vs_research.md](../02_concepts/decision_vs_research.md) |
| Changelog entry | [CHANGELOG.md](../../CHANGELOG.md) |
