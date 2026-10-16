# MMM transformation/model-selection audit (documentation only)

> **Status:** audit record, not a selection decision. No family is adopted,
> no transform is changed, and no implementation authority follows.
> **Task:** `MMM_TRANSFORM_SELECTION_AUDIT_001` (docs-only).
> **Verified baseline:** MMM `origin/main = 00cad45de5f9f04c41a6eed51ccbc22149a42b54`
> (superset of audit baseline `4ccf8ca`; transform inventory below verified
> against this tree). All `mmm/` file:line citations were read from this
> tree during publication. External claims carry pinned sources where
> verified; all other external detail is explicitly labeled for Phase 0
> recheck and must not be treated as verified.

## 1. Current MMM inventory (implemented vs canonical)

| Family | Location | Registry/canonical status |
|---|---|---|
| Geometric adstock (infinite lag, immediate peak) | `mmm/transforms/adstock/geometric.py:12` (`decay in (0,1)`) | Canonical; BO fast path only (`mmm/transforms/stack.py:41-44` raises otherwise) |
| Weibull adstock (`shape`/`scale`/`max_lag`, PDF kernel — only delayed-peak-capable kernel in repo) | `mmm/transforms/adstock/weibull.py:16` (defaults 1.5/2.0/12) | Registry-available; excluded from canonical stacks and BO fast path |
| Hill saturation | `mmm/transforms/saturation/hill.py:12` | Canonical |
| Logistic saturation | `mmm/transforms/saturation/logistic.py:12` | Registry-available; non-canonical |
| Log saturation | `mmm/transforms/saturation/log_sat.py:13` | Registry-available; non-canonical |

Canonical production stacks are geometric+Hill only
(`mmm/contracts/canonical_transforms.py:16-23`). The BO fast path
(`mmm/transforms/stack.py:24-60`) constructs one shared adstock/saturation
object applied identically to every channel — production fitting is
**shared-parameter, single-family**. Per-channel heterogeneity appears only
in validation harnesses (e.g. `mmm/validation/synthetic/optimizer_truth.py`
`shared_ridge_transform_params` / `per_channel_uniform`). There is no
delayed-peak-first-class alternative with selection evidence, no
per-channel family/decay selection, no order/normalization switch, and no
sign discipline (Ridge is unconstrained).

## 2. Source-pinned external precedents

### Meridian — VERIFIED from primary docs (fetched October 2026)

- Adstock weight functions `geometric` and `binomial`, per-channel spec
  including dict form; `max_lag` guidance geometric 2–10, binomial 4–20
  ([adstock-decay-spec](https://developers.google.com/meridian/docs/advanced-modeling/set-adstock-decay-spec-parameter), last updated 2026-09-05).
- Binomial form: `w(s) = (1 − s/(1+L))^α*`, `α* = 1/α − 1`, x-intercept
  always at `L+1`; convex/linear/concave by α ≶ 0.5. It is a deterministic
  decay-weight curve — **not** a Binomial distribution and **not** a
  negative-binomial delay kernel. Both Meridian kernels peak at lag 0;
  they span persistent-vs-immediate, not delayed-peak.
- Hill `ec`/`slope` with concave-default prior and explicit S-shape
  optimization/convergence warnings; `hill_before_adstock` order switch;
  alpha prior default `U(0,1)` with Beta-shaping guidance; extrapolation
  caution ([media-saturation-lagging](https://developers.google.com/meridian/docs/advanced-modeling/media-saturation-lagging), last updated 2026-07-08).
- Meridian explicitly warns binomial misapplication underestimates
  short-term effects: do not assume all TV is delayed or social immediate.

### PyMC-Marketing — PARTIALLY VERIFIED (repo landing fetched; module APIs for Phase 0 recheck)

- Verified: `GeometricAdstock(l_max)` + `LogisticSaturation` quickstart
  defaults, pluggable custom adstock/saturation functions, channel-dim
  (per-channel) priors, NUTS sampler choice (BlackJax/NumPyro/Nutpie) +
  GPU, time-varying intercept/media via GPs, lift-test calibration,
  Python ≥3.12 ([repo](https://github.com/pymc-labs/pymc-marketing)).
- For Phase 0 recheck: exact delayed-adstock/saturation module names and
  signatures, default prior families, funnel-model interaction with
  transform selection.

### Robyn — PARTIALLY VERIFIED (repo landing fetched; R API details for Phase 0 recheck)

- Verified: Ridge + Nevergrad hyperparameter search (thetas/decay,
  Hill alpha/gamma), Prophet trend/season decomposition, calibration
  inputs, gradient-based allocator ([repo](https://github.com/facebookexperimental/Robyn)).
- For Phase 0 recheck: exact Weibull parameterization (`shape`/`scale`/
  `theta` roles), per-channel hyperparameter bounds, calibration-loss
  weighting. Robyn is the closest precedent for MMM's BO path:
  evidence-driven per-channel decay/saturation search within
  geometric/Weibull-class kernels.

## 3. Gap analysis (summary)

Immediate-decay (geometric ✓), persistent (binomial-class ✗; Weibull
shape>1 partially covers delayed-peak but is non-canonical and
unevaluated). Production fits shared parameters; all three sources go
per-channel — the largest structural gap. Global defaults only for
bounds/priors/lag windows (Meridian's max_lag + prior-curve practice is
the template). Fixed adstock→saturation order (Meridian ships an order
switch). No transform-family recovery worlds; decay↔slope↔β compensation
unmeasured on the production path; calibration enters diagnostics, not
selection. Compute: shared Ridge BO is cheap; per-channel family search
multiplies BO dimension (Robyn precedent, still CPU-feasible); Bayesian
per-channel inference is GPU/NUTS-class (PyMC precedent) — excluded until
H-track gates pass.

## 4. Exclusions (with rationale)

- S-shaped Hill as default (Meridian convergence/optimization warnings;
  keep concave-default).
- Negative-binomial delay kernel as a borrowed item (no source ships it;
  evaluate only if Weibull/delayed alternatives fail recovery worlds).
- Full Bayesian per-channel transform inference now (cost + H-gate order).
- Time-varying media efficiency (GP-class; revisit after static
  per-channel selection proves out).

## 5. Phased plan (deferred) and revisit trigger

- **Phase 0 (docs-only, small):** pin upstream versions, verify §2
  recheck items against pinned APIs, record exclusions.
- **Phase 1:** per-channel decay/Hill search in BO fast path (bounded,
  CPU) + family-recovery worlds (geometric vs Weibull vs binomial-class)
  measuring decay↔slope↔β compensation.
- **Phase 2:** order switch + lag-window/prior discipline;
  calibration-informed priors.
- **Phase 3 (gated):** Bayesian per-channel inference only after H-track
  reliability gates.
- **Revisit trigger:** execute iff a recovery world shows shared-geometric
  failing a channel class, or an explicit owner goal requests per-channel
  lag. This plan displaces no current fixes or handoff work; a long
  detour stays postponed.
