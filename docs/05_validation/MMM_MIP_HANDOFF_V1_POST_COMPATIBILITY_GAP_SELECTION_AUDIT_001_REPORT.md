# Post-compatibility MMM–MIP handoff gap-selection audit

**Audit ID:** `MMM_MIP_HANDOFF_V1_POST_COMPATIBILITY_GAP_SELECTION_AUDIT_001`
**Audited base commit:** `c96848e`
**Verdict:** `post_compatibility_gap_audit_complete_next_narrow_producer_task_selected_no_simulation_response_surface_consumer_or_interface_freeze_authorization`

## Executive selection

Select exactly one next task:

`MMM_MIP_HANDOFF_V1_SUPPORTED_RANGE_EVIDENCE_001` — establish a versioned,
producer-owned, positive supported-range evidence artifact that both a future
public simulation export (R11) and a future response-surface evidence contract
(R12) can reference. This is the narrow shared prerequisite, not an
authorization to simulate, optimize, recommend, or expose a response surface.

The selected dependency model is **Model C: narrow shared prerequisite first**.
`mmm.planning.decision_simulate.simulate` is the canonical internal full-panel
Ridge Δμ computation, while `mmm.decomposition` and `mmm.simulation.engine`
provide diagnostic curves and grid interpolation. Neither supplies a stable,
positive, serialized supported-range record with identity, scope, provenance,
restrictions, diagnostic evidence, and artifact references. Existing
`UNSUPPORTED_EXTRAPOLATION` failures prove only the negative outcome.

## Reconciled requirement matrix

| Requirement | Owner | Status | Current evidence | Missing producer-boundary evidence | Recommendation |
|---|---|---|---|---|---|
| `R11_PUBLIC_SIMULATION_EXPORT` (historical `R11_FULL_PANEL_DELTA_MU_SIMULATION`) | MMM | PARTIAL | `decision_simulate.simulate` computes full-panel Δμ; `PlanningScenario`/`BaselinePlan` model internal inputs; point and gated posterior summaries exist. | No public plan identity/schema/version; no supported-range contract; no governed uncertainty projection; no typed simulation payload, manifest linkage, artifact reference, or goldens. | Defer until supported-range evidence exists. |
| `R12_RESPONSE_SURFACE_EVIDENCE` | MMM | PARTIAL | `ResponseCurve`, curve bundles, typed quantity envelopes, curve diagnostics, and diagnostic grid interpolation exist. | No public surface identity/version; no governed domain/range record; no public uncertainty, lineage, limitation linkage, typed failure integration, or goldens. | Defer until supported-range evidence exists. |

## Internal evidence is not a public artifact

### R11 evidence

- `mmm.planning.decision_simulate.SimulationResult` serializes internal
  full-panel quantities including baseline and candidate means, Δμ, spend
  summaries, candidate construction, and optional posterior percentiles.
- `mmm.planning.scenario.PlanningScenario` hashes internal scenario semantics,
  and `mmm.planning.baseline.BaselinePlan` retains baseline provenance. Neither
  is a versioned MMM–MIP public producer contract.
- `mmm.contracts.mip_export.MMMSimulationResultArtifact` contains only
  `scenario_id` and an unconstrained `delta_mu_summary`; the runtime adapter
  puts raw `simulation_result` into `source_payload`, marks it blocked, and
  does not make it consumer-ready.
- The current run manifest, failure packet, calibration lineage, diagnostics,
  compatibility registry, and goldens provide reusable infrastructure, but
  they do not define a simulation artifact or successful simulation fixture.

### R12 evidence

- `mmm.decomposition.curves.ResponseCurve` has a spend grid, modeling-scale
  response, and marginal ROI. `curve_bundle_to_artifact` adds diagnostics,
  stress, and typed quantity metadata for diagnostic/optimizer surfaces.
- `mmm.simulation.engine.simulate_curve_diagnostic` is explicitly diagnostic
  curve interpolation, not the canonical full-panel Δμ path. It defaults to
  endpoint clamping and labels research-only linear extrapolation separately.
- `MMMResponseCurveArtifact` is only a generic export family with channel names
  and a note that curves explain while full-panel simulation decides. The
  adapter preserves raw curve data in `source_payload` and blocks claims.
- Curve grid checks, monotonicity diagnostics, and stress reports are useful
  internal evidence, but are not a versioned public response-surface contract.

## Supported range, uncertainty, and typed failures

Supported range is the immediate gap. Curve code can infer a numeric grid
minimum/maximum and flag out-of-grid spend, while the full-panel simulator does
not emit a common public range object. Existing failure packets can carry
`supported_range_evidence`, and the golden suite includes an
`UNSUPPORTED_EXTRAPOLATION` failure fixture, but neither identifies a positive
range artifact or the provenance that made the range valid.

R11 has point and conditional posterior summaries; R12 has optional diagnostic
bootstrap/interpolation outputs. Neither is a public uncertainty contract with
scope, model identity, diagnostics/limitations linkage, range status, or
deterministic producer fixtures. Existing typed failures, manifests, and
diagnostics can reference the future range artifact; they do not replace it.

## Dependency models evaluated

| Model | Finding | Decision |
|---|---|---|
| A — R12 before R11 | A reusable curve surface is not required for the internal canonical full-panel Δμ computation. | Rejected as the mandatory order. |
| B — R11 before R12 | Full-panel computation can produce outcomes without exporting curves, but lacks the same governed range evidence required to make plan results safe. | Rejected as the next task. |
| C — narrow shared prerequisite first | Both public paths need a positive supported-range definition and extrapolation provenance. Negative failure evidence and curve grids are insufficient. | **Selected.** |
| D — parallel independent work | The common range gap would duplicate semantics and risk divergent extrapolation handling. | Rejected. |

## Selected next task

| Field | Selection |
|---|---|
| Task ID | `MMM_MIP_HANDOFF_V1_SUPPORTED_RANGE_EVIDENCE_001` |
| Title | Define MMM supported-range evidence |
| Requirement / prerequisite | Shared R11/R12 producer prerequisite |
| Dependency model | Model C — narrow shared prerequisite first |
| Why now | It is smaller than either export, already has failure/diagnostic vocabulary, and prevents full-panel and curve paths from declaring incompatible ranges. |
| Prerequisites | `30d4543`, `8d73e0c`, `7f615fb`, `949d7a6`, `fdc69f9`, `c96848e`; current failure, manifest, calibration, diagnostics, golden, and compatibility-policy artifacts. |
| Expected public evidence | A versioned supported-range record with model/config/dataset and channel/KPI/geo/time scope, validated numeric domain, provenance, restrictions, diagnostics/limitations references, and safe artifact reference. |
| Expected boundary integration | Future R11/R12 exports and known extrapolation failures can reference the record; no consumer parser or recommendation policy. |
| Expected fixtures / tests | Deterministic in-range, restricted, and blocked examples; range identity, scope, serialization, failure-reference, diagnostics linkage, and cross-artifact consistency tests. |
| Explicit non-goals | No simulation export, response-surface export, plan evaluation, optimization, recommendation, MIP parsing, TrustReport, model-math change, or interface freeze. |
| Still blocked | R11 and R12 remain partial; R16 remains blocked; all authorization flags remain false. |

## Deferred alternatives

1. `MMM_MIP_HANDOFF_V1_PUBLIC_SIMULATION_EXPORT_001` — deferred because its
   candidate/baseline outcome must prove a supported operating range before it
   can safely serialize a positive result.
2. `MMM_MIP_HANDOFF_V1_RESPONSE_SURFACE_EVIDENCE_CONTRACT_001` — deferred
   because a grid alone is not a governed range, and its range semantics should
   not diverge from future full-panel simulation.
3. `MMM_MIP_HANDOFF_V1_CANDIDATE_PLAN_CONTRACT_001` — deferred because
   `PlanningScenario` is useful internal evidence but plan identity is R11
   specific; it is not the smaller shared R11/R12 dependency.

## Status and authorization boundary

R6, R7, R9, R10, calibration-treatment lineage, R13, and R15 remain
implemented. R11 and R12 remain partial. R16 MIP consumer readiness remains
blocked. Interface freeze remains unauthorized. The following remain false:
MIP consumer readiness, producer interface freeze, recommendation, budget
optimization, candidate-plan simulation, response-surface production, Bayesian
production, LLM decisioning, and live scheduling authorization.
