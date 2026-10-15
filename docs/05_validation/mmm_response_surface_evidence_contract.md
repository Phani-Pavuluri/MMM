# MMM diagnostic response-surface evidence contract (PROPOSED)

> **Status:** PROPOSED definition only. Not an accepted interface, not an
> authorized implementation. No execution, merge, production, or analytical
> authority follows from this document. Contract adoption, schema/parser/
> fixture work, and any analytical implementation require separately
> authorized successors.
>
> **Task:** `MMM_MIP_HANDOFF_V1_RESPONSE_SURFACE_EVIDENCE_CONTRACT_001`
> (documentation preparation only).
> **Verified baseline:** MMM `origin/main = a9db6469ed2654e19c20464c28717767990740ce`.
> All merged-source claims below were verified against this baseline during
> publication; historical pins from the advisory draft are retained as
> provenance where noted.
> **R11 status correction:** R11 public simulation export is implemented
> (`MMMPublicSimulationExport`, `simulation_v1` golden fixtures); R12 remains
> partial and unauthorized. Stale advisory present-tense claims to the
> contrary are corrected here.
> **Provenance:** Muse-authored advisory draft revision 2 (OpenCode session
> `ses_ebf149b3cffeLg1zkK5LsjuDsh`, message `msg_140ed513c001QGW21yBEQtEKlQ`),
> independently reviewed by Codex before publication. Earlier source-date and
> active-task descriptions in that draft (including references to
> `MMM_LIVE_STATUS_REFRESH_001` as the then-current task) are historical; that
> task has since merged and closed.
>
> **No invented numbers.** All numeric values below are schematic shape
> illustrations, never evidence.

Owner-facing recommendation: describe how each channel responds as spend
changes, with clear limits and traceable evidence. These curves explain the
model; full-panel simulation remains the basis for budget decisions. Recommend
per-channel curves first; a joint multi-channel surface would be separate scope.

## 1. Existing vs proposed names

**EXISTING (merged, reuse verbatim — do not rename):** `MMMSupportedRangeEvidence` / `MMMSupportedRangeRecord` / `MMMRangeBound` / `MMMRangeScope` / `MMMRangeScale` (`RAW`/`TRANSFORMED`) / `MMMRangeEvidenceBasis` / `MMMRangeAvailabilityStatus` / `MMMRangeRelation` / `MMMExtrapolationClassification` / `MMMRangeRestriction` (`mmm/contracts/supported_range.py`, versions `mmm_supported_range_evidence_v1` / `mmm_supported_range_record_v1`); `ResponseCurve` (`spend_grid`, `response`, `marginal_roi`), `build_curve_for_channel` (strict), `build_curve_for_channel_research_only`, `validate_response_curve_diagnostics`, `curve_bundle_to_artifact` (`mmm/decomposition/curves.py`, `curve_bundle.py`); `CurveQuantityResult` (`mmm_quantity_envelope_v1`, `non_canonical_builder: "none" | "research_only_sparse_or_relaxed"`, `spend_grid` / `response_on_modeling_scale` / `marginal_roi_modeling_scale`, safety invariants in `mmm/contracts/quantity_models.py`); `MMMFailureCode` / `MMMFailureStage` / `MMMRetryDisposition` (`mmm_mip_failure_v1`); `MMMDiagnosticsLimitations` / `MMMTechnicalClaim` / `MMMTechnicalClaimDisposition`; `MMMSimulationStatus`, `MMMSimulationUncertaintyStatus` (`UNAVAILABLE`/`AVAILABLE`/`RESEARCH_ONLY`), `MMMSimulationScope`, `MMMSimulationPlan` (`mmm_public_simulation_export_v1`); compatibility policy `mmm_mip_schema_compatibility_policy_v1`; canonical stacks in `mmm/contracts/canonical_transforms.py` (`ridge_bo`/`bayesian` × `semi_log`/`log_log` × `geometric`/`weibull` × `hill`/`log`/`logistic`, with only `geometric`+`hill` combos listed in `CANONICAL_MEDIA_STACKS`).

**PROPOSED (new, marked `PROPOSED` everywhere):** `MMMResponseSurfaceEvidence` (`PROPOSED`, `mmm_response_surface_evidence_v1`) and `MMMResponseSurfaceRecord` (`PROPOSED`, `mmm_response_surface_record_v1`). Every other name in this draft is an existing enum/field reused by reference.

**Pointwise-relation correction:** existing `MMMSupportedRangeRecord.range_relation` and `extrapolation_classification` are **single per-record labels** on the range record itself (with validators, e.g. `BOUNDARY` requires a boundary relation; `LIMITED_EXTRAPOLATION` requires `governed_extrapolation_evidence`; `UNSUPPORTED_EXTRAPOLATION` requires a `BLOCKED`/`RESTRICTED` claim effect). They are **not** per-grid-point evaluations of a curve. The surface contract therefore needs explicitly **proposed** per-point fields (§5): `per_point_relation: list[MMMRangeRelation]` (proposed) and `per_point_classification: list[MMMExtrapolationClassification]` (proposed), each length-matched to the grid, derived only by comparing grid points against the referenced supported bounds on a comparable scale (§3). The record-level `export_relation_summary` (proposed) is a roll-up (e.g. worst-case), never a substitute for the referenced range record's own labels.

**Uncertainty-availability correction:** do not assume "always unavailable." Merged evidence already models availability-by-reference: `MMMSupportedRangeRecord` carries `uncertainty_available: bool` with invariant "available requires `uncertainty_artifact_reference`; unavailable cannot claim artifact or semantics"; `MMMSimulationUncertainty` carries `status` defaulting `UNAVAILABLE` but permitting `AVAILABLE` only with `artifact_reference`. The surface contract follows the same pattern (§6): availability is **resolved from referenced governed artifacts**, defaulting to unavailable only when no such artifact is referenced — it is not axiomatically unavailable.

## 2. Identity, ownership, non-goals (existing policy, no fresh choice needed)

- **Proposed kind/version:** `artifact_kind = "MMMResponseSurfaceEvidence"` (proposed literal), `artifact_schema_version = "mmm_response_surface_evidence_v1"` (proposed). Strict `extra="forbid"` models throughout, per existing strict-contract precedent. Serialized semantic change requires a new version; version reuse for incompatible semantics prohibited (existing compatibility policy).
- **Ownership:** MMM-owned producer technical evidence (producer identity/version/compatibility only). MIP owns parsing/display/TrustReport/recommendation policy. GeoX unaffected. Follows existing ownership split; not a new decision.
- **Preamble non-identity (required):** diagnostic evidence only — not a simulation, contribution/ROI artifact, recommendation, TrustReport, production authorization, or interface freeze. R11 is implemented; R12 remains partial and unauthorized; R16 remains blocked (per `platform_roadmap.md` as corrected by `MMM_LIVE_STATUS_REFRESH_001`). Follows existing roadmap/audit boundary; not a new decision.
- **Strictness posture:** strict unknown-field rejection for this surface (same class as `MMMFailurePacket`/`MMMSupportedRangeEvidence`, not the permissive export-bundle base). This follows existing strict-contract policy for safety-critical typed blocks; recorded here as applying existing policy, not as an owner question.

## 3. Coordinate-scale reconciliation

Reading `_response_curve_core` (`mmm/decomposition/curves.py`): for each grid value `s`, the builder forms a **constant series at level `s`** (`np.full(...)`), applies **adstock then saturation internally** (`apply_adstock_saturation_series(series, ad, sat)[-1]`), optionally `safe_log` for `log_log`, then scales by `beta`. Therefore:

- **`spend_grid` is the RAW spend axis** (e.g. spend per week in `spend_unit`, finite, non-negative, strictly increasing, n≥5 canonical).
- **`response` is on modeling scale** (`β·x`, where `x` is the transformed steady-state value; `log-mean contribution` per `curve_bundle_to_artifact`'s `marginal_roi_definition`). It is not level-space revenue.
- **`transformation_id` does not label the axis.** It is an opaque reference identifying the **internal pipeline** (adstock+saturation+`model_form` configuration: fitted `decay`, `hill_half`/`hill_slope`, `beta`, `horizon_weeks`, config/stack pointer). **No canonical `transformation_id` string exists in merged code** — `transformation_id` appears only as an opaque text field with RAW/TRANSFORMED consistency rules (`supported_range.py`, `public_simulation.py`) and test placeholders (`"log1p"`, `"sqrt"` in `tests/contracts/test_mmm_public_simulation.py`). The definition task must specify the identifier scheme (recommended: pointer to transform policy + configuration hash + stack names from `canonical_transforms.py`); the example below uses `"<transform-config-ref>"` as an unresolved-scheme placeholder, never a resolvable identifier.
- **No double-transform:** consumers must plot/use the grid as RAW spend; the response values already embody the transform. Re-applying adstock/saturation to the grid is an error the contract forbids in its transform-semantics note.
- **Range-comparison rule:** a grid point may be range-checked only against referenced bounds with **matching `(unit, scale, transformation_id)`** (same invariant as `_pair_ok` and the aggregate no-mixed-scale validator). Curve axis is RAW, so a referenced bound with `scale=TRANSFORMED` is **not comparable** in v1 (no governed RAW↔transformed mapping exists) → the record is `BLOCKED` under `SUPPORTED_RANGE_EVIDENCE_UNUSABLE` (§4), not extrapolated. This is a scale-compatibility failure, not a numeric position.

## 4. Scope mismatch vs extrapolation — separate typed failures

Dimensional/scope incompatibility is **not** numeric extrapolation. Following `public_simulation.py` precedent (`_resolve_range_record` matching on channel/KPI/model/scope/grain/unit/scale, then `_blocked(... SUPPORTED_RANGE_EVIDENCE_UNUSABLE ...)` vs a separate `UNSUPPORTED_EXTRAPOLATION` path):

- **`SUPPORTED_RANGE_EVIDENCE_UNUSABLE`** (stage `EXPORT`, retry typically `RETRY_AFTER_INPUT_CHANGE` or `RETRY_AFTER_GOVERNANCE_CHANGE`): no referenced record matches the curve's (channel, KPI, model, geography, segment, grain, unit, scale, `transformation_id`); referenced evidence missing, version-unknown, scope-ambiguous, or scale-incomparable (§3). No success output. Deterministic fixture: `blocked_range_unusable.json`.
- **`UNSUPPORTED_EXTRAPOLATION`** (stage `EXPORT`, requires `BLOCKED`/`RESTRICTED` claim effect via `MMMRangeRestriction`, mirroring the range-record validator): record matches dimensionally, but one or more grid points fall outside supported bounds on a comparable scale (see `per_point_*` in §5). Terminal `BLOCKED`, no success output. Fixture: `blocked_unsupported_extrapolation.json`.
- **`INVALID_PLAN_INPUT`**: malformed grid itself (non-finite, non-increasing, sparse `n<5`, negative spend, `RAW`+`transformation_id` contradiction). Terminal `FAILED` (request validation failed before evaluation), no success output. Fixture: `failed_invalid_grid.json`.

## 5. Per-record shape (proposed fields; existing types reused)

`MMMResponseSurfaceRecord` (PROPOSED), one per (channel × fixed operating context):

- `channel_id: str` (safe text; existing `_text`/`scope_text` rules: non-empty, no leading `/`/`~`, no secret/traceback markers).
- `spend_grid: list[float]` RAW axis (§3); `spend_unit: str`; `spend_scale: Literal["RAW"]` fixed in v1 (a future transformed-axis variant, if ever needed, requires a new schema version).
- `transformation_ref: str` (PROPOSED name; opaque internal-pipeline reference, scheme TBD at adoption — see §3; schematic placeholder `"<transform-config-ref>"`).
- `model_form: str` (`semi_log`/`log_log` per canonical stacks); `horizon_weeks: int`; `aggregation: Literal["steady_state_last_week"]` (extensions need new version); `evaluation_start`/`evaluation_end` timezone-aware, `end > start` (existing `MMMSimulationScope` rule).
- `scope` (channel/KPI/geography/segment/time-window/outcome-or-estimand/grain subset mirroring `MMMRangeScope`); `supported_range_evidence_id: str` + `supported_range_record_ids: list[str]` (existing IDs referenced, never restated).
- `per_point_relation: list[MMMRangeRelation]` (PROPOSED, length == grid) and `per_point_classification: list[MMMExtrapolationClassification]` (PROPOSED, length == grid), comparable-scale comparison only; `export_relation_summary` (PROPOSED roll-up).
- `response_on_modeling_scale: list[float]`, `marginal_roi_modeling_scale: list[float]` (length == grid; see §6 for units/claims).
- `non_canonical_builder: Literal["none"]` fixed — research-only relaxed builds (`"research_only_sparse_or_relaxed"`) are excluded from this export.
- Bundle-level (PROPOSED `MMMResponseSurfaceEvidence`): `surface_id`, `model_run_id`/`model_id`/`model_family`/`model_version`/`configuration_hash`/`dataset_fingerprint`/`package_version`/`git_commit`/`generated_at`/`source_artifacts` (pointers to `curve_bundles`, transform policy, fit/diagnostics — pointers only), `MMMRunManifest` + `MMMExportManifestOutcome` linkage every terminal state, run-ID match enforced, `diagnostics_limitations_id` reference.

## 6. Derivative units and allowed claims

`marginal_roi_modeling_scale` is `np.gradient(response, spend_grid)` = **d(modeling-scale response)/d(RAW spend)**, units: *modeling-response-units per spend-unit* (e.g. log-mean contribution per currency-unit/week). It is a diagnostic slope for shape/saturation discussion, carried inside `CurveQuantityResult` whose merged safety invariants are: `tier=DIAGNOSTIC`, `surface=DIAGNOSTIC`, `semantics=APPROX_CURVE`, `approximate=True`, `decision_safe=False`, `prod_safe=False`, with a non-empty `unsupported_for` list blocking downstream misuse (verified invariant: approximate quantities must declare `unsupported_for` and cannot pair with `decision_safe=True`).

- **Allowed claims (diagnostic only):** curve shape / saturation / diminishing-returns discussion, steady-state diagnostic comparison across grids, linkage to `ResponseDiagnostics`/`CurveStressReport`/`curve_decision_alignment` evidence. The ban covers **level-space ROI/ROAS/contribution and allocation decisions**, which are **excluded** from this contract. Naming overlap ("marginal ROI") does not import ROI-artifact semantics.
- **Forbidden claims (existing inventory §6.5 + quantity safety):** budget allocation from curve increments, replacing DecisionSurface/full-panel Δμ, "highest ROI channel" narratives, causal lift or profit forecasts, and any level-space economics attachment (excluded from v1 scope).
- **Uncertainty:** `uncertainty: {status, artifact_reference?, semantics?}` mirroring `MMMSimulationUncertainty` + range-record `uncertainty_available` invariants: `AVAILABLE` only with a referenced governed artifact; otherwise `UNAVAILABLE` (expected for v1, since only diagnostic bootstrap/interpolation outputs exist — recorded as observed absence, not axiom); `RESEARCH_ONLY` draws stay research-only. No values invented.

## 7. Recommendation retained: univariate bundle over joint surface

Recommend the univariate diagnostic curve bundle (§5): it standardizes the already-merged per-channel `curve_bundle_to_artifact` shape (channel, horizon, aggregation, `model_form`, grid, modeling-scale response/marginal-ROI, typed `CurveQuantityResult`, diagnostics, stress, marginal-ROI definition) with identity/version/scope/lineage/range/failure wrappers, without changing its math. Reject a joint multi-channel surface for R12: no governed conjunctive range semantics exist (range records are per-channel/context), the canonical decision path is full-panel Δμ (`decision_simulate.simulate` + `MMMPublicSimulationExport`), and the gap-selection audit already rejected divergent parallel range semantics (Model D). A joint surface, if ever needed, is a separate versioned contract. Governing sentence retained: "**Curves explain; full-panel simulation decides**" (`docs/02_concepts/response_curves.md`).

## 8. Deterministic acceptance examples (schematic shapes; identifiers marked TBD/schematic)

- **A — valid record (`SUCCEEDED`).** Strict parse accepts: proposed kind/version, `surface_id`, lineage + manifest linkage with matching `run_id`; one record with RAW strictly-increasing finite grid (n≥5), `spend_scale="RAW"`, `transformation_ref="<transform-config-ref>"` (scheme TBD), modeling-scale vectors length-matched, `non_canonical_builder="none"`, resolvable range-record IDs with comparable `(unit, scale)` bounds, `per_point_*` length-matched with all points within range, diagnostics/limitations reference resolvable, uncertainty `UNAVAILABLE` with no artifact claim. Fixture: `succeeded_single_channel.json`; plus `succeeded_multi_channel_bundle.json` (records non-interchangeable, never pooled) and `succeeded_with_limitations.json` (a `MMMRangeRestriction` with code/summary/scope/evidence-references/claim-effects).
- **B — numeric outside-supported (`BLOCKED`, `UNSUPPORTED_EXTRAPOLATION`).** Dimensionally matched record, ≥1 grid point outside supported bounds on comparable scale → terminal `BLOCKED`, typed failure evidence, no success output.
- **C — dimensional/scale mismatch (`BLOCKED`, `SUPPORTED_RANGE_EVIDENCE_UNUSABLE`).** No matching record, scope-ambiguous, version-unknown, or TRANSFORMED-bounds-vs-RAW-grid incomparability → terminal `BLOCKED`, no success output. (Corrected per §4.)
- **D — malformed grid (`FAILED`, `INVALID_PLAN_INPUT`)** plus strictness rejections: non-finite/non-increasing/sparse grid, `RAW`+transform contradiction, unknown schema version, extra field, relaxed-builder provenance, Bayesian-as-production (research-only boundary), run-ID mismatch, unknown safety-critical enum, contradictory success/failure state. Each maps to the narrowest existing failure code with parse rejection.
- **Gate:** additive `tests/fixtures/mip_export/response_surface_v1/` collection (A–D set); contract tests assert identity/version, scope/record resolution, RAW/scale invariants, per-point length/roll-up consistency, lineage/manifest linkage, failure codes, strict-parser behavior, and `golden_v1` unchanged. Fixture/schema implementation itself is a successor task, not this proposal.

## 9. Reserved decisions (only genuine adoption/implementation choices)

With definition preparation already authorized, the following are **documented here with recommendations** and reserved for **contract adoption/implementation authorization**, not for publishing this proposal:

1. Adopt the univariate-bundle dimensionality (§7) vs a future joint surface (recommend bundle).
2. Adopt terminal vocabulary reuse (`SUCCEEDED` / `SUCCEEDED_WITH_LIMITATIONS` / `BLOCKED` / `FAILED`, recommend reuse for R11 consistency).
3. Adopt `spend_scale="RAW"`-only v1 with TRANSFORMED-bounds incomparability → `BLOCKED` (recommend; follows merged builder + no-mixed-scale policy).
4. Adopt the `transformation_ref` identifier scheme (recommend config-hash + stack-name pointer — no invented canonical string).
5. Adopt uncertainty-by-reference posture (recommend §6 pattern).
6. Adopt strict-parser row registration under the compatibility policy (recommend strict; follows existing strict-contract class).
7. Exclude level-space ROI/mROAS/economics attachment from v1 (recommend exclusion; inclusion would import ROI/recommendation semantics needing separate authorization).

Non-reserved (already settled): Model C prerequisite order, curves-don't-decide, observed/support/validated distinction with no invented support, strict-parser/fail-closed behavior, `golden_v1` immutability, R12-partial/R16-blocked/no-freeze/no-recommendation boundaries, and this proposal's non-interference with merged tasks.
