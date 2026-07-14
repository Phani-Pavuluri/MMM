# MMMExportBundle ingestion and answerability

**Investigation:** INV-MIP-EXPORT-001

**Status:** implemented

**Boundary:** MIP-side parsing and answer safety; no MMM computation

## What MIP consumes

MIP treats `MMMExportBundle` as an artifact produced by an external system. The
consumer parser does not import the MMM producer schema or run MMM code. It
normalizes only the envelope and artifact fields needed to decide answerability:

- identity: `schema_version`, `bundle_id`, and `model_run_id`
- artifact inventory and each `artifact_type`
- explicit switches: `llm_exposure_allowed`, `demo_fixture_allowed`,
  `recommendation_allowed`, `planning_allowed`, and `production_claim_allowed`
- claim policy: `allowed_claims` and `forbidden_claims`
- gates: `diagnostic_status`, `promotion_status`, `uncertainty_status`, and
  `artifact_safety_status`
- provenance: `lineage`, including compatible top-level producer fingerprint fields
- recommendation-contract references and proposed shifts, when present

The parser preserves raw payloads for downstream inspection but never derives
ROI, contribution, curves, simulation output, or recommendations from them.

## Answerability rules

All required permission must be explicit at both bundle and artifact level.
Missing, malformed, or unknown safety fields normalize to a blocked value.
`forbidden_claims` always overrides `allowed_claims`.

| Intent | Minimum answer rule |
|---|---|
| `mmm_readiness` | A model-fit or diagnostic artifact must explicitly permit readiness explanation and be readiness/diagnostic safe. |
| `model_diagnostics` | A diagnostic artifact must explicitly permit diagnostic explanation. The explanation cannot be extended into ROI or advice. |
| `channel_contribution` | Requires explicit contribution claim permission plus production-safe, promotion, uncertainty, and LLM gates. |
| `channel_roi` | ROI/ROAS is blocked by default. Production use requires explicit claim permission plus every production gate. |
| `response_curve` | Explanatory planning use can be allowed explicitly, but it cannot imply an optimal allocation or budget shift. |
| `simulation_result` | Explanatory planning use can be allowed explicitly, but simulation output does not carry recommendation authority. |
| `budget_recommendation` | Requires an actual `MMMRecommendationContract`, explicit recommendation and planning permission, production gates, optimizer lineage, TrustReport references, and proposed shifts. |

An optimizer artifact that says it has a contract is not itself a contract. A
contract-shaped artifact with `recommendation_allowed=false` also remains blocked.

## Cannot-say behavior

The classifier returns a structured result containing `allowed`, scope, a stable
reason code, verifier-facing reasons, disclosures, and a `Cannot say:` message for
blocked results. Callers should surface that reason instead of filling gaps from
model knowledge or inferring permission from artifact contents.

## Demo fixture behavior

Synthetic ROI fixture values may be explained only when both the bundle and ROI
artifact explicitly set `demo_fixture_allowed=true` and allow
`demo_fixture_only`. Every such answer must label values as synthetic/demo, state
that they are not production or business truth, and avoid channel ranking or
budget advice. Demo permission never becomes production permission.

## Why ROI and recommendations remain blocked by default

The presence of a numeric field proves neither governance nor fitness for a
claim. ROI needs approved semantics, uncertainty, promotion, and claim exposure;
recommendations additionally need decision lineage and TrustReport-backed
authority. Fail-closed defaults prevent a partial export, optimizer output, or
demo fixture from being presented as decision-grade truth.

## Handoff and next step

This work consumes the schema and fixtures established by MMM-EXPORT-002 and the
claim-gated runtime handoff from MMM-EXPORT-003. It adds no producer behavior and
does not alter GeoX, DecisionSurface, TrustReport, fitting, simulation, or
optimization logic.

The next step is to connect the classifier result to the MIP prompt/verifier
boundary so every MMM-backed response records its bundle ID, artifact selection,
intent, and answerability decision in the response trace.
