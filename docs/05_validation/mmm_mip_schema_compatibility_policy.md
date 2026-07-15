# MMM–MIP producer schema compatibility policy

**Policy ID:** `MMM_MIP_HANDOFF_V1_SCHEMA_COMPATIBILITY_POLICY_001`
**Policy schema version:** `mmm_mip_schema_compatibility_policy_v1`
**Status:** implemented producer policy; runtime enforcement remains exactly as currently implemented
**Evidence base:** `fdc69f9` and the public contracts and `golden_v1` fixtures registered in the companion [machine-readable registry](archives/MMM_MIP_HANDOFF_V1_SCHEMA_COMPATIBILITY_POLICY_001_registry.json).

## Scope and ownership

This is an MMM-owned declaration of compatibility for producer artifacts. It covers the public export bundle, typed failure packet and outcome, run manifest and manifest outcome, artifact reference, calibration-treatment lineage, diagnostics/limitations aggregate, supported-range evidence, and the deterministic `golden_v1` fixture set.

MMM owns producer schema identities, versions, compatibility classification, deprecation declarations, fixture-version declarations, and producer-side evidence. MIP owns consumer parsing implementation, parser configuration, user-facing compatibility errors, intent routing, conversational handling, TrustReport assembly, and recommendation or optimization authorization. This policy does not prescribe MIP UI or conversational behavior.

The registry is normative for the supported versions and the observed parser matrix. It is intentionally a policy artifact, not a new runtime API or a schema-negotiation implementation.

## Supported contracts and observed behavior

Only the versions recorded in the registry are supported. No historical MMM–MIP contract version is claimed merely because a similarly named internal artifact exists.

| Surface | Supported current version | Extra/unknown field behavior | Missing field behavior |
|---|---|---|---|
| `MMMExportBundle` and export artifact family | `mmm_mip_export_v1` | Bundle and artifact base currently preserve extra fields; strict nested safety blocks reject them. | Required identity fields reject; optional fields retain absent/default semantics; `schema_version` defaults and only checks non-empty text. |
| `MMMFailurePacket` / `MMMExportOutcome` | `mmm_mip_failure_v1` for the packet; outcome is a composition | Strictly rejects unknown fields. | Required fields reject; packet schema version defaults when absent and rejects unsupported supplied values. |
| `MMMRunManifest` / `MMMExportManifestOutcome` / `MMMArtifactReference` | `mmm_mip_run_manifest_v1` for the manifest; compositions/reference have no independent version | Strictly rejects unknown fields. | Required fields reject; manifest schema version defaults when absent and rejects unsupported supplied values. |
| `MMMCalibrationTreatmentLineage` | `mmm_calibration_treatment_lineage_v1` | Strictly rejects unknown fields. | Required fields reject; schema version defaults when absent and rejects unsupported supplied values. |
| `MMMDiagnosticsLimitations` | `mmm_diagnostics_limitations_v1` | Strictly rejects unknown fields. | Required fields reject; schema version defaults when absent and rejects unsupported supplied values. |
| `MMMSupportedRangeEvidence` | `mmm_supported_range_evidence_v1` | Strictly rejects unknown fields. | Required identity and records reject; absent optional linkage never means unrestricted support. |
| `golden_v1` fixture-set index | `mmm_producer_golden_fixture_set_v1` | There is no production parser; the fixture test asserts this exact version. | Missing index/version is invalid fixture evidence and fails fixture validation. |

The differences above are intentional observations, not a claim that all contracts have uniform runtime enforcement. In particular, this policy must not be read as claiming that an absent schema-version field is detected at runtime where a Pydantic default currently supplies it, or that `MMMExportBundle` runtime parsing rejects an arbitrary non-empty version. A future enforcement task would need separate authorization; this task does not change those paths.

## Compatibility classifications and versioning

The registry uses these stable classifications:

- `BACKWARD_COMPATIBLE`: evidence shows an unchanged supported consumer can retain the prior meaning.
- `CONDITIONALLY_COMPATIBLE`: only compatible under the per-contract parser and semantic restrictions recorded in the registry.
- `BREAKING`: requires a new version, evidence, fixtures, and a compatibility review.
- `PROHIBITED_UNTIL_AUTHORIZED`: requires separately authorized governance and is not authorized by this policy.

An optional metadata field may be conditionally compatible only for the export models that currently preserve unknown fields, and only when it has absent semantics, has no effect on claim safety, promotion, terminal state, calibration treatment, or authorization, and is documented in the registry. For strict public contracts, adding even an optional serialized field is breaking for the current parser because it rejects extras. A required field, field removal or rename, type/nullability/unit/identifier/timestamp change, semantic or estimand change, narrowing a value domain, terminal-state change, or cross-artifact-reference change is breaking. Reusing a version identifier for incompatible semantics is prohibited.

Documentation-only clarification does not change a serialized version when it does not alter meaning. Any serialized semantic change needs a new schema or contract version as recorded by the affected contract. Multiple versions may coexist only with an explicit support entry, fixtures where applicable, and evidence; a version is never silently migrated or negotiated by MMM here.

Compatibility does not mean semantic equivalence for every consumer use, and schema support does not mean model promotion.

## Unknown, missing, and enum values

The observed unknown-field matrix is in the registry and is deliberately more specific than a blanket "forward compatible" claim. Unknown top-level and artifact-base fields can be preserved by the export schema today; they cannot silently authorize a claim, change promotion or terminal state, alter a failure meaning, change calibration treatment, remove a limitation, or authorize a recommendation or optimization. Unknown nested fields in strict typed blocks are rejected. The policy classifies unknown required semantic fields as unsupported and fail-closed even where the current export parser cannot infer their semantic importance from an arbitrary extra key.

Missing required identity, version, terminal-status, and required cross-artifact-reference evidence is fail-closed by policy. Missing optional fields retain absent semantics: they do not fabricate promotion evidence, authorization, uncertainty, or an absence of limitations. The runtime-default exceptions in the table are recorded evidence gaps, not permission to treat an omitted version as verified support.

Enum expansion is only informational or conditionally compatible when the new value cannot affect a safety decision and the affected contract has explicit support. Unknown terminal status, failure code, promotion status, claim disposition, calibration-treatment status, and authorization-related values are unsupported/fail-closed. The present typed Pydantic enum contracts reject an unknown supplied value.

## Fixture-set support

`golden_v1` is the sole supported fixture-set version. Its eight scenario IDs, component references, terminal outcomes, failure codes, and research-only boundary are deterministic fixture semantics. Adding a scenario is additive only if every existing scenario retains its meaning. Changing an existing scenario's semantics requires a new fixture-set version; removing a required scenario is breaking for fixture consumers. Correcting an invalid fixture must be documented, retested, and versioned when it changes serialized meaning.

Supported contracts represented in `golden_v1` must retain deterministic fixture evidence. Fixture support enables regression testing only; it neither freezes the interface nor authorizes MIP readiness.

## Deprecation and removal

The lifecycle is `ACTIVE` → `DEPRECATED` → `UNSUPPORTED` → `REMOVED`. All currently registered versions are `ACTIVE`; this policy does not deprecate any of them. A future deprecation notice must name the contract or fixture version, replacement (or explicitly record its absence), reason, migration guidance, support status, affected fixtures/tests, and an evidence-based earliest removal condition. There is no invented calendar deadline because the repository has no governed release cadence for this boundary.

Removal additionally requires explicit prior deprecation, a supported replacement and golden evidence, producer validation and regression coverage, consumer migration evidence, no unresolved consumer dependency, roadmap/audit updates, and separate removal authorization. A replacement alone never authorizes removal.

## Fail-closed safety boundary

MMM classifies the following as fail-closed or unsupported: unsupported required schema versions; missing version evidence; unknown terminal status or safety-critical enum value; unresolved required artifact references; inconsistent run IDs; contradictory success/failure terminal state; and a research-only artifact presented as production-supported. This is a producer compatibility boundary, not conversational error handling. MIP decides how to present incompatibility to users.

Failure packets do not authorize recommendations, optimization, candidate-plan simulation, response-surface production, TrustReport assembly, user-intent classification, or model promotion. Bayesian MMM remains research-only.

## Status and examples

R15 is implemented as documented, deterministic compatibility policy evidence. It does **not** freeze the MMM–MIP interface: current versions remain subject to governed additive evolution, and breaking changes require a new version and evidence. R11 and R12 remain partial; R16 MIP consumer readiness remains blocked. Interface freeze requires a separate authorization artifact.

Examples:

- Adding a `MMMExportBundle` optional annotation that has no safety semantics is conditionally compatible only under the export-model rules above.
- Adding an optional field to `MMMFailurePacket` is breaking today because its parser rejects unknown fields.
- Renaming `run_id`, changing a failure-code meaning, or changing a fixture's terminal outcome is breaking and requires a new applicable version.
- Removing `golden_v1`, silently dropping a deprecated field, weakening unsupported-version handling, or changing producer/MIP ownership through a schema update is prohibited until separately authorized.

## Follow-up — post-compatibility gap selection

The policy evidence audit is complete. The next narrow producer prerequisite is
`MMM_MIP_HANDOFF_V1_SUPPORTED_RANGE_EVIDENCE_001`, selected because R11 public
simulation export and R12 response-surface evidence both need a positive,
versioned supported-range record. This does not change runtime compatibility
behavior, authorize an interface freeze, or unblock R16 MIP consumer readiness.

## Follow-up — supported-range evidence

`MMM_MIP_HANDOFF_V1_SUPPORTED_RANGE_EVIDENCE_001` adds the active, strict
`mmm_supported_range_evidence_v1` producer contract and a separate additive
`supported_range_v1` fixture collection; it leaves `golden_v1` semantics
unchanged. The contract is additive to the current manifest/export composition
through an optional stable reference, but future serialized changes need a new
compatibility review. R11/R12 remain partial, R16 remains blocked, and interface
freeze remains unauthorized.
