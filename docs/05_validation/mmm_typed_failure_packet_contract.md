# MMM typed failure packet contract

**Contract version:** `mmm_mip_failure_v1`
**Owner:** MMM producer boundary
**Status:** implemented; producer interface not frozen

## Purpose

`MMMFailurePacket` is the versioned technical failure contract emitted by the
MMM producer boundary. It lets a downstream platform consume governed failure
facts without inspecting Python exceptions, logs, estimator objects, traces,
DataFrames, or arbitrary internal dictionaries.

`MMMExportOutcome` contains exactly one of a successful `MMMExportBundle` or an
`MMMFailurePacket`. The existing successful adapter remains available; the
outcome wrapper is additive.

## Contract fields

Each packet contains a schema version, stable failure ID, timezone-aware
creation time, optional package/git identity, optional run/model/configuration/
dataset linkage, a typed code and lifecycle stage, source component, concise
technical summary, retry disposition, typed remediation actions, blockers, and
optional validation, diagnostic, calibration, governance, scope, and range
evidence. Pre-run failures intentionally leave run and model references absent.

`technical_context` is limited to finite JSON values. It rejects exception
objects, stack traces, model objects, DataFrames, filesystem paths, and keys
associated with secrets or raw internal payloads.

## Failure taxonomy and deterministic policy

| Code | Default retry disposition | Required remediation category |
|---|---|---|
| `INSUFFICIENT_HISTORY` | input change | `INPUT_DATA` |
| `INCOMPATIBLE_GRAIN` | configuration change | `CONFIGURATION` |
| `KPI_NOT_SUPPORTED` | model or method change | `MODEL_OR_METHOD` |
| `SPEND_VARIATION_INSUFFICIENT` | input change | `INPUT_DATA` |
| `CONTROL_DATA_MISSING` | input change | `INPUT_DATA` |
| `CALIBRATION_SCOPE_MISMATCH` | calibration change | `CALIBRATION` |
| `CALIBRATION_SIGNAL_EXPIRED` | calibration change | `CALIBRATION` |
| `MODEL_INSTABILITY` | model or method change | `MODEL_OR_METHOD` |
| `HOLDOUT_FAILURE` | configuration change | `CONFIGURATION` |
| `UNSUPPORTED_EXTRAPOLATION` | candidate-plan change | `CANDIDATE_PLAN` |
| `IDENTIFIABILITY_FAILURE` | model or method change | `MODEL_OR_METHOD` |
| `MODEL_NOT_PROMOTED` | governance change | `GOVERNANCE` |

The code mapping is deterministic in `DEFAULT_FAILURE_POLICY`. A caller may
override its retry disposition only with explicit technical override evidence;
retryable packets require a required remediation action. `NOT_RETRYABLE` is a
supported disposition and never means an unchanged retry can succeed.

## Lifecycle stages

The stable stages are `DATA_INTAKE`, `DATA_VALIDATION`, `CALIBRATION`,
`MODEL_FIT`, `MODEL_VALIDATION`, `PROMOTION_GATE`, `SIMULATION`, and `EXPORT`.
They describe producer lifecycle position rather than user intent or display
policy.

## Producer boundary behavior

`emit_known_failure_outcome` accepts an explicitly selected known failure code
and evidence, and returns a failure `MMMExportOutcome`. It does not parse
exception strings and does not catch arbitrary `Exception`; unexpected defects
continue to fail loudly. `adapt_runtime_artifacts_to_export_outcome` wraps the
existing success adapter without changing `adapt_runtime_artifacts_to_export_bundle`.

MIP may rely on the schema, ID, technical classification, retry evidence,
remediation requirements, and constrained lineage/evidence references. MIP must
not infer user-facing wording, user intent, a TrustReport, recommendation
authority, model promotion, or permission to retry automatically.

Retry disposition is evidence, not automatic retry execution. Remediation
actions are technical requirements, not user-facing advice. Failure packets do
not authorize recommendations, assemble a TrustReport, classify user intent,
produce conversational language, or promote a model.

## Remaining boundary status

R10 typed failure packets are implemented and covered by producer contract and
boundary tests. R16 MIP consumer readiness remains blocked: MMM has not added
consumer parsing or platform policy, and the MMM–MIP interface remains
unfrozen. The next narrow producer gap is a typed run manifest (R9), subject to
its own repository-lock and scope review.
