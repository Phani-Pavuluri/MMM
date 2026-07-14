# MMM calibration-treatment lineage contract

`MMMCalibrationTreatmentLineage` (`mmm_calibration_treatment_lineage_v1`) is
versioned producer evidence for every CalibrationSignal that crossed MMM's
consideration boundary. Each `MMMCalibrationTreatmentRecord` keeps freshness,
compatibility, final disposition, roles, scope/estimand/uncertainty references,
transformations, and optional typed calibration failure separate.

Considered does not mean applied; accepted does not mean model-affecting; and
evidence context does not mean refit. Ridge signals are diagnostic/evidence
context only: they do not adjust coefficients, priors, likelihoods, constraints,
optimizer inputs, or recommendations. Bayesian prior/likelihood records require
explicit research-only and governing-policy evidence; this contract does not
change Bayesian fitting or promotion status.

Fresh, stale, expired, and unknown are independent of compatible, partially
compatible, incompatible, and unknown. Expired or incompatible signals cannot
be applied. Stale application requires explicit policy and downweight evidence.
Conflicting signals remain separate records; lineage never represents silent
averaging or coefficient override. Transformation steps are ordered, typed, and
contain only safe producer metadata.

The additive export-boundary helper can link a lineage ID and considered signal
IDs to `MMMRunManifest`; it preserves the existing export outcome API. Known
calibration failures reuse `MMMFailurePacket` codes rather than creating a
second failure taxonomy.

This is technical producer evidence, not a TrustReport, recommendation, budget
optimization authorization, promotion decision, intent classifier, or
conversational response. MIP owns consumer parsing and user-facing policy. The
interface remains **not frozen**, and R16 MIP consumer readiness remains
**blocked**.
