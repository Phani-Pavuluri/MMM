# MMM supported-range evidence contract

`MMMSupportedRangeEvidence` (`mmm_supported_range_evidence_v1`) is an MMM-owned,
versioned producer artifact describing the technical domain in which a scoped
model output may be evaluated. It is technical evidence, not MIP parsing,
conversational policy, a response surface, a simulation, a TrustReport, a
recommendation, or production authorization.

## Meaning and scope

Each `MMMSupportedRangeRecord` identifies a run and optional model, then scopes
range evidence to channel, KPI, geography, segment, time window, outcome or
estimand, grain, and transformation. `MMMRangeBound` carries a finite value,
inclusive/exclusive semantics, unit, and either raw scale or an explicit
transformation identity. Incompatible units, mixed scale semantics, invalid
bounds, unsafe text, and unknown required schema versions fail closed.

Observed bounds, supported bounds, and validated bounds are distinct fields.
Observed does not mean supported; supported does not mean causally valid or
production-authorized. A validated bound requires explicit validation evidence.
When only observed evidence exists, the record is `UNAVAILABLE` or
`PARTIALLY_AVAILABLE`; MMM does not invent support with a multiplier, percentile,
standard-deviation rule, tolerance, boundary buffer, or automatic widening.

`MMMRangeEvidenceBasis` records whether evidence is observed data, training
domain, holdout/diagnostic validation, calibration or governance restriction,
model structural evidence, research-only, or unknown. Availability is separately
typed as available, partially available, unavailable, blocked, or research-only.
Unavailable evidence never means unrestricted support.

## Relations, extrapolation, restrictions, and uncertainty

`MMMRangeRelation` distinguishes in-domain, explicit boundary, outside-observed
but supported, outside-supported, and unknown states. No near-boundary heuristic
is introduced. `MMMExtrapolationClassification` distinguishes interpolation,
boundary use, limited extrapolation only when existing governed evidence is
provided, unsupported extrapolation, and unknown. Unsupported extrapolation
requires typed restricted or blocked claim effects through `MMMRangeRestriction`.

Restrictions identify a stable code, technical summary, affected scope,
evidence references, affected technical claims, and claim dispositions. They
reuse `MMMTechnicalClaim` and `MMMTechnicalClaimDisposition`; listing a claim
does not authorize it. Optional uncertainty is a reference and semantics only:
no uncertainty value is fabricated, and unavailable uncertainty cannot claim an
artifact.

## Existing producer evidence and boundaries

Ridge range records map existing observed/training-domain evidence only. This
contract changes neither coefficients nor fitting and introduces no candidate-plan
or response-surface result. Existing curve diagnostic interpolation remains
diagnostic; its explicit research-only linear extrapolation does not create
governed production support. Bayesian evidence is represented as `RESEARCH_ONLY`
with production-use claims blocked; Bayesian fitting and promotion are unchanged.

The aggregate links additively by stable IDs to a run manifest, diagnostics and
limitations, calibration lineage, export artifact, and terminal failure. The
manifest/export boundary accepts an optional aggregate only when its run ID
matches. Existing `UNSUPPORTED_EXTRAPOLATION` failures can reference records;
this contract creates no duplicate failure taxonomy and catches no unexpected
exceptions.

## Fixtures and compatibility

The deterministic `tests/fixtures/mip_export/supported_range_v1/` collection is
a separate additive fixture collection. `golden_v1` is unchanged, as required by
the schema compatibility policy. Fixtures cover governed Ridge evidence,
observed-only unavailable evidence, boundary restriction, blocked unsupported
extrapolation, pre-model insufficiency, Bayesian research-only evidence, and
non-interchangeable multi-channel evidence.

R11 public simulation export and R12 response-surface evidence remain partial.
R16 MIP consumer readiness remains blocked and the producer interface remains
unauthorized for freeze. In-range status does not authorize recommendations;
supported-range evidence is not a response surface, simulation, or TrustReport.
