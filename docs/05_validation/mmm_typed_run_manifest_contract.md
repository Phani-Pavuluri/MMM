# Typed MMM run-manifest contract

`MMMRunManifest` (`mmm_mip_run_manifest_v1`) is MMM-owned, versioned technical
producer evidence for the MMM–MIP boundary. It records a stable manifest/run
identity, package and contract versions, optional model/configuration identity,
dataset and calibration lineage, ordered execution steps, artifact references,
and one terminal outcome.

The contract is implemented in `mmm/contracts/run_manifest.py`. The additive
`adapt_runtime_artifacts_to_export_manifest_outcome` path in
`mmm/contracts/mip_export_adapter.py` links it to the existing `MMMExportOutcome`
without changing the successful export API.

## Stable semantics

- `MMMRunStatus` is `RUNNING`, `SUCCEEDED`, `FAILED`, or `BLOCKED`.
- `MMMRunStep` has a contiguous `sequence`, a producer lifecycle stage, a typed
  step status, safe artifact references, validation/diagnostic IDs, and optional
  `MMMFailurePacket` identity.
- `MMMArtifactReference` contains only type, stable ID, contract version,
  optional fingerprint/logical name, and availability. It never carries a path,
  binary payload, DataFrame, model, trace, exception, or secret.
- A successful manifest has an `MMMExportBundle` reference and no failure
  packet. Failed/blocked manifests have exactly one typed failure packet and no
  successful export reference. A running manifest has neither terminal payload.
- JSON serialization omits unavailable optional identifiers, is deterministic,
  and rejects unsupported schema versions.

MIP may rely on these typed producer facts and their failure linkage. It must
not infer recommendation authorization, production promotion, user intent,
conversational answerability, retry execution, or a TrustReport from a
manifest. Retry disposition remains evidence on `MMMFailurePacket`, not an
automatic retry command.

The manifest is technical producer evidence. It is not a TrustReport, does not
classify intent or produce conversational language, does not authorize
recommendations or budget optimization, and does not promote a model. MMM
continues to own the producer artifacts and technical evidence; MIP owns
consumer parsing policy and all platform/user-facing policy.

The MMM–MIP interface remains **not frozen**. R16
`MIP_CONSUMER_READINESS` remains **blocked**; no MIP consumer behavior is
implemented by this contract.
