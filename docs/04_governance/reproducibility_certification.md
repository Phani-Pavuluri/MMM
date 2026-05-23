# Reproducibility certification

Proves **artifact equivalence** under identical inputs — not causal validity.

## Artifact

`extension_report.reproducibility_certification_report` when `extensions.reproducibility_certification.enabled: true`.

| Field | Meaning |
|-------|---------|
| `self_certification` | `true` when no independent reference run was supplied |
| `identical_output` | `null` on self-cert; `true`/`false` only after independent comparison |
| `reproducibility_evidence` | `true` only when an independent run comparison passed |
| `certification_status` | `pass`, `fail`, or `incomplete` (snapshot-only) |
| `coefficients_match` | Independent-run coef equality |
| `design_matrix_match` | Transform + panel fingerprint hash match |
| `decision_output_match` | Optimizer Δμ / allocation match |
| `fingerprint_match` | Panel fingerprint token match |
| `reference_run_path` | Optional path to prior `extension_report.json` |

## Independent evidence

```yaml
extensions:
  reproducibility_certification:
    enabled: true
    reference_run_path: /path/to/prior/run   # or extension_report.json
```

Self-certification alone does **not** set `reproducibility_evidence` and cannot satisfy strict production readiness.

## API

`mmm.governance.reproducibility_certification`:

- `extract_reproducibility_snapshot(...)`
- `compare_reproducibility_snapshots(reference, candidate)`
- `build_reproducibility_certification_report(reference=..., reference_run_path=...)`

## Limitations

- Requires the same data panel, resolved seeds, config fingerprints, transform parameters, and promotion lineage.
- Does not prove causal incrementality or external validity.
