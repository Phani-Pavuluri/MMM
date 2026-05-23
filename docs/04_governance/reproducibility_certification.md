# Reproducibility certification

Proves **artifact equivalence** under identical inputs — not causal validity.

## Artifact

`extension_report.reproducibility_certification_report` when `extensions.reproducibility_certification.enabled: true`.

| Field | Meaning |
|-------|---------|
| `reproducibility_status` | `certified` or `mismatch` |
| `identical_output` | All compared components match |
| `mismatched_components` | Named components that differ |
| `coefficient_deltas` | Numeric coef drift when applicable |
| `optimizer_output_deltas` | Δμ / allocation drift |
| `bundle_hash_match` | Decision bundle subset hash equality |

## API

`mmm.governance.reproducibility_certification`:

- `extract_reproducibility_snapshot(...)`
- `compare_reproducibility_snapshots(reference, candidate)`
- `build_reproducibility_certification_report(...)`

## Limitations

- Requires the same data panel, resolved seeds, config fingerprints, transform parameters, and promotion lineage.
- Self-certification on a single run only checks internal consistency; cross-run comparison needs two stored snapshots.
