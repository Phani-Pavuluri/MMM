# Decision trace (`decision_trace.json`)

Single JSON artifact explaining a **simulate** or **optimize-budget** recommendation end-to-end.

## When emitted

Prod `mmm decide` paths with `--out` write `decision_trace.json` beside the decision JSON.

Python API returns `decision_trace` on the result dict.

## Sections

| Section | Contents |
|---------|----------|
| `identity` | `decision_id`, promotion ids, surface |
| `lineage` | Fingerprints, seed resolution, artifact refs |
| `calibration` | Replay summary, gap, evidence, freshness |
| `decision` | Baseline, constraints, Δμ, ROI, `decision_safe` |
| `governance` | Unsupported questions, warnings, approvals |

## Limitations

- Trace summarizes artifacts; it does not replace reading raw `extension_report.json` for research diagnostics.
- Missing optional reports are omitted — not fabricated.
