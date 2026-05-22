# Cross-branch documentation (not on this branch)

**Status:** Quarantined reference only. The following topics are **not implemented** on `fix/prod-governance-evidence-gates` (and typical `main` / governance merges). Do not add user-facing docs under `docs/` that describe them as current product until the matching code is merged.

| Topic | Intended branch | Package entry (when merged) | Do not link from quickstart / runbooks until shipped |
|-------|-----------------|------------------------------|------------------------------------------------------|
| Experiment scheduler | `feat/feature-separability-governance` | `mmm.evaluation.experiment_scheduler` | `experiment_scheduler_report`, `extensions.experiment_scheduler` |
| Feature separability | `feat/feature-separability-governance` | `mmm.evaluation.feature_separability` | `feature_separability_report` |
| Control template CLI | `feat/feature-separability-governance` | `mmm.helpers.control_templates`, `mmm generate-control-template` | — |

## If you merge docs from another branch

1. **Do not** copy `docs/02_concepts/experiment_scheduler.md`, `feature_separability.md`, or `control_templates.md` without the code paths above.
2. If those files appear in a docs-restructure merge, move them here or delete them until the feature branch lands.
3. `experiment_scheduler_seed` in config/seed contract is reserved for future wiring; it does not enable scheduling on this branch.

## Related on this branch (shipped)

- Seed contract (`seed_resolution` artifact) including `experiment_scheduler_seed` placeholder
- `split_channel_policy` governance hook (no-op without `feature_separability_report`)
- Fingerprint v2, drift report, curve–Δμ alignment, Ridge uncertainty disclosure — see [documentation_truth_audit.md](../documentation_truth_audit.md)
