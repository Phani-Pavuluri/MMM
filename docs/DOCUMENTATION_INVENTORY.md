# Documentation inventory

Canonical docs live under `docs/` in numbered journey folders. There are **no** duplicate flat copies at `docs/*.md` (except this inventory, `README.md`, and `documentation_truth_audit.md`).

| Source file | Canonical location | Status | Notes |
|-------------|-------------------|--------|-------|
| `docs/01_getting_started/best_practices.md` | `01_getting_started/best_practices.md` | canonical | — |
| `docs/01_getting_started/config_yaml.md` | `01_getting_started/config_yaml.md` | canonical | — |
| `docs/01_getting_started/overview.md` | `01_getting_started/overview.md` | canonical | — |
| `docs/01_getting_started/python_api.md` | `01_getting_started/python_api.md` | canonical | — |
| `docs/01_getting_started/quickstart.md` | `01_getting_started/quickstart.md` | canonical | — |
| `docs/02_concepts/architecture.md` | `02_concepts/architecture.md` | canonical | — |
| `docs/02_concepts/bayesian.md` | `02_concepts/bayesian.md` | canonical | — |
| `docs/02_concepts/calibration.md` | `02_concepts/calibration.md` | canonical | — |
| `docs/02_concepts/control_templates.md` | `02_concepts/control_templates.md` | canonical | — |
| `docs/02_concepts/cv_modes.md` | `02_concepts/cv_modes.md` | canonical | — |
| `docs/02_concepts/statistical_validation.md` | `02_concepts/statistical_validation.md` | canonical | — |
| `docs/02_concepts/decision_vs_research.md` | `02_concepts/decision_vs_research.md` | canonical | — |
| `docs/02_concepts/diagnostics.md` | `02_concepts/diagnostics.md` | canonical | — |
| `docs/02_concepts/experiment_scheduler.md` | `02_concepts/experiment_scheduler.md` | canonical | — |
| `docs/02_concepts/feature_separability.md` | `02_concepts/feature_separability.md` | canonical | — |
| `docs/02_concepts/response_curves.md` | `02_concepts/response_curves.md` | canonical | — |
| `docs/02_concepts/ridge_bo.md` | `02_concepts/ridge_bo.md` | canonical | — |
| `docs/03_planning/budget_optimization.md` | `03_planning/budget_optimization.md` | canonical | — |
| `docs/03_planning/decision_runbook.md` | `03_planning/decision_runbook.md` | canonical | — |
| `docs/03_planning/planning_execution.md` | `03_planning/planning_execution.md` | canonical | — |
| `docs/03_planning/planning_howto.md` | `03_planning/planning_howto.md` | canonical | — |
| `docs/04_governance/artifact_schema.md` | `04_governance/artifact_schema.md` | canonical | Replaces legacy `planning_artifact_schema.md` |
| `docs/04_governance/operator_runbook.md` | `04_governance/operator_runbook.md` | canonical | — |
| `docs/04_governance/prod_safety_checklist.md` | `04_governance/prod_safety_checklist.md` | canonical | — |
| `docs/04_governance/ridge_uncertainty_research.md` | `04_governance/ridge_uncertainty_research.md` | canonical | Research findings only |
| `docs/README.md` | `README.md` | canonical | Entry index |
| `docs/documentation_truth_audit.md` | `documentation_truth_audit.md` | canonical | Truth audit |
| `docs/planning_artifact_schema.md` | `04_governance/artifact_schema.md` | deprecated | Redirect stub only |
| `docs/DOCUMENTATION_INVENTORY.md` | `DOCUMENTATION_INVENTORY.md` | canonical | This file |
| `docs/_archive/cross_branch_not_shipped.md` | `_archive/cross_branch_not_shipped.md` | archived | Cross-branch quarantine policy |
| `docs/_archive/roadmap_causal_calibration_governance.md` | `_archive/roadmap_causal_calibration_governance.md` | archived | Roadmap only |

## Deprecated redirects

| Former path | Redirect |
|-------------|----------|
| `docs/planning_artifact_schema.md` | Use [`04_governance/artifact_schema.md`](04_governance/artifact_schema.md) |
| `docs/quickstart.md` (flat) | Use [`01_getting_started/quickstart.md`](01_getting_started/quickstart.md) |
| `docs/decision_runbook.md` (flat) | Use [`03_planning/decision_runbook.md`](03_planning/decision_runbook.md) |

## Validation

```bash
python scripts/validate_docs.py
```
