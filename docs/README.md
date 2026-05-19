# Documentation

Entry point for the MMM geo-level library. Docs are grouped by **user journey**, not by implementation history.

## New here?

Start with what the system is, then run a train:

1. [01_getting_started/overview.md](01_getting_started/overview.md) — purpose and design goals  
2. [01_getting_started/quickstart.md](01_getting_started/quickstart.md) — install, train, first artifacts  
3. [01_getting_started/config_yaml.md](01_getting_started/config_yaml.md) — configuration reference  
4. [01_getting_started/python_api.md](01_getting_started/python_api.md) — programmatic train and decide APIs  

Optional: [01_getting_started/best_practices.md](01_getting_started/best_practices.md)

## Understand the system

Concepts and modeling behavior (read before production decisioning):

| Topic | Doc |
|--------|-----|
| System shape | [02_concepts/architecture.md](02_concepts/architecture.md) |
| Decision vs diagnostic outputs | [02_concepts/decision_vs_research.md](02_concepts/decision_vs_research.md) |
| Experiments & calibration | [02_concepts/calibration.md](02_concepts/calibration.md) |
| Post-fit diagnostics | [02_concepts/diagnostics.md](02_concepts/diagnostics.md) |
| Split-channel separability | [02_concepts/feature_separability.md](02_concepts/feature_separability.md) |
| Experiment prioritization | [02_concepts/experiment_scheduler.md](02_concepts/experiment_scheduler.md) |
| Control CSV templates (onboarding) | [02_concepts/control_templates.md](02_concepts/control_templates.md) |
| Response curves (diagnostic) | [02_concepts/response_curves.md](02_concepts/response_curves.md) |
| CV modes | [02_concepts/cv_modes.md](02_concepts/cv_modes.md) |
| Ridge + BO | [02_concepts/ridge_bo.md](02_concepts/ridge_bo.md) |
| Bayesian MMM | [02_concepts/bayesian.md](02_concepts/bayesian.md) |

## Plan or optimize budgets

Operational workflows for analysts and data scientists:

| Topic | Doc |
|--------|-----|
| **How-to (start here)** | [03_planning/planning_howto.md](03_planning/planning_howto.md) |
| Decision rules & CLI contract | [03_planning/decision_runbook.md](03_planning/decision_runbook.md) |
| Budget optimization concepts | [03_planning/budget_optimization.md](03_planning/budget_optimization.md) |
| Execution pipeline (developers) | [03_planning/planning_execution.md](03_planning/planning_execution.md) |

## Run in production

Governance, safety checks, and operations:

| Topic | Doc |
|--------|-----|
| Operator workflows | [04_governance/operator_runbook.md](04_governance/operator_runbook.md) |
| Prod safety checklist | [04_governance/prod_safety_checklist.md](04_governance/prod_safety_checklist.md) |
| Artifact & bundle fields | [04_governance/artifact_schema.md](04_governance/artifact_schema.md) |

## Archive

Historical / roadmap notes (not current operator guidance):

- [_archive/roadmap_causal_calibration_governance.md](_archive/roadmap_causal_calibration_governance.md)
