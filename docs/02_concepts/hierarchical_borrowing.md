# Hierarchical borrowing (Ridge, opt-in)

Hierarchical borrowing is **explicit regularization** for Ridge Bayesian optimization. It is **not** causal identification, **not** experiment evidence, and **not** enabled unless you supply a `HierarchyDefinition` JSON file.

## What it does

When `hierarchy.enabled=true`, the Ridge BO objective adds:

```text
hierarchical_penalty = lambda_hier * sum((local_effect - parent_effect)^2)
```

over **explicitly mapped** media coefficients (typically child channel → parent channel). The penalty is added to the same calibration loss slot used for replay (alongside predictive / stability / plausibility / complexity terms).

## What it does not do

- Infer parent/child structure from the panel
- Imply that parent-level experiments justify child-level causal claims
- Change planning, decision APIs, or Bayesian training (PR 4B is separate)
- Pool across unrelated branches unless `hierarchy.allow_cross_branch_pooling=true`

## Configuration

```yaml
hierarchy:
  enabled: false
  hierarchy_definition_path: null
  hierarchy_type: geography   # advisory; file is source of truth
  regularization_strength: 0.1
  min_children_per_parent: 2
  allow_cross_branch_pooling: false
```

Legacy `calibration.hierarchical_regularization_enabled` is deprecated; use `hierarchy.enabled` instead.

## Model form

- Hierarchical borrowing is supported for **`model_form=semi_log`** (production Ridge path).
- **`model_form=log_log` + `hierarchy.enabled`** is blocked: LOG_LOG hierarchy support awaits formal model-form validation.
- Production canonical Ridge decisions use **SEMI_LOG** only; see [ridge_bo.md](ridge_bo.md).

## Bayesian hierarchy (research-only)

Ridge uses an explicit L2 penalty (`hierarchy.enabled`). Bayesian uses **`bayesian.use_hierarchy`** with the same `HierarchyDefinition` JSON:

- **child ~ Normal(parent, hier_sigma_group)** on media coefficients
- Emits **`bayesian_hierarchy_report`** (not a prod decision input)
- Does not enable prod Bayesian decisioning; see [bayesian.md](bayesian.md)
- **LOG_LOG + `bayesian.use_hierarchy`** is blocked

## HierarchyDefinition contract

JSON fields: `hierarchy_id`, `hierarchy_type`, `parent_nodes`, `child_nodes`, `node_mapping`, `metadata`, `version`.

Examples:

- **Channel:** `Paid_Social` → `Meta`, `TikTok`, `Reddit` (all names must appear in `data.channel_columns`)
- **Geography:** geo ids in `node_mapping`; Ridge penalty on channels requires `metadata.ridge_effect_pairs` listing channel child/parent pairs

Validation rejects cycles, duplicate child assignment, missing entities, insufficient children per parent, orphan nodes, disconnected structures, and parent-as-descendant assignments.

## Shrinkage interpretation

Diagnostics report how far each child coefficient sits from its mapped parent after fit. Larger `regularization_strength` increases the BO penalty and typically pulls noisy child estimates toward the parent during hyperparameter search. This stabilizes variance; it does **not** prove the parent effect is causal for the child.

## Governance

Extension reports emit `hierarchy_diagnostics`, `hierarchy_effect_summary`, and fixed governance warnings. Unsupported questions include local causal claims from parent-only evidence and child-level experiment conclusions from parent evidence.
