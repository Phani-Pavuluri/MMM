# Promotion workflow (explicit, immutable)

Promotion records which training run artifacts may be used for **prod decision surfaces**. Promotion is **human-reviewed**, **append-only**, and **never automatic**.

## Lifecycle

```text
train run → extension_report + fingerprints → human review → promotion record → prod decide (optional gate)
```

## Config

```yaml
governance:
  require_promoted_model_for_prod_decision: false   # default: backward compatible
  promotion_registry_path: path/to/promotions.jsonl
```

When `require_promoted_model_for_prod_decision: true`, prod `mmm decide` paths must supply `promoted_model_id` or a promotion JSON path. The record’s data/config fingerprints must match loaded artifacts; expired or unsupported surfaces fail closed.

## PromotionRecord fields

- `promotion_id`, `run_id`, `model_id`, `artifact_uri`
- `data_fingerprint`, `config_fingerprint`, `model_fingerprint`, `seed_resolution`
- `promoted_by`, `promoted_at`, `approval_status`, `approval_notes`
- `allowed_surfaces` (e.g. `simulate`, `optimize_budget`)
- `expiration_date`, `rollback_of`, `parent_promotion_id`
- `governance_summary`, `calibration_summary`, `unsupported_questions`
- `signature_hash` (tamper detection)

## Rules

- **No auto-promotion** — call `promote_run` explicitly after review.
- **Immutable registry** — append-only JSONL; duplicate `promotion_id` rejected.
- **Rollback** — `rollback_promotion` appends a new record referencing the prior promotion (does not delete history).
- **Eligibility** — requires `governance.approved_for_optimization`, `model_release` planning/reporting allowed, fingerprint v2, and prod replay evidence when applicable.

## Decision bundle lineage

When promotion is supplied, decision bundles may include:

- `promoted_model_id`, `promotion_id`, `promotion_registry_ref`
- `promotion_fingerprint_match`, `promotion_expiration_date`, `rollback_lineage`

## What promotion does not mean

Promotion certifies that **artifacts passed package governance checks at review time**. It does **not** prove causal incrementality, that experiments were well powered, or that replay calibration was unbiased — see replay refit disclosure and experiment evidence docs.
