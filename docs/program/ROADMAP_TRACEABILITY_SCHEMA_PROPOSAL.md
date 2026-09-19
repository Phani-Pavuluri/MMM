# Roadmap traceability schema — proposal (NON-NORMATIVE DRAFT)

> **Status: proposal only — not implemented, not enforced, not authorizing.**
> Nothing in this document applies to any tooling, gate, or process until a separately
> authorized task adopts it. It exists so a future task can review a concrete shape instead
> of starting from prose.

## Purpose

Give roadmap and phase references a stable, machine-checkable shape for **traceability only**:
which canonical roadmap a document or task traces to, and which product phase it belongs to.
These refs would contextualize work; they would never authorize it. Execution authority would
remain exclusively `docs/execution/EXECUTION_STATE.json` under the task-owned lifecycle.

## Proposed fields

| Field | Type | Meaning |
|-------|------|---------|
| `roadmap_ref` | canonical roadmap path (string) | The single roadmap document this item traces to, e.g. `docs/05_validation/platform_roadmap.md`. Empty only where `phase_ref` is `not_applicable`. |
| `phase_ref` | closed enum (string) | The product phase this item belongs to. No free text. |

## Proposed `phase_ref` enum

Product phases are owned by the canonical roadmap; this proposal only suggests the bucket names:

- Product phases as named in `docs/05_validation/platform_roadmap.md` and its Track 2 detail
  `docs/05_validation/synthetic_validation_roadmap.md`
- `maintenance` — hygiene, dependency, and tooling upkeep with no product meaning
- `governance` — authority, lifecycle, and audit mechanics (e.g. this inventory's own track)
- `documentation` — navigation, inventory, and proposal docs that change no behavior
- `not_applicable` — escape hatch; avoids forced labels on items that trace nowhere

## Non-goals of this proposal

- No change to `docs/execution/EXECUTION_STATE.json` meaning or authority.
- No change to taskctl code, schema, transitions, or generated views.
- No new required metadata on any existing document or task.
- No validation beyond what a future adopting task defines.

## Adoption

Requires a separately authorized task that names exact owned/prohibited paths, validation, and
the taskctl checks (if any) that would enforce these fields. Until then, this file is a draft
pointer and nothing more.
