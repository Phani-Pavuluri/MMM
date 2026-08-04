# MMM Lean Repository Delivery Standard

## Delivery shape

One authorized task has one independently mergeable outcome. Split when a
portion can be validated/reviewed alone or changes a public contract, migration,
integration surface, or authority boundary. One correction cycle is the default;
new independent work becomes a deferred successor.

An executable task declares exact observable behavior and preserved boundaries,
surface-appropriate resolved decisions, inputs/outputs/invariants/failures,
compatibility or migration policy (or `not_applicable`), named deterministic evidence,
named acceptance tests, owned/prohibited paths, risk tier, validation, deferred successors,
and `unresolved execution-blocking design questions: none`. Otherwise it stays
proposed, becomes design-blocked, or is split.

## Risk tiers

| Tier | Scope | Minimum evidence |
|---|---|---|
| 1 | Documentation or governance | Focused structure, path, and semantic tests |
| 2 | Public/package surface | Focused tests plus surface-required validation |
| 3 | Cross-repository, analytical, authority, or production boundary | Owner evidence, coordination review, full applicable gate |

This model never weakens existing MMM validation. Docker-backed `make validate`
is mandatory whenever the repository gate, active task, Tier 3, or changed
analytical/public/package surface requires it. Required failures are Git-durable
`blocked`; genuinely inapplicable categories are `not_required`.

## Durable terminal outcome

Successful orientation is not completion. Continue to a Git-durable
`ready_for_review` with an exact-tree receipt, or a Git-durable `blocked` state
with diagnosis, attempted evidence, validation-category statuses, and a live
resolution condition. No chat-only or orientation-only result is completion.
