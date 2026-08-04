# TASK_PROPOSAL_REPORT

## Current decision

- **Task:** `MMM_GIT_AUTHORITATIVE_THIN_LAUNCHER_STANDARD_ADOPTION_001`
- **Repository:** `Phani-Pavuluri/MMM`
- **Status:** `proposed`
- **Pre-authoring base:** `ac546548784385baab67d7c935e5a4fcdfc9e1af`
- **Intended branch:** `docs/mmm-git-authoritative-thin-launcher-standard-adoption-001`
- **Feature branch created:** `false`
- **Task execution authorized:** `false`
- **Correction, merge, and PR authority:** `false`
- **Analytical and capability authority changed:** `false`

This report records a durable MMM-owned proposal. It does not authorize
implementation or create a feature branch.

## Orientation and current MMM state

Connected GitHub verified MMM `main` at
`ac546548784385baab67d7c935e5a4fcdfc9e1af` before authoring. The prior task
`MMM_REPOSITORY_EXECUTION_PROTOCOL_ADOPTION_001` is merged and closed. Its
approved review head was
`c370dc7cd59a61cc2e19025d1a2328c7867b63be`, its implementation SHA was
`0e77ce6b787bd508600c1496288a459b8d821edf`, and its sole closure commit was
`ac546548784385baab67d7c935e5a4fcdfc9e1af`.

All retained MMM remote branches inspected at proposal time were fully behind
`main`; no branch was ahead, and no open pull request or active MMM feature task
covered this proposed execution-governance surface.

## Upstream dependency evidence

The canonical replacement standard is not yet merged in MIP.

Live connected GitHub observed:

- MIP `main`: `9bed0f30879e68473a37b0e65d449ea0b6a6e3f3`
- MIP task: `MIP_GIT_AUTHORITATIVE_THIN_LAUNCHER_STANDARD_001`
- MIP feature branch:
  `docs/mip-git-authoritative-thin-launcher-standard-001`
- MIP candidate implementation SHA:
  `dde6969b1192b97aea519c9589d27186f19b6db2`
- MIP candidate review head:
  `e390f1b47f8a7c5dfaa7a05613c2c4de73e4a548`
- MIP candidate status: `ready_for_review`
- MIP merge, PR, sibling-adoption, product, analytical, and capability authority:
  `false`

The candidate receipt reports five focused tests, Ruff, mypy, JSON, Markdown,
changed-path, diff, and exact-tree checks as passed, with Docker/full-suite
validation marked `not_required` by the MIP Tier-1 gate. Those are candidate
feature-branch results, not merged dependency evidence.

## Dependency decision

- **Dependency ID:** `DEP-MMM-THIN-LAUNCHER-MIP-STANDARD-MERGED-001`
- **Blocker ID:** `BLOCK-MMM-THIN-LAUNCHER-UPSTREAM-NOT-MERGED-001`
- **Blocker state:** `open`

A feature branch cannot satisfy a merged dependency. MMM must not pin the MIP
candidate review head as a canonical standard, create the proposed MMM feature
branch, or invoke Codex implementation.

The blocker resolves only after live MIP `main` records the standard task as
merged through an externally approved exact head and one closure commit. MMM
must then read the exact merged MIP main SHA, stable execution files, canonical
standard paths, validation/closure evidence, and authority state. Any material
difference from this proposal must be reconciled in Git before authorization.

## Proposed outcome and boundaries

After dependency resolution, the task will replace MMM's strict invocation-only
prompt rule with the exact merged Git-authoritative thin-launcher standard while
retaining MMM's full Docker validation and exact-head merge discipline.

The proposal allows launcher text to carry stable operational controls only:
repository path, synchronization and Git reads, Git-declared branch resumption,
continuation through exact-tree publication and remote verification,
non-terminal progress semantics, durable terminal outcomes, prohibited
operations, and the externally approved exact merge SHA. Durable task meaning
remains exclusively in Git.

The complete proposed behavior, launchers, acceptance evidence, owned paths,
validation gate, reporting requirements, dependency resolution, and prohibited
scope are recorded in `docs/execution/ACTIVE_TASK.md`.

No MMM analytical, fitting, calibration, simulation, optimization, contract,
adapter, fixture, numerical, runtime, package, release, MIP, GeoX, product, or
capability behavior is modified or authorized.

## Proposed validation

After authorization, MMM must run its complete repository-authored gate,
including Docker-backed `make validate`, on the implementation tree, exact
review head, and post-fast-forward `main`. Focused tests, Ruff, mypy, JSON,
Markdown/current-state, task-authoring boundary, changed-path, diff, exact-tree
receipt, and local/remote branch equality are also required.

Required validation that cannot run must produce Git-durable `blocked`; focused
success cannot substitute for the full MMM gate.

## Reporting debt carried forward

The prior closure validly completed but did not separately enumerate every
GitHub-observed versus locally reported fact, limitation, validation-debt,
consumer-verification, and newly eligible-work field. This proposal requires
those fields in its future completion report without reopening or amending the
closed task.

## Task-authoring boundary

The authoring range begins at
`ac546548784385baab67d7c935e5a4fcdfc9e1af` and changes only:

- `docs/execution/ACTIVE_TASK.md`
- `docs/execution/LATEST_COMPLETION_REPORT.md`

The commit containing this report is the final proposal-authoring head. The
immediate next commit must change only
`docs/execution/EXECUTION_STATE.json` to record the exact proposal head,
`status: proposed`, the open dependency/blocker, and all execution/merge/PR/
capability flags as false.

That state-only proposal commit must not authorize execution or create the
feature branch. A later, separate state-only authorization boundary is required
after the dependency is resolved and the proposed task is reverified or updated.

## Next action

Wait for the MIP standard to merge. Then re-fetch live MMM and MIP `main`, read
their stable execution evidence, reconcile the exact MIP merged pin and text,
and author the separate MMM state-only authorization boundary. No Codex launcher
is valid before that transition.
