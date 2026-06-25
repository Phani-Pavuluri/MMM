# Future MMM Package-Side Support Agents — Roadmap (Deferred)

**Status:** **Roadmap capture only** — no runtime implementation  
**Date:** 2026-05-22  
**Related:** [platform_roadmap.md](platform_roadmap.md) Track 5 · [ROADMAP_ALIGNMENT_GATE.md](../ROADMAP_ALIGNMENT_GATE.md) · [INV-039](../06_investigations/open_investigations.md#inv-039--auto-retrain-auto-promotion-agentic-orchestration-out-of-v1-scope) · [AUDIT-MIP-C6](../audits/AUDIT-MIP-C6_INTEGRATION_READINESS_CHECKPOINT.md)

---

## 1. Purpose

Record a **future** roadmap for **MMM package-side support agents** — optional interpretive layers around the deterministic MMM package. This document captures architecture boundaries, prerequisites, agent roles, typed handoff concepts, and examples.

**This is docs-only.** No agents, LLM calls, LangGraph, provider SDKs, runtime orchestration, model execution changes, or new APIs are authorized by this roadmap.

---

## 2. Architecture stance

### Division of responsibility

| Layer | Owns |
|-------|------|
| **MIP** | Orchestration, user-facing agent routing, TrustReport governance, evidence routing, decision-support boundaries |
| **MMM package** | MMM modeling, refresh diagnostics, calibration diagnostics, model artifacts, decision-surface diagnostics, MMM-specific execution outputs |
| **Future MMM package-side agents** | Interpret MMM diagnostics and failures **only** — must not become the modeling authority |

### Design principle

> Agents are optional support layers around the MMM package.  
> They are **not** the MMM modeling engine.

Agents reduce **operator burden after the system already knows what failed**. They must **not** compensate for weak validation, incomplete deterministic package design, or missing framework contracts.

### What agents may do

- Explain
- Diagnose
- Validate (against governed contracts)
- Summarize
- Propose **safe remediation** around governed MMM package outputs

### What agents must never do

- Fit models or change priors / likelihoods outside governed package logic
- Impute data silently
- Change modeling assumptions
- Promote artifacts or models
- Approve decision surfaces
- Recommend budgets
- Bypass TrustReport
- Silently equate incompatible estimands
- Override CalibrationSignal mapping status
- Declare Bayesian production-ready when roadmap says research-only

---

## 3. Priority decision

### Now

- Document future MMM package-side agents and their boundaries (this roadmap).
- Keep the MMM package **deterministic** and **diagnostics-first**.
- Strengthen typed validation, run manifests, diagnostic summaries, failure packets, artifact lineage, and allowed/blocked retry actions.

### Near term

- Typed run manifests and failure packets (MMM-side contracts).
- Stable CalibrationSignal-to-MMM ingestion (MIP-C1–C5 complete; C6 scheduler deferred).
- Structured refresh/readiness diagnostic failure reasons.

### Medium term

- Integrate MMM structured outputs with MIP agent contracts:
  - `AgentRunManifest`
  - `AgentFailurePacket`
  - `AgentResolutionPlan`
  - `AgentValidationReport`

### Only later

- Implement MMM package-side support agents **if** repeated MMM workflow failures show that diagnostic interpretation, failure recovery, calibration reconciliation, or model-governance explanation would reduce **real** operator burden.

---

## 4. Prerequisites (do not prioritize agents before these)

| # | Prerequisite | Current status (MMM repo) |
|---|--------------|---------------------------|
| 1 | MMM deterministic interfaces are stable | **Partial** — train/decide contracts exist; run-manifest schema not finalized |
| 2 | MMM run manifests exist or are planned | **Planned** — lineage in extension reports; `MMMRunManifest` not yet a first-class contract |
| 3 | MMM diagnostic outputs are typed and machine-readable | **Partial** — H7–H11 Ridge diagnostics typed; not all refresh paths emit structured summaries |
| 4 | CalibrationSignal ingestion/mapping into MMM is stable | **Yes (file bridge)** — MIP-C1–C5; live scheduler deferred (MIP-C6) |
| 5 | Model refresh/readiness diagnostics expose structured failure reasons | **Partial** — validation errors exist; `MMMFailurePacket` not standardized |
| 6 | MIP has agent run manifest, failure packet, resolution plan, and validation report contracts | **Planned** — MIP-side; not in this repo |
| 7 | Package adapters define safe allowed/blocked actions | **Planned** — `MMMRetryPolicy` / adapter action matrix not yet defined |

**Rule:** Do not add MMM package-side agents before deterministic diagnostics and failure packets are stable.

---

## 5. Typed handoff concepts (future contracts)

These MMM-side contract names are **roadmap concepts** — prerequisites before package-side agents become useful. They align with planned MIP agent contracts.

| Contract | Role |
|----------|------|
| **MMMRunManifest** | Top-level record of a train/refresh/decide run: run_id, config fingerprint, data fingerprint, step list, disposition, lineage |
| **MMMStepManifest** | Per-step status within a run (ingest, fit, validate, calibrate, diagnose, export) |
| **MMMFailurePacket** | Structured failure: step, error class, human summary, machine codes, artifact pointers — not raw stack trace alone |
| **MMMDiagnosticSummary** | Typed rollup of diagnostics (Ridge, calibration, separability, governance flags) — extends H8-style summaries |
| **MMMResolutionPlan** | Safe remediation options with allowed/blocked actions and escalation path |
| **MMMRetryPolicy** | Governed retry rules: max attempts, idempotency keys, partial-run prohibition |
| **MMMValidationReport** | Input/schema/config validation outcome before execution |
| **MMMModelPromotionReview** | Governance checklist: readiness, TrustReport, research-only flags, gate pass/fail |
| **MMMCalibrationReconciliationReport** | CalibrationSignal alignment summary: usable / stale / conflicting / missing |

**MIP-side counterparts (planned):** `AgentRunManifest`, `AgentFailurePacket`, `AgentResolutionPlan`, `AgentValidationReport`.

Agents consume these contracts; they do not invent parallel semantics.

---

## 6. Future MMM package-side agents

### 6.1 MMM Data Contract Agent

**Purpose:** Validate MMM input data contracts and explain missing or incompatible fields.

**Responsibilities:**

- spend / outcome / control / calendar schema checks
- time-grain checks
- channel coverage checks
- missing week detection
- geo / national compatibility checks
- control variable availability
- seasonality / calendar field readiness

| Allowed | Not allowed |
|---------|-------------|
| Explain why input is invalid | Silently impute missing spend |
| Ask for missing spend/outcome/control fields | Invent controls |
| Propose safe data correction questions | Change grain |
| Route back to MIP readiness / common intake | Drop channels without governed warning |
| | Approve model execution |

**Trigger:** Add when MMM ingestion contracts and refresh prerequisites are stable.

---

### 6.2 MMM Diagnostic Interpreter Agent

**Purpose:** Explain MMM diagnostic outputs in user-facing or operator-facing language.

**Responsibilities:**

- residual diagnostics
- fold diagnostics
- collinearity warnings
- coefficient instability
- saturation / adstock plausibility warnings
- calibration conflict summaries
- Ridge diagnostic interpretation (H7–H11 stack)
- Bayesian research-only status explanation

| Allowed | Not allowed |
|---------|-------------|
| Summarize diagnostics | Declare model valid if gates fail |
| Explain likely causes | Promote diagnostic curves to decision authority |
| Propose safe next checks | Claim causal ROI without TrustReport / decision-surface governance |
| Distinguish diagnostic-only vs production-safe outputs | |

**Trigger:** Add after MMM diagnostics are sufficiently structured and stable.

---

### 6.3 MMM Calibration Reconciliation Agent

**Purpose:** Interpret how CalibrationSignals align or conflict with MMM calibration needs.

**Responsibilities:**

- CalibrationSignal freshness
- metric alignment
- estimand alignment
- channel / scope / time-window alignment
- uncertainty availability
- conflicting evidence explanation
- local vs national evidence interpretation

| Allowed | Not allowed |
|---------|-------------|
| Explain which signals are usable, stale, conflicting, or missing | Fabricate missing uncertainty |
| Ask for missing uncertainty / scope fields | Override CalibrationSignal mapping status |
| Recommend evidence review | Execute calibration |
| | Change priors / likelihoods without governed package logic |
| | Certify evidence as causal |

**Trigger:** Add after CalibrationSignal-to-MMM ingestion is stable and multiple calibration evidence sources exist (MIP-C1–C5 foundation in place; agent deferred).

---

### 6.4 MMM Refresh Failure Agent

**Purpose:** Diagnose failed MMM refresh or scheduled diagnostic runs.

**Responsibilities:**

- failed refresh step identification
- stack trace summarization (into `MMMFailurePacket`)
- missing input detection
- config mismatch explanation
- dependency / version issue triage
- artifact missing / freshness issue triage
- safe retry plan proposal

| Allowed | Not allowed |
|---------|-------------|
| Summarize failure packet | Retry indefinitely |
| Propose safe retry per `MMMRetryPolicy` | Patch data silently |
| Ask for missing data / config correction | Change model configuration without record |
| Escalate to MIP / human review | Approve partial runs |

**Trigger:** Add when refresh jobs, schedulers, MLflow/model registry, or production-like batch runs exist.

---

### 6.5 MMM Model Governance Agent

**Purpose:** Check whether MMM model artifacts and decision surfaces satisfy governance requirements before promotion.

**Responsibilities:**

- model readiness status
- diagnostic pass/fail interpretation
- decision-surface prerequisites
- TrustReport requirement checks
- production vs research-only model status
- Ridge / Bayesian / optimizer gate interpretation

| Allowed | Not allowed |
|---------|-------------|
| Summarize governance status | Promote model artifacts |
| Block unsafe promotion (advisory) | Approve budget recommendations |
| Request missing reports / artifacts | Override production gates |
| Explain why a model is research-only or diagnostic-only | Declare Bayesian production-ready if roadmap says research-only |

**Trigger:** Add when model artifacts, decision surfaces, and promotion workflows become productized.

---

## 7. Deferred general agents (optional, not immediate)

| Agent | Trigger condition |
|-------|-------------------|
| **ML Engineering / MLOps Specialist Agent** | Deployment, Docker/API services, schedulers, MLflow/model registry, monitoring, or production refresh pipelines exist |
| **Feature Store Explorer Agent** | MMM integrates with a production feature store (Feast, Tecton, Databricks Feature Store, or equivalent) |
| **Research Scout Agent** | Core MMM workflows are stable; scouts new MMM/causal/calibration methods and proposes research intake items |
| **Privacy / Security Review Agent** | Before persistent uploads, platform-managed LLM keys, multi-user workspaces, or production customer data handling |

These are **MIP-orchestrated** candidates; MMM package-side agents focus on §6 roles first.

---

## 8. Examples

### Example 1 — Missing spend weeks

**Scenario:** MMM refresh fails because Meta spend is missing for six weeks.

**MMM Refresh Failure Agent** reads `MMMFailurePacket`, explains the gap, and proposes safe options:

| Safe options | Blocked |
|--------------|---------|
| Ask user to provide spend | Silently impute spend |
| Confirm true zero spend with explicit record | Continue without recording assumption |
| Exclude channel with governed warning if policy allows | Recommend budget from failed run |

---

### Example 2 — Calibration conflict

**Scenario:** Two CalibrationSignals conflict on channel lift scale or estimand.

**MMM Calibration Reconciliation Agent** explains the conflict via `MMMCalibrationReconciliationReport` and routes to evidence review / TrustReport.

| Safe options | Blocked |
|--------------|---------|
| Surface conflict, stale flags, estimand mismatch | Average conflicting signals silently |
| Request missing uncertainty / scope | Use stale evidence without warning |
| Escalate to human review | Override mapping report status |

---

### Example 3 — Bayesian model request

**Scenario:** User asks to use Bayesian MMM in production.

**MMM Model Governance Agent** explains current production/research status (Bayes-H5 research-only; Ridge production path).

| Safe options | Blocked |
|--------------|---------|
| Explain research-only status and prerequisites | Promote Bayesian model if governance says research-only |
| Point to diagnostic / shadow artifacts | Generate decision recommendation from research artifact |
| Request TrustReport and promotion gate completion | Bypass `approved_for_prod` |

---

## 9. Acceptance criteria (roadmap capture)

| Criterion | Status |
|-----------|--------|
| Future MMM package-side agents documented as **deferred** | ✅ This document |
| Agent boundaries preserve MMM package as deterministic/statistical engine | ✅ §2 |
| Agents cannot alter modeling assumptions, priors, calibration status, or production gates | ✅ §2, §6 |
| Run manifest / failure packet prerequisites documented | ✅ §4, §5 |
| Five package-side agents described | ✅ §6.1–6.5 |
| Deferred general agents with trigger conditions | ✅ §7 |
| Examples: missing spend, calibration conflict, Bayesian governance | ✅ §8 |

---

## 10. Recommended next phase (not agents)

Per [AUDIT-MIP-C6](../audits/AUDIT-MIP-C6_INTEGRATION_READINESS_CHECKPOINT.md), the near-term program priority is **not** package-side agents:

1. **GeoX estimator / inference OC** (unified MIP/GeoX program) — evidence producer quality.
2. **MMM deterministic contracts** — `MMMRunManifest`, `MMMFailurePacket`, typed validation (medium term).
3. **C6 production scheduler governance** — only if operational urgency (deferred by default).

Package-side agents remain **later** until structured failure/diagnostic contracts are stable and repeated operator pain justifies interpretive automation.

---

## 11. Related investigations

- [INV-039](../06_investigations/open_investigations.md#inv-039--auto-retrain-auto-promotion-agentic-orchestration-out-of-v1-scope) — auto-retrain / agentic orchestration out of v1 scope
- [INV-MMM-AGENTS-ROADMAP](../06_investigations/open_investigations.md#inv-mmm-agents-roadmap--future-package-side-support-agents-roadmap-capture) — this roadmap capture
