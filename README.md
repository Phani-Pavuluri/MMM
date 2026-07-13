# MMM — Geo-level marketing mix modeling for measurement and budget decisions

A weekly **geo-level** MMM and planning system for causal measurement, experiment-informed calibration, and **decision-grade** budget recommendations—not a loose toolkit of model scripts.

---

## What this system answers

- **What is the incremental impact of media?** Full-panel counterfactuals on the trained model, not curve extrapolation alone.
- **How should budgets change?** Constrained optimization scored on the same model and panel used at fit time.
- **Which outputs are safe for decisions?** Explicit decision bundles, governance gates, and disclosures—not every chart is decision truth.
- **How do experiments and MMM reconcile?** Matched geo experiments feed calibration and replay checks; mismatches surface in diagnostics, not silently in coefficients.

---

## Core principles

### Decision truth = full-panel Δμ

Budget and scenario recommendations are scored with **full-panel simulation** on the **same** trained model as fit:

- Same design matrix, transforms, recursive adstock, and panel structure  
- Same control and media semantics as training  

**Response curves** and **decomposition** help explain and stress-test the model; they are **diagnostic**, not the primary allocator for production decisions.

### Experiment-informed calibration

- Geo-level experiment observations with configurable match levels (geo, channel, time, device, product)  
- Objective and replay paths that integrate experiment evidence where configured  
- Calibration admissibility and matching traces in extension reports  

### Governance-first

- **Decision bundles** with artifact tiers and semantic contracts  
- **Lineage** (config, data fingerprint, dependency lock) for audit  
- **Planning assumptions** and scenario lineage on simulate / optimize paths  
- **Unsupported-question** disclosures when a claim is not supported  
- **Release gates** (identifiability, panel QA, falsification, operational health) before optimization is advised  

---

## Workflow

```text
Train → Validate → Calibrate → Diagnostics → Simulate scenarios → Optimize budget → Publish decision bundle
```

| Stage | Purpose |
|--------|---------|
| **Train** | Fit Ridge+BO or Bayesian MMM on weekly geo panel |
| **Validate** | CV, post-fit checks, panel QA |
| **Calibrate** | Match and weight experiments; replay calibration when enabled |
| **Diagnostics** | Identifiability, separability, falsification, operational health |
| **Simulate** | Candidate media plans; optional control overlays; Δμ vs BAU |
| **Optimize** | Media budget under constraints; fixed non-media world when scenario supplied |
| **Publish** | Persist decision-safe JSON bundle for downstream systems |

---

## Major components

### Modeling

- Ridge + Bayesian optimization (composite objective, documented penalties)  
- Bayesian MMM (PyMC; optional Stan path)  
- **Production Ridge+BO path:** geometric adstock + Hill saturation only (`canonical_transforms`); Weibull / log / logistic kinds exist in the transform registry for research stubs but are **not** wired into train/simulate/decide unless explicitly validated  
- Pooling: none, full, or partial (hierarchical)  

### Diagnostics

- Identifiability and collinearity (VIF, conditioning, bootstrap dispersion)  
- Feature separability guidance for split channels (rollup / experiment hints—no auto-merge)  
- Falsification and placebo stress tests  
- Operational health and model-release state  

### Planning

- Full-panel Δμ simulation with explicit planning assumptions  
- Media scenarios and sparse control overlays  
- Budget optimization (national or geo-pooled) on simulation objective  

### Governance

- Artifact and semantics contracts (decision vs diagnostic tier)  
- Lineage and run manifests  
- Optimization safety gates and environment policies (research vs prod)  
- Future package-side support agents (deferred roadmap): [docs/05_validation/mmm_package_side_agents_roadmap.md](docs/05_validation/mmm_package_side_agents_roadmap.md)  
- MMM→MIP export contract inventory (MMM-EXPORT-001): [docs/05_validation/mmm_to_mip_export_contract_inventory.md](docs/05_validation/mmm_to_mip_export_contract_inventory.md)

---

## Quickstart

Requires **Python 3.11+** (3.11 or 3.12 recommended).

```bash
pip install -e ".[bayesian,bo,dev]"
mmm train --config examples/minimal_train.yaml
```

Python API:

```python
from mmm.api import MMMTrainer
from mmm.config.load import load_config

trainer = MMMTrainer.from_yaml("examples/minimal_train.yaml")
result = trainer.run()
```

For decision paths after training, use `mmm decide simulate` and `mmm decide optimize-budget` with a resolved config and extension report—see the planning guide below.

---

## Documentation

**Start here:** [docs/README.md](docs/README.md) — navigation by user journey (getting started, concepts, planning, governance).

| Topic | Guide |
|--------|--------|
| Overview & quickstart | [docs/01_getting_started/](docs/01_getting_started/) |
| Concepts (architecture, calibration, diagnostics) | [docs/02_concepts/](docs/02_concepts/) |
| Planning & budget decisions | [docs/03_planning/planning_howto.md](docs/03_planning/planning_howto.md) |
| Production governance | [docs/04_governance/](docs/04_governance/) |
| **v1.0.0 release** | [CHANGELOG.md](CHANGELOG.md) · [release notes](docs/04_governance/v1_release_notes.md) |
