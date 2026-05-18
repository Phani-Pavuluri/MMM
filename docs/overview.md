# Package overview

`mmm` is a weekly geo-level Marketing Mix Modeling library designed for production workflows: strict data contracts, config-driven runs, pluggable artifacts, two modeling frameworks (Bayesian MCMC and Ridge + Bayesian Optimization), optional experiment calibration, decomposition, response curves, constrained budget optimization, and reporting.

Design goals: reproducibility (resolved configs + seeds), safe defaults (local artifacts, lazy optional integrations), and explicit objective decomposition for the optimization-based path.

**Planning contract:** `mmm decide simulate` changes media on the training panel with optional sparse control overlays; non-media defaults to observed historical values. `mmm decide optimize-budget` optimizes media only under one fixed non-media world. **How-to:** [planning_howto.md](planning_howto.md). Reference: [decision_runbook.md](decision_runbook.md) §2e, [planning_artifact_schema.md](planning_artifact_schema.md), `mmm/planning/scenario.py`.
