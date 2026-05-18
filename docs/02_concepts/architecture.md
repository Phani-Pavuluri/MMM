# Architecture

Layers and responsibilities:

- **data**: schema validation, loading, preprocessing, holdouts.
- **transforms**: adstock, saturation, scaling; registry + plugin-friendly base classes.
- **models**: `RidgeBOMMMTrainer` (inner ridge, outer BO) and `BayesianMMMTrainer` (PyMC; Stan optional stub).
- **hierarchy**: pooling specifications and index helpers for partial pooling.
- **validation**: rolling / expanding CV with auto mode selection.
- **calibration**: experiment schema, multi-level matching, BO loss, Bayesian likelihood hooks.
- **decomposition / curves**: contributions and marginal ROI grids on the transformed spend path.
- **optimization**: budget SQP-style optimizer with channel bounds.
- **artifacts**: `ArtifactStoreBase` with `LocalArtifactStore` default and lazy `MLflowArtifactStore`.
- **reporting**: JSON-first `ReportBuilder`.
- **api / cli**: aligned entrypoints for YAML-driven and programmatic use.
