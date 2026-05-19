# Response curves

**Curves explain; full-panel simulation decides.** Univariate response curves and marginal ROI grids are diagnostic surfaces. Budget and scenario decisions must use full-panel Δμ from `mmm decide simulate` / `optimize-budget`, not curve increments alone. See `curve_decision_alignment` in extension artifacts and `tests/test_curve_decision_alignment.py`.

Curves are generated on the **transformed** spend path (adstock then saturation) to avoid double-applying transforms in the linear predictor.

`mmm.decomposition.curves.build_curve_for_channel` evaluates steady-state spend grids and finite-difference marginal ROI. Extend with geo-specific grids as needed.
