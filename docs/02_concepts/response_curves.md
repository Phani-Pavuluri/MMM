# Response curves

Curves are generated on the **transformed** spend path (adstock then saturation) to avoid double-applying transforms in the linear predictor.

`mmm.decomposition.curves.build_curve_for_channel` evaluates steady-state spend grids and finite-difference marginal ROI. Extend with geo-specific grids as needed.
