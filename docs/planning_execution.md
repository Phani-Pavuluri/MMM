# Planning execution (developer)

```
training panel (sorted geo × week)
    ↓
counterfactual panel copy
    ↓
media spend overwrite (constant | per-geo | piecewise path)
    ↓
optional ControlOverlaySpec (sparse geo×week×column values)
    ↓
build_design_matrix (adstock + saturation + control_columns from panel)
    ↓
predict_ridge (fixed coef from ridge_fit_summary)
    ↓
aggregate mean μ → Δμ = μ(plan) − μ(baseline)
```

**Optimize path:** SLSQP varies media spend vector; each evaluation calls the same `simulate()` with optional fixed `OptimizeNonMediaContext` overlays.

**Not supported (yet):** multi-world `E[Δμ]`, optimizing controls, generated macro/promo calendars without panel columns.
