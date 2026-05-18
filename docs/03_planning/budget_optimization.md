# Budget optimization

`BudgetOptimizer` maximizes a concave proxy objective under linear budget and box constraints using `scipy.optimize.minimize` (SLSQP). Replace the response proxy with model-implied curves from `DecompositionEngine` / posterior draws for production decisioning.

Risk-aware extensions: pass P10/P90 marginal ROI vectors derived from Bayesian samples as bounds or weights.
