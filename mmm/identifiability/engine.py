"""E1: design-matrix identifiability and attribution stability signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from mmm.config.extensions import IdentifiabilityRunConfig
from mmm.models.ridge_bo.ridge import fit_ridge


@dataclass
class IdentifiabilityReport:
    identifiability_score: float
    instability_score: float
    condition_number: float
    effective_rank: int
    max_vif: float
    vif_by_channel: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    correlation_matrix: list[list[float]] | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _vif_numpy(X: np.ndarray, names: list[str]) -> dict[str, float]:
    """Gaussian VIF via R^2 of each column on others (ridge-stabilized)."""
    n, p = X.shape
    if p < 2:
        return {names[0]: 1.0} if p == 1 else {}
    x = X - X.mean(axis=0)
    out: dict[str, float] = {}
    ridge = 1e-6
    for j in range(p):
        y = x[:, j]
        Z = np.delete(x, j, axis=1)
        a = Z.T @ Z + ridge * np.eye(Z.shape[1])
        b = Z.T @ y
        coef = np.linalg.solve(a, b)
        pred = Z @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        vif = 1.0 / max(1e-9, 1.0 - min(r2, 0.999999))
        out[names[j]] = float(vif)
    return out


def _effective_rank(X: np.ndarray, tol: float = 1e-10) -> int:
    s = np.linalg.svd(X, compute_uv=False)
    if s.size == 0:
        return 0
    return int(np.sum(s > tol * s[0]))


class IdentifiabilityEngine:
    """E1: compute collinearity, VIF, conditioning, bootstrap coefficient dispersion."""

    def __init__(self, cfg: IdentifiabilityRunConfig | None = None) -> None:
        self.cfg = cfg or IdentifiabilityRunConfig()

    def analyze(
        self,
        X_media: np.ndarray,
        channel_names: list[str],
        y_log: np.ndarray,
        rng: np.random.Generator,
        *,
        ridge_log_alpha: float | None = None,
    ) -> IdentifiabilityReport:
        warnings: list[str] = []
        X = np.asarray(X_media, dtype=float)
        n, p = X.shape
        if n <= p + 2:
            warnings.append("n_obs close to n_features; identifiability weak by construction")
        corr = np.corrcoef(X.T)
        cn = float(np.linalg.cond(X)) if p else 1.0
        er = _effective_rank(X)
        vifs = _vif_numpy(X, channel_names)
        max_vif = max(vifs.values()) if vifs else 1.0
        if max_vif > self.cfg.vif_threshold:
            warnings.append(f"high_collinearity: max VIF {max_vif:.1f} > {self.cfg.vif_threshold}")
        if cn > self.cfg.condition_threshold:
            warnings.append(f"ill_conditioned: cond(X)={cn:.2e} > {self.cfg.condition_threshold}")
        if er < max(1, int(0.7 * p)):
            warnings.append("low_effective_rank_vs_channels: possible many equivalent solutions")

        alpha_fit = float(10 ** float(ridge_log_alpha if ridge_log_alpha is not None else 0.0))
        boot_std: list[float] = []
        rounds = max(0, self.cfg.bootstrap_rounds)
        m = max(int(self.cfg.bootstrap_frac * n), p + 2)
        for _ in range(rounds):
            idx = rng.choice(n, size=m, replace=True)
            Xb, yb = X[idx], y_log[idx]
            coef, _ = fit_ridge(Xb, yb, alpha=alpha_fit)
            boot_std.append(float(np.mean(np.abs(coef))))
        instability = float(np.std(boot_std)) if boot_std else 0.0
        if instability > 0.15 * (np.mean(boot_std) + 1e-9):
            warnings.append("unstable_attribution: bootstrap ridge coef dispersion high")

        risk_vif = min(1.0, max_vif / (2 * self.cfg.vif_threshold))
        risk_cond = min(1.0, cn / (2 * self.cfg.condition_threshold))
        risk_rank = 1.0 - (er / max(p, 1))
        ident_score = float(np.clip(0.4 * risk_vif + 0.35 * risk_cond + 0.25 * risk_rank, 0.0, 1.0))
        inst_score = float(np.clip(instability / (np.mean(boot_std) + 1e-6), 0.0, 1.0)) if boot_std else 0.0

        return IdentifiabilityReport(
            identifiability_score=ident_score,
            instability_score=inst_score,
            condition_number=cn,
            effective_rank=er,
            max_vif=max_vif,
            vif_by_channel=vifs,
            warnings=warnings,
            correlation_matrix=corr.tolist(),
        )

    def posterior_correlation_summary(self, posterior_draws: np.ndarray | None) -> dict[str, Any] | None:
        """If draws shape (samples, p), return mean abs corr; else None (E1 Bayesian hook)."""
        if posterior_draws is None or posterior_draws.ndim != 2:
            return None
        c = np.corrcoef(posterior_draws.T)
        off = c[np.triu_indices_from(c, k=1)]
        return {"mean_abs_posterior_corr": float(np.mean(np.abs(off)))}
