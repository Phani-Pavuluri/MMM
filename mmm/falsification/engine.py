"""E6: falsification checks for spurious attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmm.config.extensions import FalsificationConfig
from mmm.data.schema import PanelSchema
from mmm.models.ridge_bo.ridge import fit_ridge


@dataclass
class FalsificationReport:
    placebo_channel_coef_mean: float
    flags: list[str] = field(default_factory=list)
    tests: dict[str, Any] = field(default_factory=dict)
    null_calibration_method: str = "fixed_scale_floor"
    null_metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        families = sorted(self.tests.keys()) if isinstance(self.tests, dict) else []
        return {
            "placebo_channel_coef_mean": self.placebo_channel_coef_mean,
            "flags": self.flags,
            "tests": self.tests,
            "null_calibration_method": self.null_calibration_method,
            "null_metadata": dict(self.null_metadata),
            "placebo_families_run": families,
            "interpretation": (
                "Falsification is a screening battery under explicit null designs; it is not proof of causality. "
                "Flags indicate elevated sensitivity to a particular placebo family and must be triaged with "
                "domain context — they do not alone invalidate a production release."
            ),
        }


class FalsificationEngine:
    def __init__(self, schema: PanelSchema, cfg: FalsificationConfig) -> None:
        self.schema = schema
        self.cfg = cfg

    def run(
        self,
        X_media: np.ndarray,
        y_log: np.ndarray,
        rng: np.random.Generator,
        *,
        ridge_log_alpha: float | None = None,
        geo_ids: np.ndarray | None = None,
    ) -> FalsificationReport:
        flags: list[str] = []
        tests: dict[str, Any] = {}
        if not self.cfg.enabled or self.cfg.placebo_draws <= 0:
            return FalsificationReport(
                0.0,
                flags,
                tests,
                null_calibration_method=str(getattr(self.cfg, "null_calibration_method", "fixed_scale_floor")),
                null_metadata={},
            )
        alpha = float(10 ** float(ridge_log_alpha if ridge_log_alpha is not None else 0.0))
        tests["ridge_alpha_used"] = alpha
        tests["ridge_log_alpha_passed"] = ridge_log_alpha
        n = X_media.shape[0]
        p = X_media.shape[1] if X_media.ndim == 2 else 0

        # --- 1) Gaussian noise column(s), same regularization as selected model ---
        coefs_single: list[float] = []
        for _ in range(self.cfg.placebo_draws):
            noise = rng.normal(0, 1.0, size=(n, 1))
            Xn = np.hstack([X_media, noise])
            c, _ = fit_ridge(Xn, y_log, alpha=alpha)
            coefs_single.append(float(c[-1]))
        mean_noise = float(np.mean(np.abs(coefs_single)))
        cref, _ = fit_ridge(X_media, y_log, alpha=alpha)
        scale = float(np.mean(np.abs(cref)) + 1e-9)
        # Scale-aware null: combine fixed floor, relative-to-signal scale, and mild p/n dependence.
        p_feat = float(max(X_media.shape[1], 1))
        n_obs = float(max(X_media.shape[0], 1))
        null_floor = max(1e-6, 0.02 * scale, 0.04 * scale * (p_feat / n_obs) ** 0.5)
        tests["gaussian_noise_placebo"] = {
            "mean_abs_coef_on_noise": mean_noise,
            "reference_scale_mean_abs_media_coef": scale,
            "n_draws": int(self.cfg.placebo_draws),
            "null_threshold_used": null_floor,
        }
        if mean_noise > null_floor:
            flags.append("spurious_attribution_risk: placebo channel absorbs signal")

        # --- 2) Optional dual-noise: two independent columns ---
        if self.cfg.dual_noise_placebo and self.cfg.placebo_draws > 0:
            dual_a: list[float] = []
            dual_b: list[float] = []
            for _ in range(min(self.cfg.placebo_draws, 8)):
                n1 = rng.normal(0, 1.0, size=(n, 1))
                n2 = rng.normal(0, 1.0, size=(n, 1))
                Xd = np.hstack([X_media, n1, n2])
                cd, _ = fit_ridge(Xd, y_log, alpha=alpha)
                dual_a.append(float(cd[-2]))
                dual_b.append(float(cd[-1]))
            ma = float(np.mean(np.abs(dual_a))) if dual_a else 0.0
            mb = float(np.mean(np.abs(dual_b))) if dual_b else 0.0
            tests["dual_gaussian_noise_placebo"] = {
                "mean_abs_coef_noise_a": ma,
                "mean_abs_coef_noise_b": mb,
                "reference_scale_mean_abs_media_coef": scale,
            }
            if max(ma, mb) > 0.05 * scale:
                flags.append("spurious_attribution_risk: dual_noise_placebo_absorbs_signal")

        # --- 3) Random permutation of media columns (fungibility / collinearity) ---
        if self.cfg.media_column_permutation_placebo and p >= 2:
            perm = rng.permutation(p)
            X_perm = np.asarray(X_media[:, perm], dtype=float, order="C")
            cperm, _ = fit_ridge(X_perm, y_log, alpha=alpha)
            c0, _ = fit_ridge(X_media, y_log, alpha=alpha)
            rel = float(np.linalg.norm(cperm - c0) / (np.linalg.norm(c0) + 1e-9))
            tests["media_column_permutation_placebo"] = {
                "relative_coef_l2_change_vs_unpermuted": rel,
                "permutation": perm.tolist(),
            }
            if rel < 0.02 and scale > 1e-6:
                flags.append("collinearity_or_symmetry_risk: media_coef_vector nearly invariant to column shuffle")

        # --- 4) Time-shifted media (misaligned timing vs outcome) ---
        if self.cfg.time_shifted_media_placebo and p >= 1 and n >= 8:
            shift = int(rng.integers(max(1, n // 50), max(2, n // 5)))
            X_shift = np.roll(np.asarray(X_media, dtype=float), shift=shift, axis=0)
            cshift, _ = fit_ridge(X_shift, y_log, alpha=alpha)
            mean_shift = float(np.mean(np.abs(cshift)))
            tests["time_shifted_media_placebo"] = {
                "row_shift": shift,
                "mean_abs_coef_on_shifted_design": mean_shift,
                "reference_scale_mean_abs_media_coef": scale,
            }
            if mean_shift > 0.12 * scale:
                flags.append("temporal_alignment_risk: shifted_media_design_still_explains_outcome")

        # --- 5) Grouped-channel placebo (two random aggregate columns) ---
        if self.cfg.grouped_channel_placebo and p >= 4:
            perm = rng.permutation(p)
            half = p // 2
            idx_a = perm[:half]
            idx_b = perm[half:]
            g1 = np.sum(X_media[:, idx_a], axis=1, keepdims=True)
            g2 = np.sum(X_media[:, idx_b], axis=1, keepdims=True)
            Xg = np.hstack([g1, g2])
            cg, _ = fit_ridge(Xg, y_log, alpha=alpha)
            mg = float(np.mean(np.abs(cg)))
            tests["grouped_channel_placebo"] = {
                "n_channels_group_a": int(idx_a.size),
                "n_channels_group_b": int(idx_b.size),
                "mean_abs_coef_on_two_group_features": mg,
                "reference_scale_mean_abs_media_coef": scale,
            }
            if mg > 0.2 * scale:
                flags.append("aggregation_risk: pooled_channel_groups_absorb_outcome_like_full_media")

        # --- 6) Within-geo row shuffle of media (requires geo_ids) — research-grade surface ---
        if (
            self.cfg.within_geo_media_row_shuffle_placebo
            and geo_ids is not None
            and len(geo_ids) == n
            and p >= 1
        ):
            Xw = np.asarray(X_media, dtype=float, order="C").copy()
            gid = np.asarray(geo_ids, dtype=object)
            for g in np.unique(gid):
                sel = np.where(gid == g)[0]
                if sel.size < 3:
                    continue
                order = sel.copy()
                rng.shuffle(order)
                Xw[sel, :] = X_media[order, :]
            cw, _ = fit_ridge(Xw, y_log, alpha=alpha)
            mw = float(np.mean(np.abs(cw)))
            tests["within_geo_media_row_shuffle_placebo"] = {
                "mean_abs_coef_after_within_geo_shuffle": mw,
                "reference_scale_mean_abs_media_coef": scale,
            }
            if mw > 0.12 * scale:
                flags.append("within_geo_timing_risk: shuffled_media_within_geo_still_fits")

        # --- 7) Held-out geo coef transfer (research-only; needs >=2 geos) ---
        if self.cfg.geo_split_coef_transfer_placebo and geo_ids is not None and len(geo_ids) == n and p >= 2:
            ugeo = np.unique(np.asarray(geo_ids, dtype=str))
            if ugeo.size >= 2:
                mask = rng.random(size=ugeo.size) < 0.5
                train_geos = set(ugeo[mask].tolist()) if mask.any() else {str(ugeo[0])}
                if not train_geos:
                    train_geos = {str(ugeo[0])}
                tr_idx = np.array([str(g) in train_geos for g in np.asarray(geo_ids, dtype=str)], dtype=bool)
                te_idx = ~tr_idx
                if int(np.sum(tr_idx)) > p + 2 and int(np.sum(te_idx)) > p + 2:
                    ctr, _ = fit_ridge(X_media[tr_idx], y_log[tr_idx], alpha=alpha)
                    pred_te = X_media[te_idx] @ ctr
                    rmse = float(np.sqrt(np.mean((pred_te - y_log[te_idx]) ** 2)))
                    var_te = float(np.var(y_log[te_idx]) + 1e-9)
                    r2_like = 1.0 - (rmse**2) / var_te
                    tests["geo_split_coef_transfer_placebo"] = {
                        "n_train_rows": int(np.sum(tr_idx)),
                        "n_test_rows": int(np.sum(te_idx)),
                        "pseudo_r2_on_holdout_geos": float(r2_like),
                    }
                    if r2_like > 0.25:
                        flags.append("geo_generalization_risk: coefs_from_random_geo_subset_fit_holdout_geos_well")

        null_method = str(getattr(self.cfg, "null_calibration_method", "fixed_scale_floor"))
        null_meta: dict[str, Any] = {}
        if (
            null_method == "empirical_y_permutation"
            and n >= 10
            and p >= 1
            and int(self.cfg.placebo_draws) >= 2
            and self.cfg.enabled
        ):
            n_perm = int(min(int(getattr(self.cfg, "empirical_null_n_permutations", 24)), 64))
            perm_means: list[float] = []
            inner = int(min(int(self.cfg.placebo_draws), 5))
            for _ in range(n_perm):
                yp = rng.permutation(np.asarray(y_log, dtype=float).copy())
                vals: list[float] = []
                for __ in range(inner):
                    noise = rng.normal(0, 1.0, size=(n, 1))
                    Xn = np.hstack([X_media, noise])
                    c, _ = fit_ridge(Xn, yp, alpha=alpha)
                    vals.append(float(abs(c[-1])))
                perm_means.append(float(np.mean(vals)))
            perm_arr = np.asarray(perm_means, dtype=float)
            q95 = float(np.quantile(perm_arr, 0.95))
            tail_freq = float(np.mean(perm_arr >= mean_noise))
            null_meta = {
                "n_permutations": n_perm,
                "inner_placebo_draws_per_permutation": inner,
                "observed_mean_abs_noise_coef": float(mean_noise),
                "empirical_tail_frequency_vs_null_distribution": tail_freq,
                "empirical_p_value_upper_tail_proxy": float(tail_freq),
                "empirical_quantile_95_of_null_abs_noise_coef_means": q95,
                "threshold_basis": "upper_95_percentile_of_permuted_y_null_distribution_of_mean_abs_noise_coef",
            }
            tests["empirical_y_permutation_null"] = null_meta
            if mean_noise > q95:
                flags.append(
                    "spurious_attribution_risk_empirical: placebo_coef_exceeds_permuted_y_null_distribution_q95"
                )

        return FalsificationReport(
            placebo_channel_coef_mean=mean_noise,
            flags=flags,
            tests=tests,
            null_calibration_method=null_method,
            null_metadata=null_meta,
        )
