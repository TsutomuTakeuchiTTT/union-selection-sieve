"""High-level public API for the union-selection rectangular-sieve estimator.

The numerical core is the validated Version 1.1 implementation used for the
confirmatory calculations in the accompanying manuscript.  This module adds a
small, stable interface for fitting an observed catalogue and exporting the
result without changing the likelihood or optimization algorithm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd

from . import _reference_impl as _impl


REQUIRED_COLUMNS = ("y1", "y2", "delta1", "delta2", "x1_obs", "x2_obs")


@dataclass(frozen=True)
class FitConfig:
    """Numerical controls for a fixed rectangular sieve fit."""

    coverage_tolerance: float = 1.0e-15
    mass_tolerance: float = 1.0e-12
    score_tolerance: float = 1.0e-9
    prune_mass_tolerance: float = 1.0e-7
    prune_score_margin: float = 1.0e-7
    max_iterations: int = 30_000
    warm_mixing: float = 1.0e-9
    mm_prepolish_iterations: int = 2_000
    newton_tolerance: float = 1.0e-12
    polish_zero_tolerance: float = 1.0e-13
    max_polish_cycles: int = 8
    direct_audit_starts: int = 0
    direct_audit_seed: int = 104_729
    run_uniform_start_audit: bool = False


@dataclass
class SieveEstimate:
    """A fitted distribution together with identification and KKT diagnostics."""

    grid: _impl.RectangularGrid
    mass: np.ndarray
    class_mass: np.ndarray
    classes: Sequence[Sequence[int]]
    membership_audit: pd.DataFrame
    cell_screening: pd.DataFrame
    class_table: pd.DataFrame
    diagnostics: Dict[str, Any]
    fit_config: FitConfig

    def density(self, x1: float, x2: float) -> float:
        return self.grid.density(self.mass, x1, x2)

    def cdf(self, x1: float, x2: float) -> float:
        return self.grid.cdf(self.mass, x1, x2)

    def cdf_envelope(self, x1: float, x2: float) -> Tuple[float, float, float]:
        """Return lower, equal-split, and upper CDF values at one query point."""
        coeff = self.grid.cdf_row(float(x1), float(x2))
        lower = 0.0
        equal = 0.0
        upper = 0.0
        for theta, cls_raw in zip(self.class_mass, self.classes):
            cls = list(cls_raw)
            values = coeff[cls]
            lower += float(theta) * float(values.min())
            equal += float(theta) * float(values.mean())
            upper += float(theta) * float(values.max())
        return lower, equal, upper

    def cell_masses(self) -> pd.DataFrame:
        table = self.grid.cell_table.copy()
        table["estimated_mass"] = np.asarray(self.mass, dtype=float)
        return table

    def cdf_table(self, query_grid_size: int = 41) -> pd.DataFrame:
        if query_grid_size < 2:
            raise ValueError("query_grid_size must be at least two")
        x1_values = np.linspace(self.grid.edges1[0], self.grid.edges1[-1], query_grid_size)
        x2_values = np.linspace(self.grid.edges2[0], self.grid.edges2[-1], query_grid_size)
        rows = []
        for x1 in x1_values:
            for x2 in x2_values:
                lower, equal, upper = self.cdf_envelope(float(x1), float(x2))
                rows.append(
                    {
                        "x1": float(x1),
                        "x2": float(x2),
                        "cdf_lower": lower,
                        "cdf_equal_split": equal,
                        "cdf_upper": upper,
                        "identification_width": upper - lower,
                    }
                )
        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, Any]:
        mm = dict(self.diagnostics.get("mm", {}))
        return {
            "n_cells": int(self.grid.k),
            "n_equivalence_classes": int(len(self.classes)),
            "n_nontrivial_equivalence_classes": int(sum(len(c) > 1 for c in self.classes)),
            "n_forced_zero_cells": int(self.diagnostics.get("n_forced_zero_cells", 0)),
            "n_invisible_cells": int(self.diagnostics.get("n_invisible_cells", 0)),
            "probability_mass_sum": float(np.sum(self.mass)),
            "minimum_mass": float(np.min(self.mass)),
            "maximum_mass": float(np.max(self.mass)),
            "selected_start": self.diagnostics.get("selected_start"),
            "converged": bool(mm.get("converged", False)),
            "final_loglikelihood": float(mm.get("final_loglikelihood", np.nan)),
            "active_score_residual": float(mm.get("active_score_sup_abs_residual", np.nan)),
            "inactive_positive_score_violation": float(
                mm.get("inactive_positive_score_violation", np.nan)
            ),
            "hybrid_polish_used": bool(mm.get("hybrid_polish_used", False)),
            "polish_cycles": int(mm.get("polish_cycles", 0)),
            "fit_config": asdict(self.fit_config),
        }

    def write(self, outdir: str | Path, query_grid_size: int = 41) -> Dict[str, Path]:
        """Write a compact, auditable set of fitted outputs."""
        root = Path(outdir)
        root.mkdir(parents=True, exist_ok=True)
        paths = {
            "cell_masses": root / "cell_masses.csv",
            "cdf_grid": root / "cdf_grid.csv",
            "membership_audit": root / "operator_membership_audit.csv",
            "cell_screening": root / "cell_screening.csv",
            "equivalence_classes": root / "equivalence_classes.csv",
            "diagnostics": root / "fit_diagnostics.json",
        }
        self.cell_masses().to_csv(paths["cell_masses"], index=False)
        self.cdf_table(query_grid_size=query_grid_size).to_csv(paths["cdf_grid"], index=False)
        self.membership_audit.to_csv(paths["membership_audit"], index=False)
        self.cell_screening.to_csv(paths["cell_screening"], index=False)
        self.class_table.to_csv(paths["equivalence_classes"], index=False)
        paths["diagnostics"].write_text(
            json.dumps(_jsonable(self.summary()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return paths


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def prepare_observed_catalogue(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a user catalogue to the columns required by the likelihood.

    Required information is ``y1``, ``y2``, ``delta1``, ``delta2`` and the
    observed value of every detected coordinate.  The input may provide the
    latter either as ``x1_obs``/``x2_obs`` or as ``x1``/``x2``.  Values for a
    censored coordinate may be missing; they are replaced by its threshold
    because that placeholder is ignored by the corresponding subdensity row.
    Region-D rows are rejected because they are absent from a union-selected
    catalogue by definition.
    """
    frame = data.copy()
    for canonical, alternate in (("x1_obs", "x1"), ("x2_obs", "x2")):
        if canonical not in frame.columns and alternate in frame.columns:
            frame[canonical] = frame[alternate]
    missing = {"y1", "y2", "delta1", "delta2"}.difference(frame.columns)
    if missing:
        raise ValueError(f"catalogue is missing required columns: {sorted(missing)}")
    if "x1_obs" not in frame.columns:
        frame["x1_obs"] = np.nan
    if "x2_obs" not in frame.columns:
        frame["x2_obs"] = np.nan

    for col in ("y1", "y2", "x1_obs", "x2_obs"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ("delta1", "delta2"):
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(int)
        if not frame[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0 and 1")

    if (~np.isfinite(frame["y1"]) | ~np.isfinite(frame["y2"])).any():
        raise ValueError("all thresholds must be finite")
    region_d = (frame["delta1"] == 0) & (frame["delta2"] == 0)
    if region_d.any():
        raise ValueError("Region-D rows must not be present in the observed catalogue")
    missing_x1 = (frame["delta1"] == 1) & ~np.isfinite(frame["x1_obs"])
    missing_x2 = (frame["delta2"] == 1) & ~np.isfinite(frame["x2_obs"])
    if missing_x1.any() or missing_x2.any():
        raise ValueError("every detected coordinate must have a finite observed value")

    frame.loc[frame["delta1"] == 0, "x1_obs"] = frame.loc[
        frame["delta1"] == 0, "y1"
    ]
    frame.loc[frame["delta2"] == 0, "x2_obs"] = frame.loc[
        frame["delta2"] == 0, "y2"
    ]
    frame["region"] = np.select(
        [
            (frame["delta1"] == 1) & (frame["delta2"] == 1),
            (frame["delta1"] == 1) & (frame["delta2"] == 0),
            (frame["delta1"] == 0) & (frame["delta2"] == 1),
        ],
        ["A", "B", "C"],
        default="D",
    )
    return frame.reset_index(drop=True)


def uniform_grid(
    bins_per_axis: int,
    lower1: float,
    upper1: float,
    lower2: Optional[float] = None,
    upper2: Optional[float] = None,
) -> _impl.RectangularGrid:
    if bins_per_axis < 1:
        raise ValueError("bins_per_axis must be positive")
    lower2 = lower1 if lower2 is None else lower2
    upper2 = upper1 if upper2 is None else upper2
    return _impl.RectangularGrid(
        np.linspace(float(lower1), float(upper1), int(bins_per_axis) + 1),
        np.linspace(float(lower2), float(upper2), int(bins_per_axis) + 1),
    )


def fit_catalogue(
    data: pd.DataFrame,
    grid: _impl.RectangularGrid,
    *,
    config: Optional[FitConfig] = None,
    row_weights: Optional[np.ndarray] = None,
    warm_mass: Optional[np.ndarray] = None,
) -> SieveEstimate:
    """Fit the union-selection conditional likelihood on a fixed grid."""
    cfg = config or FitConfig()
    observed = prepare_observed_catalogue(data)
    a, b, membership = _impl.build_sample_operator(observed, grid)
    fit = _impl.fit_sieve_with_start_audit(
        a,
        b,
        row_weights,
        coverage_tolerance=cfg.coverage_tolerance,
        mass_tolerance=cfg.mass_tolerance,
        score_tolerance=cfg.score_tolerance,
        prune_mass_tolerance=cfg.prune_mass_tolerance,
        prune_score_margin=cfg.prune_score_margin,
        max_iterations=cfg.max_iterations,
        warm_original_mass=warm_mass,
        run_uniform_audit=cfg.run_uniform_start_audit,
        warm_mixing=cfg.warm_mixing,
        direct_starts=cfg.direct_audit_starts,
        direct_seed=cfg.direct_audit_seed,
        mm_prepolish_iterations=cfg.mm_prepolish_iterations,
        newton_tolerance=cfg.newton_tolerance,
        polish_zero_tolerance=cfg.polish_zero_tolerance,
        max_polish_cycles=cfg.max_polish_cycles,
    )
    if not fit.selected_fit.converged:
        mm = fit.selected_fit.diagnostics
        raise RuntimeError(
            "fit did not satisfy the requested KKT tolerance: "
            f"active={mm.get('active_score_sup_abs_residual')}, "
            f"inactive={mm.get('inactive_positive_score_violation')}"
        )
    return SieveEstimate(
        grid=grid,
        mass=np.asarray(fit.original_mass, dtype=float),
        class_mass=np.asarray(fit.selected_fit.mass, dtype=float),
        classes=fit.reduction.classes,
        membership_audit=membership,
        cell_screening=fit.reduction.cell_screening,
        class_table=fit.reduction.class_table,
        diagnostics=fit.diagnostics,
        fit_config=cfg,
    )


__all__ = [
    "FitConfig",
    "SieveEstimate",
    "fit_catalogue",
    "prepare_observed_catalogue",
    "uniform_grid",
]
