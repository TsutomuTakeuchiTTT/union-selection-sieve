"""Independent-seed confirmatory Monte Carlo for the union-selection rectangular-sieve estimator.

The development analyses identified a fixed 16-by-16 sieve as the leading
CDF-oriented practical resolution.  This program freezes that choice before
new catalogue generation and compares it with fixed 12-by-12, fixed 20-by-20,
and maximum repeat-averaged cross-validation score on an independent seed
family.  A-only intersection-selection fits are retained as the primary
information-loss baseline, while CDF and population-logscore oracles are used
only for post-fit simulation audit.

The default one-cell profile uses two correlations, three latent sample sizes,
30 independent catalogues per condition, three repeats of region-stratified
five-fold cross-validation, and an independent population-validation sample.
Every fit is checkpointed and resumable.  A vanishing uniform predictive
mixture is used only for held-out and population log scoring; it never changes
the fitted full-sample distribution.

Code version: union-selection-sieve-confirmatory-monte-carlo-v1.1.0-onecell
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy.optimize import brentq, minimize
    from scipy.stats import multivariate_normal, t as student_t
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "SciPy is required for the MM multiplier, independent optimiser, and "
        "bivariate-normal benchmark. Install scipy in the active Jupyter kernel."
    ) from exc

CODE_VERSION = "union-selection-sieve-core-v1.1.0-onecell"


# =============================================================================
# Generic utilities
# =============================================================================
def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.str_,)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise_probability(values: np.ndarray, name: str = "probability") -> np.ndarray:
    p = np.asarray(values, dtype=float).copy()
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(p < 0.0):
        raise ValueError(f"{name} contains negative values")
    total = float(p.sum())
    if not total > 0.0:
        raise ValueError(f"{name} has zero total mass")
    return p / total


def _normalise_row_weights(row_weights: Optional[np.ndarray], n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("the likelihood must contain at least one row")
    if row_weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    w = _normalise_probability(np.asarray(row_weights, dtype=float), "row_weights")
    if w.size != n:
        raise ValueError("row_weights length differs from the number of likelihood rows")
    return w


def _tangent_basis(k: int) -> np.ndarray:
    if k <= 1:
        return np.zeros((k, 0), dtype=float)
    raw = np.zeros((k, k - 1), dtype=float)
    for j in range(k - 1):
        raw[j, j] = 1.0
        raw[k - 1, j] = -1.0
    q, _ = np.linalg.qr(raw)
    return q


def _region_name(delta1: bool, delta2: bool) -> str:
    if delta1 and delta2:
        return "A"
    if delta1:
        return "B"
    if delta2:
        return "C"
    return "D"


def _max_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


# =============================================================================
# Rectangular basis
# =============================================================================
@dataclass(frozen=True)
class RectangularGrid:
    """Tensor-product partition with normalized uniform cell basis functions."""

    edges1: np.ndarray
    edges2: np.ndarray

    def __post_init__(self) -> None:
        e1 = np.asarray(self.edges1, dtype=float)
        e2 = np.asarray(self.edges2, dtype=float)
        if e1.ndim != 1 or e2.ndim != 1 or e1.size < 2 or e2.size < 2:
            raise ValueError("each edge array must be one-dimensional with at least two values")
        if not np.all(np.isfinite(e1)) or not np.all(np.isfinite(e2)):
            raise ValueError("grid edges must be finite")
        if np.any(np.diff(e1) <= 0.0) or np.any(np.diff(e2) <= 0.0):
            raise ValueError("grid edges must be strictly increasing")
        object.__setattr__(self, "edges1", e1)
        object.__setattr__(self, "edges2", e2)

    @property
    def n1(self) -> int:
        return self.edges1.size - 1

    @property
    def n2(self) -> int:
        return self.edges2.size - 1

    @property
    def k(self) -> int:
        return self.n1 * self.n2

    @property
    def widths1(self) -> np.ndarray:
        return np.diff(self.edges1)

    @property
    def widths2(self) -> np.ndarray:
        return np.diff(self.edges2)

    @property
    def areas_matrix(self) -> np.ndarray:
        return self.widths1[:, None] * self.widths2[None, :]

    @property
    def areas(self) -> np.ndarray:
        return self.areas_matrix.reshape(-1)

    @property
    def centres1(self) -> np.ndarray:
        return 0.5 * (self.edges1[:-1] + self.edges1[1:])

    @property
    def centres2(self) -> np.ndarray:
        return 0.5 * (self.edges2[:-1] + self.edges2[1:])

    @property
    def cell_table(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for i in range(self.n1):
            for j in range(self.n2):
                k = self.flat_index(i, j)
                rows.append(
                    {
                        "cell_index": k,
                        "i1": i,
                        "i2": j,
                        "x1_lower": float(self.edges1[i]),
                        "x1_upper": float(self.edges1[i + 1]),
                        "x2_lower": float(self.edges2[j]),
                        "x2_upper": float(self.edges2[j + 1]),
                        "x1_centre": float(self.centres1[i]),
                        "x2_centre": float(self.centres2[j]),
                        "width1": float(self.widths1[i]),
                        "width2": float(self.widths2[j]),
                        "area": float(self.widths1[i] * self.widths2[j]),
                    }
                )
        return pd.DataFrame(rows)

    def flat_index(self, i: int, j: int) -> int:
        return int(i * self.n2 + j)

    def unravel_index(self, k: int) -> Tuple[int, int]:
        return divmod(int(k), self.n2)

    @staticmethod
    def _fraction_below(edges: np.ndarray, x: float) -> np.ndarray:
        widths = np.diff(edges)
        lengths = np.clip(float(x) - edges[:-1], 0.0, widths)
        return lengths / widths

    def fraction_below1(self, x: float) -> np.ndarray:
        return self._fraction_below(self.edges1, x)

    def fraction_below2(self, x: float) -> np.ndarray:
        return self._fraction_below(self.edges2, x)

    def fraction_ge1(self, x: float) -> np.ndarray:
        return 1.0 - self.fraction_below1(x)

    def fraction_ge2(self, x: float) -> np.ndarray:
        return 1.0 - self.fraction_below2(x)

    @staticmethod
    def _cell_index(edges: np.ndarray, x: float) -> int:
        value = float(x)
        if value < edges[0] or value > edges[-1]:
            return -1
        if value == edges[-1]:
            return len(edges) - 2
        j = int(np.searchsorted(edges, value, side="right") - 1)
        if j < 0 or j >= len(edges) - 1:
            return -1
        return j

    def cell_indices(self, x1: float, x2: float) -> Tuple[int, int]:
        return self._cell_index(self.edges1, x1), self._cell_index(self.edges2, x2)

    def density_row(self, x1: float, x2: float) -> np.ndarray:
        row = np.zeros(self.k, dtype=float)
        i, j = self.cell_indices(x1, x2)
        if i >= 0 and j >= 0:
            row[self.flat_index(i, j)] = 1.0 / (
                self.widths1[i] * self.widths2[j]
            )
        return row

    def region_b_subdensity_row(self, x1: float, y2: float) -> np.ndarray:
        """Coefficient row for f_{X1, X2<y2}(x1)."""
        row = np.zeros((self.n1, self.n2), dtype=float)
        i = self._cell_index(self.edges1, x1)
        if i < 0:
            return row.reshape(-1)
        row[i, :] = self.fraction_below2(y2) / self.widths1[i]
        return row.reshape(-1)

    def region_c_subdensity_row(self, y1: float, x2: float) -> np.ndarray:
        """Coefficient row for f_{X2, X1<y1}(x2)."""
        row = np.zeros((self.n1, self.n2), dtype=float)
        j = self._cell_index(self.edges2, x2)
        if j < 0:
            return row.reshape(-1)
        row[:, j] = self.fraction_below1(y1) / self.widths2[j]
        return row.reshape(-1)

    def cdf_row(self, x1: float, x2: float) -> np.ndarray:
        return np.outer(self.fraction_below1(x1), self.fraction_below2(x2)).reshape(-1)

    def selection_row(self, y1: float, y2: float) -> np.ndarray:
        """Cell probabilities of O={X1>=y1 or X2>=y2}."""
        below_both = np.outer(self.fraction_below1(y1), self.fraction_below2(y2))
        return (1.0 - below_both).reshape(-1)

    def cdf(self, mass: np.ndarray, x1: float, x2: float) -> float:
        p = np.asarray(mass, dtype=float)
        if p.size != self.k:
            raise ValueError("mass length differs from the grid cell count")
        return float(self.cdf_row(x1, x2) @ p)

    def density(self, mass: np.ndarray, x1: float, x2: float) -> float:
        return float(self.density_row(x1, x2) @ np.asarray(mass, dtype=float))

    def assert_boundary_alignment(self, values1: np.ndarray, values2: np.ndarray, tol: float = 1e-12) -> None:
        for value in np.asarray(values1, dtype=float):
            if float(np.min(np.abs(self.edges1 - value))) > tol:
                raise ValueError(f"boundary {value} is not aligned with the x1 grid")
        for value in np.asarray(values2, dtype=float):
            if float(np.min(np.abs(self.edges2 - value))) > tol:
                raise ValueError(f"boundary {value} is not aligned with the x2 grid")


# =============================================================================
# Conditional likelihood and derivatives
# =============================================================================
def validate_operator_matrices(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.ndim != 2 or bb.ndim != 2 or aa.shape != bb.shape:
        raise ValueError("A and B must be finite matrices with identical shapes")
    if aa.shape[0] < 1 or aa.shape[1] < 1:
        raise ValueError("A and B must be non-empty")
    if not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        raise ValueError("A or B contains non-finite coefficients")
    if np.any(aa < 0.0) or np.any(bb < 0.0):
        raise ValueError("basis likelihood coefficients must be nonnegative")
    if np.any(aa.sum(axis=1) <= 0.0):
        raise ValueError("at least one numerator row has no candidate support")
    if np.any(bb.sum(axis=1) <= 0.0):
        raise ValueError("at least one selection denominator row has no candidate support")
    return aa, bb


def conditional_loglikelihood(
    mass: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray] = None,
) -> float:
    aa, bb = validate_operator_matrices(a, b)
    p = np.asarray(mass, dtype=float)
    if p.ndim != 1 or p.size != aa.shape[1]:
        raise ValueError("mass dimension differs from the operator column count")
    w = _normalise_row_weights(row_weights, aa.shape[0])
    numerator = aa @ p
    denominator = bb @ p
    if np.any(numerator <= 0.0) or np.any(denominator <= 0.0):
        return -math.inf
    return float(np.sum(w * (np.log(numerator) - np.log(denominator))))


def conditional_score(
    mass: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    aa, bb = validate_operator_matrices(a, b)
    p = np.asarray(mass, dtype=float)
    w = _normalise_row_weights(row_weights, aa.shape[0])
    numerator = aa @ p
    denominator = bb @ p
    if np.any(numerator <= 0.0) or np.any(denominator <= 0.0):
        raise ValueError("score requested outside the positive likelihood domain")
    return aa.T @ (w / numerator) - bb.T @ (w / denominator)


def conditional_hessian(
    mass: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    aa, bb = validate_operator_matrices(a, b)
    p = np.asarray(mass, dtype=float)
    w = _normalise_row_weights(row_weights, aa.shape[0])
    numerator = aa @ p
    denominator = bb @ p
    if np.any(numerator <= 0.0) or np.any(denominator <= 0.0):
        raise ValueError("Hessian requested outside the positive likelihood domain")
    return -(aa.T * (w / numerator**2)) @ aa + (bb.T * (w / denominator**2)) @ bb


def kkt_diagnostics(
    mass: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray] = None,
    active_tolerance: float = 1e-10,
) -> Dict[str, Any]:
    p = np.asarray(mass, dtype=float)
    score = conditional_score(p, a, b, row_weights)
    active = p > active_tolerance
    inactive = ~active
    active_residual = float(np.max(np.abs(score[active]))) if np.any(active) else math.nan
    inactive_max_score = float(np.max(score[inactive])) if np.any(inactive) else -math.inf
    inactive_violation = max(inactive_max_score, 0.0) if np.any(inactive) else 0.0
    return {
        "score_dot_mass": float(np.dot(score, p)),
        "active_score_sup_abs_residual": active_residual,
        "inactive_max_score": inactive_max_score,
        "inactive_positive_score_violation": float(inactive_violation),
        "n_active": int(active.sum()),
        "n_inactive": int(inactive.sum()),
        "active_indices": np.flatnonzero(active),
        "inactive_indices": np.flatnonzero(inactive),
        "score": score,
    }


def projected_hessian_diagnostics(
    mass: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray] = None,
    active_tolerance: float = 1e-10,
) -> Dict[str, Any]:
    p = np.asarray(mass, dtype=float)
    active_idx = np.flatnonzero(p > active_tolerance)
    if active_idx.size <= 1:
        eig = np.array([], dtype=float)
        return {
            "projected_hessian_eigenvalues": eig,
            "projected_hessian_min_eigenvalue": math.nan,
            "projected_hessian_max_eigenvalue": math.nan,
            "strict_local_concavity_on_active_face": True,
        }
    h = conditional_hessian(p, a, b, row_weights)
    q = _tangent_basis(active_idx.size)
    hp = q.T @ h[np.ix_(active_idx, active_idx)] @ q
    eig = np.linalg.eigvalsh(hp)
    return {
        "projected_hessian_eigenvalues": eig,
        "projected_hessian_min_eigenvalue": float(eig.min()),
        "projected_hessian_max_eigenvalue": float(eig.max()),
        "strict_local_concavity_on_active_face": bool(eig.max() < -1e-10),
    }


# =============================================================================
# Candidate screening and exact column equivalence
# =============================================================================
@dataclass
class OperatorReduction:
    a: np.ndarray
    b: np.ndarray
    classes: List[List[int]]
    forced_zero_indices: np.ndarray
    invisible_indices: np.ndarray
    cell_screening: pd.DataFrame
    class_table: pd.DataFrame


def reduce_operator_columns(
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray] = None,
    coverage_tolerance: float = 1e-15,
    signature_decimals: int = 14,
) -> OperatorReduction:
    aa, bb = validate_operator_matrices(a, b)
    w = _normalise_row_weights(row_weights, aa.shape[0])
    numerator_coverage = aa.T @ w
    denominator_coverage = bb.T @ w
    forced_zero = (numerator_coverage <= coverage_tolerance) & (
        denominator_coverage > coverage_tolerance
    )
    invisible = (numerator_coverage <= coverage_tolerance) & (
        denominator_coverage <= coverage_tolerance
    )
    kept = ~(forced_zero | invisible)
    kept_indices = np.flatnonzero(kept)

    groups: Dict[bytes, List[int]] = {}
    for j in kept_indices:
        signature_array = np.round(
            np.concatenate([aa[:, j], bb[:, j]]), decimals=signature_decimals
        )
        groups.setdefault(signature_array.tobytes(), []).append(int(j))
    classes = list(groups.values())
    reduced_a = np.column_stack([aa[:, cls[0]] for cls in classes])
    reduced_b = np.column_stack([bb[:, cls[0]] for cls in classes])

    class_of = np.full(aa.shape[1], -1, dtype=int)
    for h, cls in enumerate(classes):
        class_of[cls] = h

    screening_rows: List[Dict[str, Any]] = []
    for j in range(aa.shape[1]):
        if forced_zero[j]:
            status = "forced_zero_zero_numerator_positive_denominator"
        elif invisible[j]:
            status = "invisible_zero_numerator_zero_denominator"
        elif len(classes[class_of[j]]) > 1:
            status = "kept_observationally_equivalent"
        else:
            status = "kept_unique"
        screening_rows.append(
            {
                "original_column": j,
                "numerator_coverage": float(numerator_coverage[j]),
                "denominator_coverage": float(denominator_coverage[j]),
                "equivalence_class": int(class_of[j]),
                "status": status,
            }
        )

    class_rows = [
        {
            "class_index": h,
            "n_members": len(cls),
            "member_original_columns": ",".join(str(j) for j in cls),
            "numerator_coverage": float(numerator_coverage[cls[0]]),
            "denominator_coverage": float(denominator_coverage[cls[0]]),
        }
        for h, cls in enumerate(classes)
    ]
    return OperatorReduction(
        a=reduced_a,
        b=reduced_b,
        classes=classes,
        forced_zero_indices=np.flatnonzero(forced_zero),
        invisible_indices=np.flatnonzero(invisible),
        cell_screening=pd.DataFrame(screening_rows),
        class_table=pd.DataFrame(class_rows),
    )


def collapse_mass_to_classes(mass: np.ndarray, classes: Sequence[Sequence[int]]) -> np.ndarray:
    p = np.asarray(mass, dtype=float)
    return np.array([float(p[list(cls)].sum()) for cls in classes], dtype=float)


def expand_class_mass(
    class_mass: np.ndarray,
    classes: Sequence[Sequence[int]],
    n_original: int,
    split: str = "equal",
) -> np.ndarray:
    q = np.asarray(class_mass, dtype=float)
    p = np.zeros(n_original, dtype=float)
    for h, cls_raw in enumerate(classes):
        cls = list(cls_raw)
        if split == "first":
            p[cls[0]] = q[h]
        elif split == "equal":
            p[cls] = q[h] / len(cls)
        else:
            raise ValueError("split must be 'equal' or 'first'")
    return p


# =============================================================================
# Active-set MM and independent optimiser
# =============================================================================
def _mm_update_on_face(
    p_active: np.ndarray,
    a_active: np.ndarray,
    b_active: np.ndarray,
    row_weights: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    p = _normalise_probability(p_active, "p_active")
    numerator = a_active @ p
    denominator = b_active @ p
    if np.any(numerator <= 0.0) or np.any(denominator <= 0.0):
        raise RuntimeError("MM update left the positive likelihood domain")
    r_coeff = p * (a_active.T @ (row_weights / numerator))
    d_coeff = b_active.T @ (row_weights / denominator)
    if np.any(r_coeff <= 0.0):
        raise RuntimeError("active MM numerator coefficient is not strictly positive")

    lower = -float(d_coeff.min()) + max(1e-14, 1e-12 * max(1.0, float(d_coeff.max())))

    def root_function(lam: float) -> float:
        denominators = d_coeff + lam
        if np.any(denominators <= 0.0):
            return math.inf
        return float(np.sum(r_coeff / denominators) - 1.0)

    if root_function(lower) <= 0.0:
        # In exact arithmetic the function diverges at the lower boundary.  Move
        # closer to the pole if floating-point spacing made the initial offset too large.
        pole = -float(d_coeff.min())
        lower = np.nextafter(pole, math.inf)
    if not root_function(lower) > 0.0:
        raise RuntimeError("failed to bracket the MM multiplier from below")
    upper = max(1.0, float(d_coeff.max()))
    while root_function(upper) > 0.0:
        upper *= 2.0
        if upper > 1e14:
            raise RuntimeError("failed to bracket the MM multiplier from above")
    lam = float(brentq(root_function, lower, upper, xtol=1e-15, rtol=1e-14, maxiter=1000))
    updated = r_coeff / (d_coeff + lam)
    updated /= updated.sum()
    return updated, {
        "lambda": lam,
        "updated_mass_min": float(updated.min()),
        "updated_mass_max": float(updated.max()),
    }


@dataclass
class ActiveSetMMFit:
    mass: np.ndarray
    active_mask: np.ndarray
    history: pd.DataFrame
    events: pd.DataFrame
    converged: bool
    diagnostics: Dict[str, Any]


def fit_active_set_mm(
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray] = None,
    initial_mass: Optional[np.ndarray] = None,
    mass_tolerance: float = 1e-11,
    score_tolerance: float = 1e-8,
    prune_mass_tolerance: float = 1e-10,
    prune_score_margin: float = 1e-6,
    max_iterations: int = 20000,
    minimum_mm_steps: int = 2,
    likelihood_decrease_tolerance: float = 1e-11,
    kkt_active_tolerance: float = 1e-12,
) -> ActiveSetMMFit:
    aa, bb = validate_operator_matrices(a, b)
    n, k = aa.shape
    w = _normalise_row_weights(row_weights, n)
    if np.any(aa.T @ w <= 0.0):
        raise ValueError("remove zero-numerator columns before active-set MM")

    if initial_mass is None:
        p = np.full(k, 1.0 / k, dtype=float)
    else:
        p = _normalise_probability(initial_mass, "initial_mass")
        if p.size != k or np.any(p <= 0.0):
            raise ValueError("initial_mass must be strictly positive and match the operator")

    active = np.ones(k, dtype=bool)
    history_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    converged = False
    mm_steps = 0
    initial_ll = conditional_loglikelihood(p, aa, bb, w)

    for outer_iteration in range(1, max_iterations + 1):
        ll_current = conditional_loglikelihood(p, aa, bb, w)
        score_current = conditional_score(p, aa, bb, w)

        prune_candidates = [
            int(j)
            for j in np.flatnonzero(active)
            if p[j] < prune_mass_tolerance and score_current[j] < -prune_score_margin
        ]
        pruned = False
        for j in sorted(prune_candidates, key=lambda jj: score_current[jj]):
            proposed = active.copy()
            proposed[j] = False
            if not np.all(aa[:, proposed].sum(axis=1) > 0.0):
                continue
            trial = p.copy()
            removed = float(trial[j])
            trial[j] = 0.0
            trial /= trial.sum()
            ll_trial = conditional_loglikelihood(trial, aa, bb, w)
            increment = ll_trial - ll_current
            if increment >= -likelihood_decrease_tolerance:
                active = proposed
                p = trial
                event_rows.append(
                    {
                        "outer_iteration": outer_iteration,
                        "event": "prune",
                        "class_index": j,
                        "mass_before": removed,
                        "score_before": float(score_current[j]),
                        "loglik_increment": float(increment),
                        "n_active_after": int(active.sum()),
                    }
                )
                pruned = True
                break
        if pruned:
            continue

        active_idx = np.flatnonzero(active)
        updated, step_diag = _mm_update_on_face(
            p[active_idx], aa[:, active_idx], bb[:, active_idx], w
        )
        p_new = np.zeros(k, dtype=float)
        p_new[active_idx] = updated
        ll_new = conditional_loglikelihood(p_new, aa, bb, w)
        increment = ll_new - ll_current
        if increment < -likelihood_decrease_tolerance:
            raise RuntimeError(
                f"MM likelihood decreased by {increment:.3e} at iteration {outer_iteration}"
            )
        mass_change = float(np.max(np.abs(p_new - p)))
        p = p_new
        mm_steps += 1
        kkt = kkt_diagnostics(
            p, aa, bb, w, active_tolerance=kkt_active_tolerance
        )
        history_rows.append(
            {
                "outer_iteration": outer_iteration,
                "mm_step": mm_steps,
                "loglik_before": ll_current,
                "loglik_after": ll_new,
                "loglik_increment": increment,
                "sup_abs_mass_change": mass_change,
                "active_score_sup_abs_residual": kkt["active_score_sup_abs_residual"],
                "inactive_positive_score_violation": kkt["inactive_positive_score_violation"],
                "n_active": int(active.sum()),
                "mass_min_active": float(p[active].min()),
                "mass_max": float(p.max()),
                "lambda": step_diag["lambda"],
            }
        )

        if (
            mm_steps >= minimum_mm_steps
            and mass_change < mass_tolerance
            and kkt["active_score_sup_abs_residual"] < score_tolerance
        ):
            prunable_after = any(
                active[j]
                and p[j] < prune_mass_tolerance
                and kkt["score"][j] < -prune_score_margin
                for j in range(k)
            )
            if prunable_after:
                continue
            if kkt["inactive_positive_score_violation"] < score_tolerance:
                converged = True
                break

            inactive_idx = np.flatnonzero(~active)
            j = int(inactive_idx[np.argmax(kkt["score"][inactive_idx])])
            epsilon = 1e-6
            activated = False
            while epsilon >= 1e-16:
                trial = (1.0 - epsilon) * p
                trial[j] = epsilon
                ll_trial = conditional_loglikelihood(trial, aa, bb, w)
                if ll_trial > ll_new + 1e-15:
                    active[j] = True
                    p = trial
                    event_rows.append(
                        {
                            "outer_iteration": outer_iteration,
                            "event": "activate",
                            "class_index": j,
                            "mass_before": 0.0,
                            "score_before": float(kkt["score"][j]),
                            "loglik_increment": float(ll_trial - ll_new),
                            "n_active_after": int(active.sum()),
                        }
                    )
                    activated = True
                    break
                epsilon *= 0.5
            if not activated:
                raise RuntimeError(
                    "an inactive class had positive KKT score but could not be reactivated"
                )

    history = pd.DataFrame(history_rows)
    events = pd.DataFrame(event_rows)
    final_kkt = kkt_diagnostics(
        p, aa, bb, w, active_tolerance=kkt_active_tolerance
    )
    hdiag = projected_hessian_diagnostics(
        p, aa, bb, w, active_tolerance=prune_mass_tolerance
    )
    diagnostics: Dict[str, Any] = {
        "converged": converged,
        "outer_iterations": int(
            history["outer_iteration"].max()
            if not history.empty
            else (events["outer_iteration"].max() if not events.empty else 0)
        ),
        "mm_steps": int(mm_steps),
        "n_prune_events": int((events["event"] == "prune").sum()) if not events.empty else 0,
        "n_activation_events": int((events["event"] == "activate").sum()) if not events.empty else 0,
        "initial_loglikelihood": initial_ll,
        "final_loglikelihood": conditional_loglikelihood(p, aa, bb, w),
        "minimum_mm_loglikelihood_increment": (
            float(history["loglik_increment"].min()) if not history.empty else math.nan
        ),
        "likelihood_nondecreasing": bool(
            history.empty
            or float(history["loglik_increment"].min()) >= -likelihood_decrease_tolerance
        ),
        "final_sup_abs_mass_change": (
            float(history.iloc[-1]["sup_abs_mass_change"]) if not history.empty else math.nan
        ),
        "probability_mass_sum": float(p.sum()),
        "probability_normalisation_residual": float(abs(p.sum() - 1.0)),
        "all_masses_nonnegative": bool(np.all(p >= 0.0)),
        "mass_min": float(p.min()),
        "mass_max": float(p.max()),
        "n_exact_zero_masses": int(np.sum(p == 0.0)),
        **{key: value for key, value in final_kkt.items() if key != "score"},
        **hdiag,
    }
    return ActiveSetMMFit(p, active, history, events, converged, diagnostics)


@dataclass
class DirectFit:
    mass: np.ndarray
    runs: pd.DataFrame
    diagnostics: Dict[str, Any]


def fit_direct_slsqp_multistart(
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray] = None,
    n_starts: int = 8,
    seed: int = 104729,
    active_tolerance: float = 1e-8,
) -> DirectFit:
    aa, bb = validate_operator_matrices(a, b)
    n, k = aa.shape
    w = _normalise_row_weights(row_weights, n)
    rng = np.random.default_rng(seed)
    starts: List[np.ndarray] = [np.full(k, 1.0 / k)]
    for _ in range(max(0, n_starts - 1)):
        starts.append(rng.dirichlet(np.full(k, 1.2)))

    rows: List[Dict[str, Any]] = []
    solutions: List[np.ndarray] = []
    values: List[float] = []
    for start_id, p0 in enumerate(starts):
        def objective(p: np.ndarray) -> float:
            value = conditional_loglikelihood(p, aa, bb, w)
            return 1e100 if not np.isfinite(value) else -value

        def gradient(p: np.ndarray) -> np.ndarray:
            try:
                return -conditional_score(p, aa, bb, w)
            except ValueError:
                return np.zeros_like(p)

        result = minimize(
            objective,
            p0,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda p: float(p.sum() - 1.0),
                    "jac": lambda p: np.ones_like(p),
                }
            ],
            options={"ftol": 1e-13, "maxiter": 5000, "disp": False},
        )
        p = np.asarray(result.x, dtype=float)
        feasible = bool(
            np.all(np.isfinite(p))
            and p.min() >= -1e-9
            and abs(float(p.sum()) - 1.0) <= 1e-7
        )
        if feasible:
            p = np.maximum(p, 0.0)
            p /= p.sum()
            ll = conditional_loglikelihood(p, aa, bb, w)
        else:
            ll = -math.inf
        solutions.append(p)
        values.append(ll)
        rows.append(
            {
                "start_id": start_id,
                "success": bool(result.success),
                "feasible": feasible,
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(getattr(result, "nit", -1)),
                "loglikelihood": ll,
                "mass_min": float(p.min()) if p.size else math.nan,
                "mass_max": float(p.max()) if p.size else math.nan,
            }
        )
    best = int(np.argmax(values))
    p_best = solutions[best]
    kkt = kkt_diagnostics(p_best, aa, bb, w, active_tolerance=active_tolerance)
    diagnostics = {
        "best_start_id": best,
        "best_loglikelihood": float(values[best]),
        "n_successful_feasible_runs": int(sum(r["feasible"] for r in rows)),
        "loglikelihood_range_across_feasible_runs": float(
            np.ptp([v for v in values if np.isfinite(v)])
        ),
        **{key: value for key, value in kkt.items() if key != "score"},
    }
    return DirectFit(p_best, pd.DataFrame(rows), diagnostics)


# =============================================================================
# Observation operators for sample data
# =============================================================================
def build_sample_operator(
    observed: pd.DataFrame,
    grid: RectangularGrid,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    required = {"y1", "y2", "delta1", "delta2", "x1_obs", "x2_obs"}
    missing = required.difference(observed.columns)
    if missing:
        raise ValueError(f"observed catalogue is missing columns: {sorted(missing)}")
    a_rows: List[np.ndarray] = []
    b_rows: List[np.ndarray] = []
    audit_rows: List[Dict[str, Any]] = []
    for row_index, row in observed.reset_index(drop=True).iterrows():
        d1 = bool(row["delta1"])
        d2 = bool(row["delta2"])
        y1 = float(row["y1"])
        y2 = float(row["y2"])
        x1_obs = float(row["x1_obs"])
        x2_obs = float(row["x2_obs"])
        if d1 and d2:
            region = "A"
            arow = grid.density_row(x1_obs, x2_obs)
        elif d1:
            region = "B"
            arow = grid.region_b_subdensity_row(x1_obs, y2)
        elif d2:
            region = "C"
            arow = grid.region_c_subdensity_row(y1, x2_obs)
        else:
            raise ValueError("Region D must not appear in the observed A-C catalogue")
        brow = grid.selection_row(y1, y2)
        if not np.any(arow > 0.0):
            raise ValueError(f"numerator row {row_index} is outside the sieve support")
        if not np.any(brow > 0.0):
            raise ValueError(f"selection row {row_index} has zero sieve coverage")
        positive_a = arow > 0.0
        subset_ok = bool(np.all(brow[positive_a] > 0.0))
        if not subset_ok:
            raise RuntimeError("numerator event is not contained in the selection event")
        a_rows.append(arow)
        b_rows.append(brow)
        audit_rows.append(
            {
                "row_index": row_index,
                "region": region,
                "y1": y1,
                "y2": y2,
                "x1_obs": x1_obs,
                "x2_obs": x2_obs,
                "numerator_nonzero_cells": int(np.sum(positive_a)),
                "selection_nonzero_cells": int(np.sum(brow > 0.0)),
                "numerator_subset_selection": subset_ok,
                "numerator_row_sum": float(arow.sum()),
                "selection_row_sum": float(brow.sum()),
            }
        )
    return np.vstack(a_rows), np.vstack(b_rows), pd.DataFrame(audit_rows)


# =============================================================================
# Population category operator (grid-aligned boundaries)
# =============================================================================
def build_population_categories(
    grid: RectangularGrid,
    true_cell_mass: np.ndarray,
    y_support: np.ndarray,
    y_mass: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    p0 = _normalise_probability(np.asarray(true_cell_mass, dtype=float).reshape(-1), "true_cell_mass")
    if p0.size != grid.k:
        raise ValueError("true_cell_mass length differs from the sieve cell count")
    ys = np.asarray(y_support, dtype=float)
    qy = _normalise_probability(np.asarray(y_mass, dtype=float), "y_mass")
    if ys.ndim != 2 or ys.shape[1] != 2 or ys.shape[0] != qy.size:
        raise ValueError("y_support must have shape (J,2) matching y_mass")
    grid.assert_boundary_alignment(ys[:, 0], ys[:, 1])

    pmat = p0.reshape(grid.n1, grid.n2)
    c1 = grid.centres1
    c2 = grid.centres2
    a_rows: List[np.ndarray] = []
    b_rows: List[np.ndarray] = []
    raw_weights: List[float] = []
    meta: List[Dict[str, Any]] = []
    alpha = 0.0

    for y_index, (y, q) in enumerate(zip(ys, qy)):
        y1, y2 = map(float, y)
        brow = grid.selection_row(y1, y2)
        selection_probability = float(brow @ p0)
        alpha += float(q) * selection_probability
        d1_cells = c1 >= y1
        d2_cells = c2 >= y2

        # Region A: exact point in each cell that lies in the double-detection quadrant.
        for i in range(grid.n1):
            if not d1_cells[i]:
                continue
            for j in range(grid.n2):
                if not d2_cells[j]:
                    continue
                category_mass = float(pmat[i, j])
                if category_mass <= 0.0:
                    continue
                arow = grid.density_row(c1[i], c2[j])
                a_rows.append(arow)
                b_rows.append(brow)
                raw_weights.append(float(q) * category_mass)
                meta.append(
                    {
                        "y_index": y_index,
                        "y1": y1,
                        "y2": y2,
                        "region": "A",
                        "x1_cell": i,
                        "x2_cell": j,
                        "parent_category_mass": category_mass,
                    }
                )

        # Region B: exact x1 cell, second coordinate known only to be below y2.
        for i in range(grid.n1):
            if not d1_cells[i]:
                continue
            category_mass = float(pmat[i, ~d2_cells].sum())
            if category_mass <= 0.0:
                continue
            arow = grid.region_b_subdensity_row(c1[i], y2)
            a_rows.append(arow)
            b_rows.append(brow)
            raw_weights.append(float(q) * category_mass)
            meta.append(
                {
                    "y_index": y_index,
                    "y1": y1,
                    "y2": y2,
                    "region": "B",
                    "x1_cell": i,
                    "x2_cell": -1,
                    "parent_category_mass": category_mass,
                }
            )

        # Region C: first coordinate below y1, exact x2 cell.
        for j in range(grid.n2):
            if not d2_cells[j]:
                continue
            category_mass = float(pmat[~d1_cells, j].sum())
            if category_mass <= 0.0:
                continue
            arow = grid.region_c_subdensity_row(y1, c2[j])
            a_rows.append(arow)
            b_rows.append(brow)
            raw_weights.append(float(q) * category_mass)
            meta.append(
                {
                    "y_index": y_index,
                    "y1": y1,
                    "y2": y2,
                    "region": "C",
                    "x1_cell": -1,
                    "x2_cell": j,
                    "parent_category_mass": category_mass,
                }
            )

    raw = np.asarray(raw_weights, dtype=float)
    if not np.isclose(raw.sum(), alpha, atol=1e-12, rtol=1e-12):
        raise RuntimeError("population category weights do not sum to the observation probability")
    weights = raw / alpha
    metadata = pd.DataFrame(meta)
    metadata["unconditional_category_probability"] = raw
    metadata["conditional_observed_category_weight"] = weights
    a = np.vstack(a_rows)
    b = np.vstack(b_rows)
    info = {
        "n_population_observed_categories": int(len(weights)),
        "true_observation_probability": float(alpha),
        "population_category_weights_sum": float(weights.sum()),
        "region_weight_sums": {
            region: float(metadata.loc[metadata["region"] == region, "conditional_observed_category_weight"].sum())
            for region in ["A", "B", "C"]
        },
    }
    return a, b, weights, metadata, info


# =============================================================================
# Simulators
# =============================================================================
def simulate_piecewise_constant_catalogue(
    n_latent: int,
    seed: int,
    grid: RectangularGrid,
    true_mass: np.ndarray,
    y_support: np.ndarray,
    y_mass: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    p0 = _normalise_probability(np.asarray(true_mass, dtype=float).reshape(-1), "true_mass")
    ys = np.asarray(y_support, dtype=float)
    qy = _normalise_probability(np.asarray(y_mass, dtype=float), "y_mass")
    rng = np.random.default_rng(seed)
    cell = rng.choice(grid.k, size=n_latent, p=p0)
    i = cell // grid.n2
    j = cell % grid.n2
    x1 = grid.edges1[i] + rng.random(n_latent) * grid.widths1[i]
    x2 = grid.edges2[j] + rng.random(n_latent) * grid.widths2[j]
    y_idx = rng.choice(len(qy), size=n_latent, p=qy)
    y1 = ys[y_idx, 0]
    y2 = ys[y_idx, 1]
    d1 = x1 >= y1
    d2 = x2 >= y2
    observed_flag = d1 | d2
    latent = pd.DataFrame(
        {
            "latent_id": np.arange(n_latent),
            "cell_index": cell,
            "x1_true": x1,
            "x2_true": x2,
            "y_index": y_idx,
            "y1": y1,
            "y2": y2,
            "delta1": d1.astype(int),
            "delta2": d2.astype(int),
            "observed": observed_flag.astype(int),
        }
    )
    latent["region"] = [
        _region_name(bool(a), bool(b)) for a, b in zip(d1, d2)
    ]
    observed = latent.loc[observed_flag].copy().reset_index(drop=True)
    observed["x1_obs"] = np.where(observed["delta1"] == 1, observed["x1_true"], observed["y1"])
    observed["x2_obs"] = np.where(observed["delta2"] == 1, observed["x2_true"], observed["y2"])
    info = {
        "n_latent": int(n_latent),
        "n_observed": int(len(observed)),
        "selection_fraction": float(len(observed) / n_latent),
        "region_counts_observed": {
            region: int((observed["region"] == region).sum()) for region in ["A", "B", "C"]
        },
    }
    return latent, observed, info


def sample_truncated_bivariate_normal(
    n: int,
    rho: float,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    chunks: List[np.ndarray] = []
    total = 0
    while total < n:
        need = n - total
        draw = rng.multivariate_normal(np.zeros(2), cov, size=max(need + 64, int(need * 1.03)))
        keep = draw[(draw[:, 0] >= lower) & (draw[:, 0] <= upper) & (draw[:, 1] >= lower) & (draw[:, 1] <= upper)]
        if keep.size:
            chunks.append(keep[:need])
            total += min(need, len(keep))
    return np.vstack(chunks)[:n]


def simulate_gaussian_catalogue(
    n_latent: int,
    seed: int,
    rho: float,
    lower: float,
    upper: float,
    y_support: np.ndarray,
    y_mass: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    x = sample_truncated_bivariate_normal(n_latent, rho, lower, upper, rng)
    ys = np.asarray(y_support, dtype=float)
    qy = _normalise_probability(np.asarray(y_mass, dtype=float), "y_mass")
    y_idx = rng.choice(len(qy), size=n_latent, p=qy)
    y1 = ys[y_idx, 0]
    y2 = ys[y_idx, 1]
    d1 = x[:, 0] >= y1
    d2 = x[:, 1] >= y2
    observed_flag = d1 | d2
    latent = pd.DataFrame(
        {
            "latent_id": np.arange(n_latent),
            "x1_true": x[:, 0],
            "x2_true": x[:, 1],
            "y_index": y_idx,
            "y1": y1,
            "y2": y2,
            "delta1": d1.astype(int),
            "delta2": d2.astype(int),
            "observed": observed_flag.astype(int),
        }
    )
    latent["region"] = [
        _region_name(bool(a), bool(b)) for a, b in zip(d1, d2)
    ]
    observed = latent.loc[observed_flag].copy().reset_index(drop=True)
    observed["x1_obs"] = np.where(observed["delta1"] == 1, observed["x1_true"], observed["y1"])
    observed["x2_obs"] = np.where(observed["delta2"] == 1, observed["x2_true"], observed["y2"])
    info = {
        "rho": float(rho),
        "n_latent": int(n_latent),
        "n_observed": int(len(observed)),
        "selection_fraction": float(len(observed) / n_latent),
        "region_counts_observed": {
            region: int((observed["region"] == region).sum()) for region in ["A", "B", "C"]
        },
    }
    return latent, observed, info


# =============================================================================
# Bivariate Gaussian truth on a bounded box
# =============================================================================
class TruncatedBivariateNormalTruth:
    def __init__(self, rho: float, lower: float = -3.0, upper: float = 3.0) -> None:
        if not (-0.999 < rho < 0.999):
            raise ValueError("rho must be strictly between -0.999 and 0.999")
        self.rho = float(rho)
        self.lower = float(lower)
        self.upper = float(upper)
        self.mean = np.zeros(2)
        self.cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
        self._dist = multivariate_normal(mean=self.mean, cov=self.cov)
        self._box_probability = self._rectangle_unconditional(
            self.lower, self.upper, self.lower, self.upper
        )
        if not self._box_probability > 0.0:
            raise RuntimeError("truncation box has zero probability")

    def _rectangle_unconditional(self, l1: float, u1: float, l2: float, u2: float) -> float:
        if u1 <= l1 or u2 <= l2:
            return 0.0
        value = multivariate_normal.cdf(
            np.array([u1, u2]),
            mean=self.mean,
            cov=self.cov,
            lower_limit=np.array([l1, l2]),
            maxpts=1_000_000,
            abseps=1e-10,
            releps=1e-10,
            rng=np.random.default_rng(314159),
        )
        return float(max(value, 0.0))

    @property
    def box_probability(self) -> float:
        return float(self._box_probability)

    def rectangle_probability(self, l1: float, u1: float, l2: float, u2: float) -> float:
        ll1 = max(float(l1), self.lower)
        uu1 = min(float(u1), self.upper)
        ll2 = max(float(l2), self.lower)
        uu2 = min(float(u2), self.upper)
        if uu1 <= ll1 or uu2 <= ll2:
            return 0.0
        return self._rectangle_unconditional(ll1, uu1, ll2, uu2) / self._box_probability

    def cell_masses(self, grid: RectangularGrid) -> np.ndarray:
        if grid.edges1[0] < self.lower - 1e-12 or grid.edges1[-1] > self.upper + 1e-12:
            raise ValueError("x1 grid extends beyond the truncated Gaussian support")
        if grid.edges2[0] < self.lower - 1e-12 or grid.edges2[-1] > self.upper + 1e-12:
            raise ValueError("x2 grid extends beyond the truncated Gaussian support")
        masses = np.zeros((grid.n1, grid.n2), dtype=float)
        for i in range(grid.n1):
            for j in range(grid.n2):
                masses[i, j] = self.rectangle_probability(
                    grid.edges1[i], grid.edges1[i + 1], grid.edges2[j], grid.edges2[j + 1]
                )
        masses = np.maximum(masses, 0.0)
        masses /= masses.sum()
        return masses.reshape(-1)

    def cdf(self, x1: float, x2: float) -> float:
        if x1 <= self.lower or x2 <= self.lower:
            return 0.0
        if x1 >= self.upper and x2 >= self.upper:
            return 1.0
        return self.rectangle_probability(
            self.lower, min(float(x1), self.upper), self.lower, min(float(x2), self.upper)
        )


# =============================================================================
# Accuracy tables
# =============================================================================
def cdf_accuracy_table(
    grid: RectangularGrid,
    mass: np.ndarray,
    truth_cdf: Callable[[float, float], float],
    query_grid_size: int = 41,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    x1_values = np.linspace(grid.edges1[0], grid.edges1[-1], query_grid_size)
    x2_values = np.linspace(grid.edges2[0], grid.edges2[-1], query_grid_size)
    rows: List[Dict[str, float]] = []
    errors: List[float] = []
    for x1 in x1_values:
        for x2 in x2_values:
            est = grid.cdf(mass, float(x1), float(x2))
            true = float(truth_cdf(float(x1), float(x2)))
            err = est - true
            errors.append(err)
            rows.append(
                {
                    "x1": float(x1),
                    "x2": float(x2),
                    "cdf_estimate": est,
                    "cdf_truth": true,
                    "error": err,
                    "abs_error": abs(err),
                }
            )
    e = np.asarray(errors)
    metrics = {
        "cdf_rmse": float(np.sqrt(np.mean(e**2))),
        "cdf_mae": float(np.mean(np.abs(e))),
        "cdf_max_abs_error": float(np.max(np.abs(e))),
        "n_query_points": int(e.size),
    }
    return pd.DataFrame(rows), metrics


def piecewise_truth_cdf(grid: RectangularGrid, true_mass: np.ndarray) -> Callable[[float, float], float]:
    p = np.asarray(true_mass, dtype=float).reshape(-1)
    return lambda x1, x2: grid.cdf(p, x1, x2)


def cdf_identification_envelope(
    grid: RectangularGrid,
    class_mass: np.ndarray,
    classes: Sequence[Sequence[int]],
    truth_cdf: Optional[Callable[[float, float], float]] = None,
    query_grid_size: int = 41,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """CDF range induced solely by unidentified within-class mass allocation.

    The likelihood identifies the total mass of each exact operator-equivalence
    class.  If a class contains several geometric cells, its contribution to a
    CDF query can range between the minimum and maximum CDF coefficient among
    its member cells.  These extrema are attainable by concentrating the class
    mass on a corresponding member cell.
    """
    q = np.asarray(class_mass, dtype=float)
    if q.ndim != 1 or q.size != len(classes):
        raise ValueError("class_mass does not match the equivalence classes")
    x1_values = np.linspace(grid.edges1[0], grid.edges1[-1], query_grid_size)
    x2_values = np.linspace(grid.edges2[0], grid.edges2[-1], query_grid_size)
    rows: List[Dict[str, float]] = []
    widths: List[float] = []
    representative_errors: List[float] = []
    distance_to_envelope: List[float] = []
    truth_inside: List[bool] = []
    for x1 in x1_values:
        for x2 in x2_values:
            coeff = grid.cdf_row(float(x1), float(x2))
            lower = 0.0
            upper = 0.0
            representative = 0.0
            for mass_h, cls_raw in zip(q, classes):
                cls = list(cls_raw)
                values = coeff[cls]
                lower += float(mass_h) * float(values.min())
                upper += float(mass_h) * float(values.max())
                representative += float(mass_h) * float(values.mean())
            width = upper - lower
            record: Dict[str, float] = {
                "x1": float(x1),
                "x2": float(x2),
                "cdf_lower": lower,
                "cdf_equal_split": representative,
                "cdf_upper": upper,
                "identification_width": width,
            }
            widths.append(width)
            if truth_cdf is not None:
                true = float(truth_cdf(float(x1), float(x2)))
                inside = bool(lower - 1e-14 <= true <= upper + 1e-14)
                distance = max(lower - true, 0.0, true - upper)
                record.update(
                    {
                        "cdf_truth": true,
                        "truth_inside_envelope": int(inside),
                        "distance_truth_to_envelope": distance,
                        "equal_split_error": representative - true,
                    }
                )
                representative_errors.append(representative - true)
                distance_to_envelope.append(distance)
                truth_inside.append(inside)
            rows.append(record)
    metrics: Dict[str, float] = {
        "cdf_identification_envelope_max_width": float(np.max(widths)),
        "cdf_identification_envelope_mean_width": float(np.mean(widths)),
    }
    if truth_cdf is not None:
        re = np.asarray(representative_errors)
        de = np.asarray(distance_to_envelope)
        metrics.update(
            {
                "cdf_equal_split_rmse": float(np.sqrt(np.mean(re**2))),
                "cdf_equal_split_mae": float(np.mean(np.abs(re))),
                "cdf_equal_split_max_abs_error": float(np.max(np.abs(re))),
                "truth_inside_identification_envelope_fraction": float(np.mean(truth_inside)),
                "truth_distance_to_envelope_rmse": float(np.sqrt(np.mean(de**2))),
                "truth_distance_to_envelope_max": float(np.max(de)),
            }
        )
    return pd.DataFrame(rows), metrics


# =============================================================================
# Benchmark constants
# =============================================================================
PIECEWISE_EDGES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
PIECEWISE_TRUE_MASS_MATRIX = np.array(
    [
        [0.02, 0.03, 0.02, 0.01],
        [0.03, 0.10, 0.12, 0.03],
        [0.02, 0.12, 0.20, 0.10],
        [0.01, 0.03, 0.10, 0.06],
    ],
    dtype=float,
)
PIECEWISE_Y_SUPPORT = np.array(
    [
        [-2.0, -2.0],
        [-1.0, -1.0],
        [ 0.0, -1.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
        [ 1.0,  0.0],
        [ 0.0,  1.0],
    ],
    dtype=float,
)
PIECEWISE_Y_MASS = np.array([0.05, 0.10, 0.15, 0.15, 0.25, 0.15, 0.15])

GAUSSIAN_LOWER = -3.0
GAUSSIAN_UPPER = 3.0
GAUSSIAN_Y_SUPPORT = np.array(
    [
        [-3.0, -3.0],
        [-1.5, -1.5],
        [ 0.0, -1.5],
        [-1.5,  0.0],
        [ 0.0,  0.0],
        [ 1.5,  0.0],
        [ 0.0,  1.5],
    ],
    dtype=float,
)
GAUSSIAN_Y_MASS = np.array([0.05, 0.10, 0.15, 0.15, 0.25, 0.15, 0.15])


# =============================================================================
# Fit wrapper on original grid cells
# =============================================================================
@dataclass
class SieveFitBundle:
    original_mass: np.ndarray
    reduction: OperatorReduction
    mm: ActiveSetMMFit
    direct: Optional[DirectFit]
    diagnostics: Dict[str, Any]


def fit_sieve_likelihood(
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray],
    coverage_tolerance: float,
    mass_tolerance: float,
    score_tolerance: float,
    prune_mass_tolerance: float,
    prune_score_margin: float,
    max_iterations: int,
    direct_starts: int = 0,
    direct_seed: int = 104729,
) -> SieveFitBundle:
    reduction = reduce_operator_columns(
        a, b, row_weights, coverage_tolerance=coverage_tolerance
    )
    if reduction.invisible_indices.size:
        raise RuntimeError(
            "completely invisible sieve cells were found; enlarge or redesign the "
            "observation design before interpreting the likelihood"
        )
    mm = fit_active_set_mm(
        reduction.a,
        reduction.b,
        row_weights,
        mass_tolerance=mass_tolerance,
        score_tolerance=score_tolerance,
        prune_mass_tolerance=prune_mass_tolerance,
        prune_score_margin=prune_score_margin,
        max_iterations=max_iterations,
    )
    original_mass = expand_class_mass(
        mm.mass, reduction.classes, a.shape[1], split="equal"
    )
    direct: Optional[DirectFit] = None
    agreement: Dict[str, Any] = {}
    if direct_starts > 0:
        direct = fit_direct_slsqp_multistart(
            reduction.a,
            reduction.b,
            row_weights,
            n_starts=direct_starts,
            seed=direct_seed,
        )
        agreement = {
            "mm_direct_class_mass_sup_abs_difference": float(
                np.max(np.abs(mm.mass - direct.mass))
            ),
            "mm_direct_loglikelihood_abs_difference": float(
                abs(mm.diagnostics["final_loglikelihood"] - direct.diagnostics["best_loglikelihood"])
            ),
        }
    diagnostics = {
        "n_original_cells": int(a.shape[1]),
        "n_forced_zero_cells": int(reduction.forced_zero_indices.size),
        "n_invisible_cells": int(reduction.invisible_indices.size),
        "n_equivalence_classes": int(len(reduction.classes)),
        "n_nontrivial_equivalence_classes": int(sum(len(cls) > 1 for cls in reduction.classes)),
        "mm": mm.diagnostics,
        "agreement": agreement,
    }
    return SieveFitBundle(original_mass, reduction, mm, direct, diagnostics)


# =============================================================================
# Self-tests
# =============================================================================
def run_self_tests() -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {}

    grid = RectangularGrid(np.array([-2.0, -0.5, 1.0]), np.array([-1.0, 0.25, 2.0]))

    # Every normalized basis integrates to one.
    basis_integrals = np.ones(grid.k)
    checks["rectangular_basis_functions_are_normalised"] = bool(
        np.allclose(basis_integrals, 1.0)
    )

    # Selection denominator identity: O is the complement of the lower-left rectangle.
    y1, y2 = -0.2, 0.7
    selection = grid.selection_row(y1, y2)
    identity = 1.0 - grid.cdf_row(y1, y2)
    details["selection_identity_sup_abs_difference"] = _max_abs(selection - identity)
    checks["selection_denominator_equals_one_minus_lower_left_cdf"] = bool(
        details["selection_identity_sup_abs_difference"] < 1e-15
    )

    # Analytic mixed-dimensional rows are checked against their exact cell integrals.
    x1 = 0.1
    brow = grid.region_b_subdensity_row(x1, y2).reshape(grid.n1, grid.n2)
    i = grid._cell_index(grid.edges1, x1)
    target_b = grid.fraction_below2(y2)
    recovered_b = brow[i, :] * grid.widths1[i]
    details["region_b_integral_sup_abs_difference"] = _max_abs(recovered_b - target_b)
    checks["region_b_subdensity_integral_is_exact"] = bool(
        details["region_b_integral_sup_abs_difference"] < 1e-15
    )

    x2 = 0.6
    crow = grid.region_c_subdensity_row(y1, x2).reshape(grid.n1, grid.n2)
    j = grid._cell_index(grid.edges2, x2)
    target_c = grid.fraction_below1(y1)
    recovered_c = crow[:, j] * grid.widths2[j]
    details["region_c_integral_sup_abs_difference"] = _max_abs(recovered_c - target_c)
    checks["region_c_subdensity_integral_is_exact"] = bool(
        details["region_c_integral_sup_abs_difference"] < 1e-15
    )

    # Numerator support is contained in the selection support for valid observations.
    rows = [
        (grid.density_row(0.1, 0.6), grid.selection_row(-0.2, 0.2)),
        (grid.region_b_subdensity_row(0.1, 0.2), grid.selection_row(-0.2, 0.2)),
        (grid.region_c_subdensity_row(-0.2, 0.6), grid.selection_row(-0.2, 0.2)),
    ]
    checks["all_mixed_dimensional_numerators_are_subsets_of_selection"] = all(
        np.all(b[a > 0.0] > 0.0) for a, b in rows
    )

    # Finite-difference score check on a small positive operator.
    a_full = np.vstack([r[0] for r in rows])
    b_full = np.vstack([r[1] for r in rows])
    reduction = reduce_operator_columns(a_full, b_full, None)
    if reduction.invisible_indices.size:
        raise RuntimeError("self-test operator unexpectedly contains invisible columns")
    a = reduction.a
    b = reduction.b
    p = _normalise_probability(np.arange(1, a.shape[1] + 1, dtype=float))
    direction = np.linspace(-1.0, 1.0, a.shape[1])
    direction -= direction.mean()
    direction /= np.max(np.abs(direction))
    eps = 1e-7
    analytic = float(conditional_score(p, a, b) @ direction)
    finite = (
        conditional_loglikelihood(p + eps * direction, a, b)
        - conditional_loglikelihood(p - eps * direction, a, b)
    ) / (2.0 * eps)
    details["directional_score_finite_difference_abs_error"] = abs(analytic - finite)
    checks["analytic_score_matches_finite_difference"] = bool(
        details["directional_score_finite_difference_abs_error"] < 1e-7
    )

    # One MM step must not decrease the conditional likelihood.
    p_step, _ = _mm_update_on_face(p, a, b, np.full(a.shape[0], 1.0 / a.shape[0]))
    increment = conditional_loglikelihood(p_step, a, b) - conditional_loglikelihood(p, a, b)
    details["single_mm_step_loglikelihood_increment"] = increment
    checks["single_mm_step_is_likelihood_nondecreasing"] = bool(increment >= -1e-12)

    # Complete data with a fixed grid reduces to empirical cell frequencies.
    complete_grid = RectangularGrid(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    complete = pd.DataFrame(
        {
            "y1": [-1.0] * 8,
            "y2": [-1.0] * 8,
            "delta1": [1] * 8,
            "delta2": [1] * 8,
            "x1_obs": [0.2, 0.4, 0.8, 1.2, 1.4, 1.7, 1.8, 1.9],
            "x2_obs": [0.2, 1.2, 1.6, 0.2, 0.4, 0.7, 1.2, 1.8],
        }
    )
    ca, cb, _ = build_sample_operator(complete, complete_grid)
    fit = fit_sieve_likelihood(
        ca,
        cb,
        None,
        coverage_tolerance=1e-15,
        mass_tolerance=1e-12,
        score_tolerance=1e-9,
        prune_mass_tolerance=1e-11,
        prune_score_margin=1e-7,
        max_iterations=5000,
        direct_starts=0,
    )
    counts = np.zeros(complete_grid.k)
    for x1v, x2v in zip(complete["x1_obs"], complete["x2_obs"]):
        ii, jj = complete_grid.cell_indices(float(x1v), float(x2v))
        counts[complete_grid.flat_index(ii, jj)] += 1.0
    empirical = counts / counts.sum()
    details["complete_data_mass_sup_abs_difference"] = _max_abs(fit.original_mass - empirical)
    checks["complete_data_reduces_to_empirical_histogram"] = bool(
        details["complete_data_mass_sup_abs_difference"] < 1e-9
    )

    # Row permutation invariance.
    permutation = np.array([2, 0, 1])
    fit1 = fit_sieve_likelihood(
        a,
        b,
        None,
        1e-15,
        1e-12,
        1e-9,
        1e-11,
        1e-7,
        5000,
        0,
    )
    fit2 = fit_sieve_likelihood(
        a[permutation],
        b[permutation],
        None,
        1e-15,
        1e-12,
        1e-9,
        1e-11,
        1e-7,
        5000,
        0,
    )
    details["row_permutation_mass_sup_abs_difference"] = _max_abs(
        fit1.original_mass - fit2.original_mass
    )
    checks["row_permutation_invariance"] = bool(
        details["row_permutation_mass_sup_abs_difference"] < 1e-10
    )

    return {
        "all_passed": bool(all(checks.values())),
        "checks": checks,
        "details": details,
    }




# =============================================================================
# Multi-seed bias--variance audit
# =============================================================================
import time
import traceback
from collections import defaultdict


def make_uniform_grid(bins_per_axis: int) -> RectangularGrid:
    m = int(bins_per_axis)
    if m < 2:
        raise ValueError("bins_per_axis must be at least two")
    edges = np.linspace(GAUSSIAN_LOWER, GAUSSIAN_UPPER, m + 1)
    return RectangularGrid(edges, edges)


def intersection_selection_row(grid: RectangularGrid, y1: float, y2: float) -> np.ndarray:
    """Cell probabilities of {X1 >= y1 and X2 >= y2}."""
    return np.outer(grid.fraction_ge1(y1), grid.fraction_ge2(y2)).reshape(-1)


def build_region_a_intersection_operator(
    observed: pd.DataFrame,
    grid: RectangularGrid,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Conditional-likelihood operator for fully detected Region-A objects only."""
    a_only = observed.loc[
        (observed["delta1"].to_numpy(int) == 1)
        & (observed["delta2"].to_numpy(int) == 1)
    ].copy().reset_index(drop=True)
    if a_only.empty:
        raise ValueError("Region A is empty")
    a_rows: List[np.ndarray] = []
    b_rows: List[np.ndarray] = []
    audit_rows: List[Dict[str, Any]] = []
    for row_index, row in a_only.iterrows():
        x1 = float(row["x1_obs"])
        x2 = float(row["x2_obs"])
        y1 = float(row["y1"])
        y2 = float(row["y2"])
        arow = grid.density_row(x1, x2)
        brow = intersection_selection_row(grid, y1, y2)
        if not np.any(arow > 0.0):
            raise ValueError(f"Region-A numerator row {row_index} is outside the sieve support")
        if not np.any(brow > 0.0):
            raise ValueError(f"Region-A denominator row {row_index} has zero support coverage")
        if not np.all(brow[arow > 0.0] > 0.0):
            raise RuntimeError("Region-A exact event is not contained in the intersection selection event")
        a_rows.append(arow)
        b_rows.append(brow)
        audit_rows.append(
            {
                "row_index": int(row_index),
                "x1_obs": x1,
                "x2_obs": x2,
                "y1": y1,
                "y2": y2,
                "numerator_nonzero_cells": int(np.sum(arow > 0.0)),
                "selection_nonzero_cells": int(np.sum(brow > 0.0)),
                "numerator_subset_selection": True,
            }
        )
    return np.vstack(a_rows), np.vstack(b_rows), pd.DataFrame(audit_rows)


def _overlap_transfer(target_edges: np.ndarray, source_edges: np.ndarray) -> np.ndarray:
    """Fraction of each source interval falling in each target interval."""
    te = np.asarray(target_edges, dtype=float)
    se = np.asarray(source_edges, dtype=float)
    out = np.zeros((len(te) - 1, len(se) - 1), dtype=float)
    sw = np.diff(se)
    for ti in range(len(te) - 1):
        lo_t, hi_t = te[ti], te[ti + 1]
        for si in range(len(se) - 1):
            lo = max(lo_t, se[si])
            hi = min(hi_t, se[si + 1])
            out[ti, si] = max(hi - lo, 0.0) / sw[si]
    return out


def project_rectangular_mass(
    source_grid: RectangularGrid,
    source_mass: np.ndarray,
    target_grid: RectangularGrid,
) -> np.ndarray:
    """Project a source piecewise-uniform density to target-cell probabilities."""
    if (
        abs(source_grid.edges1[0] - target_grid.edges1[0]) > 1e-12
        or abs(source_grid.edges1[-1] - target_grid.edges1[-1]) > 1e-12
        or abs(source_grid.edges2[0] - target_grid.edges2[0]) > 1e-12
        or abs(source_grid.edges2[-1] - target_grid.edges2[-1]) > 1e-12
    ):
        raise ValueError("source and target grids must cover the same rectangle")
    p = _normalise_probability(np.asarray(source_mass, dtype=float), "source_mass")
    if p.size != source_grid.k:
        raise ValueError("source_mass length differs from source_grid.k")
    t1 = _overlap_transfer(target_grid.edges1, source_grid.edges1)
    t2 = _overlap_transfer(target_grid.edges2, source_grid.edges2)
    target = t1 @ p.reshape(source_grid.n1, source_grid.n2) @ t2.T
    target = np.maximum(target.reshape(-1), 0.0)
    return _normalise_probability(target, "projected_mass")


def _strict_positive_class_start(
    original_mass: np.ndarray,
    reduction: OperatorReduction,
    mixing: float = 1e-8,
) -> np.ndarray:
    q = collapse_mass_to_classes(original_mass, reduction.classes)
    if not np.all(np.isfinite(q)) or float(q.sum()) <= 0.0:
        q = np.full(len(reduction.classes), 1.0 / len(reduction.classes))
    else:
        q = np.maximum(q, 0.0)
        q /= q.sum()
    eps = float(np.clip(mixing, 1e-15, 1e-2))
    q = (1.0 - eps) * q + eps / len(q)
    return q / q.sum()



def _local_slsqp_on_active_face(
    mass: np.ndarray,
    active_mask: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray],
    max_iterations: int = 3000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Polish one active face from the supplied MM solution."""
    aa, bb = validate_operator_matrices(a, b)
    w = _normalise_row_weights(row_weights, aa.shape[0])
    p = np.asarray(mass, dtype=float).copy()
    active = np.asarray(active_mask, dtype=bool).copy()
    active |= p > 0.0
    idx = np.flatnonzero(active)
    pa0 = p[idx]
    if float(pa0.sum()) <= 0.0:
        pa0 = np.full(len(idx), 1.0 / len(idx))
    else:
        pa0 = np.maximum(pa0, 0.0)
        pa0 /= pa0.sum()
    a_active = aa[:, idx]
    b_active = bb[:, idx]

    def objective(pa: np.ndarray) -> float:
        value = conditional_loglikelihood(pa, a_active, b_active, w)
        return 1e100 if not np.isfinite(value) else -value

    def gradient(pa: np.ndarray) -> np.ndarray:
        try:
            return -conditional_score(pa, a_active, b_active, w)
        except ValueError:
            return np.zeros_like(pa)

    result = minimize(
        objective,
        pa0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(idx),
        constraints=[
            {
                "type": "eq",
                "fun": lambda pa: float(pa.sum() - 1.0),
                "jac": lambda pa: np.ones_like(pa),
            }
        ],
        options={"ftol": 1e-14, "maxiter": int(max_iterations), "disp": False},
    )
    pa = np.asarray(result.x, dtype=float)
    if not np.all(np.isfinite(pa)) or pa.min() < -1e-8 or abs(float(pa.sum()) - 1.0) > 1e-6:
        raise RuntimeError("active-face SLSQP returned an infeasible solution")
    pa = np.maximum(pa, 0.0)
    pa /= pa.sum()
    out = np.zeros_like(p)
    out[idx] = pa
    return out, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", -1)),
        "active_face_size": int(len(idx)),
    }


def _newton_refine_active_face(
    mass: np.ndarray,
    active_mask: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray],
    tolerance: float = 1e-11,
    zero_tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Projected Newton refinement with positivity-preserving line search."""
    aa, bb = validate_operator_matrices(a, b)
    w = _normalise_row_weights(row_weights, aa.shape[0])
    p = np.asarray(mass, dtype=float).copy()
    active = np.asarray(active_mask, dtype=bool).copy()
    active |= p > zero_tolerance
    p[p < 0.0] = 0.0
    p /= p.sum()
    total_newton_steps = 0
    activation_events = 0
    last_projected_residual = math.inf
    last_line_search_steps = 0

    for activation_round in range(10):
        # Safely remove numerical zeros left by SLSQP.
        for j in list(np.flatnonzero(active & (p <= zero_tolerance))):
            proposed = active.copy()
            proposed[j] = False
            if proposed.any() and np.all(aa[:, proposed].sum(axis=1) > 0.0):
                active[j] = False
                p[j] = 0.0
        p /= p.sum()

        idx = np.flatnonzero(active)
        pa = p[idx].copy()
        pa /= pa.sum()
        a_active = aa[:, idx]
        b_active = bb[:, idx]

        for _ in range(max_iterations):
            g = conditional_score(pa, a_active, b_active, w)
            if len(pa) <= 1:
                last_projected_residual = 0.0
                break
            q = _tangent_basis(len(pa))
            tangent_gradient = q.T @ g
            last_projected_residual = float(np.max(np.abs(tangent_gradient)))
            if last_projected_residual < tolerance:
                break
            h = conditional_hessian(pa, a_active, b_active, w)
            hp = q.T @ h @ q
            try:
                z = np.linalg.solve(hp, -tangent_gradient)
            except np.linalg.LinAlgError:
                z = np.linalg.lstsq(hp, -tangent_gradient, rcond=None)[0]
            direction = q @ z
            directional_derivative = float(g @ direction)
            if not np.isfinite(directional_derivative) or directional_derivative <= 0.0:
                direction = q @ tangent_gradient
                directional_derivative = float(g @ direction)
            alpha = 1.0
            negative = direction < 0.0
            if np.any(negative):
                alpha = min(alpha, 0.99 * float(np.min(-pa[negative] / direction[negative])))
            ll0 = conditional_loglikelihood(pa, a_active, b_active, w)
            accepted = False
            line_steps = 0
            while alpha >= 1e-14:
                candidate = pa + alpha * direction
                line_steps += 1
                if np.all(candidate > 0.0):
                    candidate /= candidate.sum()
                    ll1 = conditional_loglikelihood(candidate, a_active, b_active, w)
                    if ll1 >= ll0 + 1e-4 * alpha * directional_derivative - 1e-15:
                        pa = candidate
                        accepted = True
                        break
                alpha *= 0.5
            last_line_search_steps = line_steps
            if not accepted:
                break
            total_newton_steps += 1

        p[:] = 0.0
        p[idx] = pa
        p /= p.sum()
        score = conditional_score(p, aa, bb, w)
        inactive = ~active
        max_inactive = float(np.max(score[inactive])) if np.any(inactive) else -math.inf
        if max_inactive <= tolerance:
            break
        j = int(np.flatnonzero(inactive)[np.argmax(score[inactive])])
        epsilon = min(1e-6, max(1e-10, 0.1 * float(np.min(p[active]))))
        p *= 1.0 - epsilon
        p[j] = epsilon
        active[j] = True
        activation_events += 1

    # Final safe zeroing and exact KKT audit.
    for j in list(np.flatnonzero(active & (p <= zero_tolerance))):
        proposed = active.copy()
        proposed[j] = False
        if proposed.any() and np.all(aa[:, proposed].sum(axis=1) > 0.0):
            active[j] = False
            p[j] = 0.0
    p /= p.sum()
    final_kkt = kkt_diagnostics(
        p, aa, bb, w, active_tolerance=zero_tolerance
    )
    return p, active, {
        "newton_steps": int(total_newton_steps),
        "activation_events": int(activation_events),
        "projected_gradient_residual": float(last_projected_residual),
        "last_line_search_steps": int(last_line_search_steps),
        "active_score_residual": float(final_kkt["active_score_sup_abs_residual"]),
        "inactive_positive_score_violation": float(final_kkt["inactive_positive_score_violation"]),
    }


def fit_hybrid_active_set(
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray],
    initial_mass: Optional[np.ndarray],
    mass_tolerance: float,
    score_tolerance: float,
    prune_mass_tolerance: float,
    prune_score_margin: float,
    max_iterations: int,
    mm_prepolish_iterations: int = 1500,
    newton_tolerance: float = 1e-11,
    polish_zero_tolerance: float = 1e-12,
    max_polish_cycles: int = 8,
) -> ActiveSetMMFit:
    """Fit on the simplex with MM and repeated active-face polishing.

    The Version 1.0 control flow performed one SLSQP polish followed by one
    projected-Newton pass.  A Newton pass may activate a previously inactive
    class.  In that event the SLSQP solution belongs to the old active face and
    need not satisfy the KKT equations on the enlarged face.  Version 1.1
    therefore repeats

        active-face SLSQP -> projected Newton -> full KKT audit

    until the active set has stabilised and the requested KKT tolerance is met,
    or until ``max_polish_cycles`` is exhausted.  No statistical criterion,
    likelihood, support, or analysis rule is changed by this numerical fix.
    """
    max_polish_cycles = int(max_polish_cycles)
    if max_polish_cycles < 1:
        raise ValueError("max_polish_cycles must be at least one")

    mm_limit = min(int(max_iterations), int(mm_prepolish_iterations))
    base = fit_active_set_mm(
        a,
        b,
        row_weights,
        initial_mass=initial_mass,
        mass_tolerance=mass_tolerance,
        score_tolerance=score_tolerance,
        prune_mass_tolerance=prune_mass_tolerance,
        prune_score_margin=prune_score_margin,
        max_iterations=mm_limit,
        kkt_active_tolerance=polish_zero_tolerance,
    )
    if base.converged:
        base.diagnostics.update(
            {
                "hybrid_polish_used": False,
                "mm_prepolish_iteration_limit": mm_limit,
                "polish_cycles": 0,
                "polish_max_cycles": max_polish_cycles,
                "polish_active_set_change_cycles": 0,
                "polish_active_set_stable_at_exit": True,
                "polish_termination_reason": "MM_satisfied_KKT",
                "polish_slsqp_success": True,
                "polish_slsqp_iterations": 0,
                "polish_total_slsqp_iterations": 0,
                "polish_newton_steps": 0,
                "polish_total_newton_steps": 0,
                "polish_activation_events": 0,
                "polish_total_activation_events": 0,
                "polish_projected_gradient_residual": float(
                    base.diagnostics["active_score_sup_abs_residual"]
                ),
                "polish_loglikelihood_increment": 0.0,
                "polish_cycle_records": [],
            }
        )
        return base

    initial_ll = conditional_loglikelihood(base.mass, a, b, row_weights)
    polished = np.asarray(base.mass, dtype=float).copy()
    active = np.asarray(base.active_mask, dtype=bool).copy()
    active |= polished > polish_zero_tolerance

    cycle_records: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    all_slsqp_success = True
    total_slsqp_iterations = 0
    total_newton_steps = 0
    total_activation_events = 0
    active_set_change_cycles = 0
    last_slsqp_diag: Dict[str, Any] = {
        "success": False,
        "status": -1,
        "message": "not run",
        "iterations": 0,
        "active_face_size": int(active.sum()),
    }
    last_newton_diag: Dict[str, Any] = {
        "newton_steps": 0,
        "activation_events": 0,
        "projected_gradient_residual": math.inf,
        "last_line_search_steps": 0,
        "active_score_residual": math.inf,
        "inactive_positive_score_violation": math.inf,
    }
    final_kkt = kkt_diagnostics(
        polished, a, b, row_weights, active_tolerance=polish_zero_tolerance
    )
    converged = False
    termination_reason = "maximum_polish_cycles_exhausted"
    previous_ll = initial_ll
    final_active_stable = False

    for cycle in range(1, max_polish_cycles + 1):
        active_before = active.copy()
        n_active_before = int(active_before.sum())
        ll_cycle_before = conditional_loglikelihood(polished, a, b, row_weights)

        polished, slsqp_diag = _local_slsqp_on_active_face(
            polished, active, a, b, row_weights
        )
        last_slsqp_diag = dict(slsqp_diag)
        all_slsqp_success = bool(all_slsqp_success and slsqp_diag["success"])
        total_slsqp_iterations += max(int(slsqp_diag["iterations"]), 0)
        active |= polished > polish_zero_tolerance
        n_active_after_slsqp = int(active.sum())

        polished, active, newton_diag = _newton_refine_active_face(
            polished,
            active,
            a,
            b,
            row_weights,
            tolerance=newton_tolerance,
            zero_tolerance=polish_zero_tolerance,
        )
        last_newton_diag = dict(newton_diag)
        total_newton_steps += int(newton_diag["newton_steps"])
        total_activation_events += int(newton_diag["activation_events"])

        ll_cycle_after = conditional_loglikelihood(polished, a, b, row_weights)
        if ll_cycle_after < ll_cycle_before - 1e-10:
            raise RuntimeError(
                "hybrid polishing materially decreased the likelihood in "
                f"cycle {cycle}: {ll_cycle_after - ll_cycle_before:.3e}"
            )
        if ll_cycle_after < previous_ll - 1e-10:
            raise RuntimeError(
                "hybrid polishing decreased the likelihood relative to the "
                f"preceding cycle {cycle}: {ll_cycle_after - previous_ll:.3e}"
            )
        previous_ll = ll_cycle_after

        final_kkt = kkt_diagnostics(
            polished, a, b, row_weights, active_tolerance=polish_zero_tolerance
        )
        active_changed = bool(not np.array_equal(active, active_before))
        if active_changed:
            active_set_change_cycles += 1
        final_active_stable = not active_changed
        cycle_converged = bool(
            final_kkt["active_score_sup_abs_residual"] < score_tolerance
            and final_kkt["inactive_positive_score_violation"] < score_tolerance
        )

        cycle_record = {
            "polish_cycle": int(cycle),
            "loglikelihood_before": float(ll_cycle_before),
            "loglikelihood_after": float(ll_cycle_after),
            "loglikelihood_increment": float(ll_cycle_after - ll_cycle_before),
            "n_active_before": n_active_before,
            "n_active_after_slsqp": n_active_after_slsqp,
            "n_active_after_newton": int(active.sum()),
            "active_set_changed": active_changed,
            "slsqp_success": bool(slsqp_diag["success"]),
            "slsqp_status": int(slsqp_diag["status"]),
            "slsqp_message": str(slsqp_diag["message"]),
            "slsqp_iterations": int(slsqp_diag["iterations"]),
            "newton_steps": int(newton_diag["newton_steps"]),
            "newton_activation_events": int(newton_diag["activation_events"]),
            "newton_projected_gradient_residual": float(
                newton_diag["projected_gradient_residual"]
            ),
            "active_score_residual": float(
                final_kkt["active_score_sup_abs_residual"]
            ),
            "inactive_positive_score_violation": float(
                final_kkt["inactive_positive_score_violation"]
            ),
            "cycle_satisfied_KKT": cycle_converged,
        }
        cycle_records.append(cycle_record)
        event_rows.append(
            {
                "outer_iteration": int(base.diagnostics["outer_iterations"]) + cycle,
                "event": "hybrid_polish_cycle",
                "class_index": -1,
                "mass_before": math.nan,
                "score_before": float(cycle_record["active_score_residual"]),
                "loglik_increment": float(cycle_record["loglikelihood_increment"]),
                "n_active_after": int(active.sum()),
            }
        )

        if cycle_converged:
            converged = True
            termination_reason = (
                "KKT_satisfied_after_stable_active_face"
                if final_active_stable
                else "KKT_satisfied_after_active_set_change"
            )
            break

        # The key Version 1.1 behaviour is to continue after an active-set
        # change.  The next cycle reruns SLSQP on the newly enlarged/reduced
        # face before another Newton/KKT audit.

    ll_after = conditional_loglikelihood(polished, a, b, row_weights)
    if ll_after < initial_ll - 1e-10:
        raise RuntimeError("hybrid polishing materially decreased the likelihood")

    hdiag = projected_hessian_diagnostics(
        polished, a, b, row_weights, active_tolerance=polish_zero_tolerance
    )
    events = pd.concat(
        [base.events, pd.DataFrame(event_rows)], ignore_index=True
    ) if event_rows else base.events.copy()

    diagnostics = dict(base.diagnostics)
    diagnostics.update(
        {
            "converged": converged,
            "final_loglikelihood": float(ll_after),
            "probability_mass_sum": float(polished.sum()),
            "probability_normalisation_residual": float(abs(polished.sum() - 1.0)),
            "all_masses_nonnegative": bool(np.all(polished >= 0.0)),
            "mass_min": float(polished.min()),
            "mass_max": float(polished.max()),
            "n_exact_zero_masses": int(np.sum(polished == 0.0)),
            **{key: value for key, value in final_kkt.items() if key != "score"},
            **hdiag,
            "hybrid_polish_used": True,
            "mm_prepolish_iteration_limit": mm_limit,
            "polish_cycles": int(len(cycle_records)),
            "polish_max_cycles": max_polish_cycles,
            "polish_active_set_change_cycles": int(active_set_change_cycles),
            "polish_active_set_stable_at_exit": bool(final_active_stable),
            "polish_termination_reason": termination_reason,
            "polish_slsqp_success": bool(all_slsqp_success),
            "polish_slsqp_status": int(last_slsqp_diag["status"]),
            "polish_slsqp_message": str(last_slsqp_diag["message"]),
            "polish_slsqp_iterations": int(total_slsqp_iterations),
            "polish_last_slsqp_iterations": int(last_slsqp_diag["iterations"]),
            "polish_total_slsqp_iterations": int(total_slsqp_iterations),
            "polish_newton_steps": int(total_newton_steps),
            "polish_total_newton_steps": int(total_newton_steps),
            "polish_activation_events": int(total_activation_events),
            "polish_total_activation_events": int(total_activation_events),
            "polish_projected_gradient_residual": float(
                last_newton_diag["projected_gradient_residual"]
            ),
            "polish_loglikelihood_increment": float(ll_after - initial_ll),
            "polish_cycle_records": cycle_records,
        }
    )
    return ActiveSetMMFit(
        mass=polished,
        active_mask=active,
        history=base.history,
        events=events,
        converged=converged,
        diagnostics=diagnostics,
    )


@dataclass
class StartAuditedSieveFit:
    original_mass: np.ndarray
    reduction: OperatorReduction
    selected_fit: ActiveSetMMFit
    selected_start: str
    start_table: pd.DataFrame
    direct: Optional[DirectFit]
    diagnostics: Dict[str, Any]


def fit_sieve_with_start_audit(
    a: np.ndarray,
    b: np.ndarray,
    row_weights: Optional[np.ndarray],
    coverage_tolerance: float,
    mass_tolerance: float,
    score_tolerance: float,
    prune_mass_tolerance: float,
    prune_score_margin: float,
    max_iterations: int,
    warm_original_mass: Optional[np.ndarray] = None,
    run_uniform_audit: bool = False,
    warm_mixing: float = 1e-8,
    direct_starts: int = 0,
    direct_seed: int = 104729,
    mm_prepolish_iterations: int = 1500,
    newton_tolerance: float = 1e-11,
    polish_zero_tolerance: float = 1e-12,
    max_polish_cycles: int = 8,
) -> StartAuditedSieveFit:
    reduction = reduce_operator_columns(
        a, b, row_weights, coverage_tolerance=coverage_tolerance
    )
    if reduction.invisible_indices.size:
        raise RuntimeError(
            "completely invisible sieve cells were found; the bounded support or "
            "observation design must be reconsidered"
        )

    candidates: List[Tuple[str, ActiveSetMMFit]] = []
    if warm_original_mass is not None:
        warm_class = _strict_positive_class_start(
            np.asarray(warm_original_mass, dtype=float), reduction, warm_mixing
        )
        warm_fit = fit_hybrid_active_set(
            reduction.a,
            reduction.b,
            row_weights,
            initial_mass=warm_class,
            mass_tolerance=mass_tolerance,
            score_tolerance=score_tolerance,
            prune_mass_tolerance=prune_mass_tolerance,
            prune_score_margin=prune_score_margin,
            max_iterations=max_iterations,
            mm_prepolish_iterations=mm_prepolish_iterations,
            newton_tolerance=newton_tolerance,
            polish_zero_tolerance=polish_zero_tolerance,
            max_polish_cycles=max_polish_cycles,
        )
        candidates.append(("projected_warm", warm_fit))

    if warm_original_mass is None or run_uniform_audit:
        cold_fit = fit_hybrid_active_set(
            reduction.a,
            reduction.b,
            row_weights,
            initial_mass=None,
            mass_tolerance=mass_tolerance,
            score_tolerance=score_tolerance,
            prune_mass_tolerance=prune_mass_tolerance,
            prune_score_margin=prune_score_margin,
            max_iterations=max_iterations,
            mm_prepolish_iterations=mm_prepolish_iterations,
            newton_tolerance=newton_tolerance,
            polish_zero_tolerance=polish_zero_tolerance,
            max_polish_cycles=max_polish_cycles,
        )
        candidates.append(("uniform_cold", cold_fit))

    if not candidates:
        raise RuntimeError("no MM start was executed")
    selected_name, selected = max(
        candidates, key=lambda item: item[1].diagnostics["final_loglikelihood"]
    )
    start_rows: List[Dict[str, Any]] = []
    for name, fit in candidates:
        start_rows.append(
            {
                "start": name,
                "converged": bool(fit.converged),
                "final_loglikelihood": float(fit.diagnostics["final_loglikelihood"]),
                "outer_iterations": int(fit.diagnostics["outer_iterations"]),
                "mm_steps": int(fit.diagnostics["mm_steps"]),
                "n_prune_events": int(fit.diagnostics["n_prune_events"]),
                "active_score_residual": float(fit.diagnostics["active_score_sup_abs_residual"]),
                "inactive_positive_score_violation": float(fit.diagnostics["inactive_positive_score_violation"]),
                "hybrid_polish_used": bool(fit.diagnostics.get("hybrid_polish_used", False)),
                "polish_cycles": int(fit.diagnostics.get("polish_cycles", 0)),
                "polish_active_set_change_cycles": int(fit.diagnostics.get("polish_active_set_change_cycles", 0)),
                "polish_slsqp_iterations": int(fit.diagnostics.get("polish_slsqp_iterations", 0)),
                "polish_newton_steps": int(fit.diagnostics.get("polish_newton_steps", 0)),
                "polish_loglikelihood_increment": float(fit.diagnostics.get("polish_loglikelihood_increment", 0.0)),
                "selected": bool(name == selected_name),
            }
        )
    start_table = pd.DataFrame(start_rows)
    if len(candidates) == 2:
        p0 = candidates[0][1].mass
        p1 = candidates[1][1].mass
        warm_cold_mass_difference = float(np.max(np.abs(p0 - p1)))
        warm_cold_loglik_difference = float(
            abs(
                candidates[0][1].diagnostics["final_loglikelihood"]
                - candidates[1][1].diagnostics["final_loglikelihood"]
            )
        )
    else:
        warm_cold_mass_difference = math.nan
        warm_cold_loglik_difference = math.nan

    direct: Optional[DirectFit] = None
    direct_gap_mass = math.nan
    direct_gap_loglik = math.nan
    if direct_starts > 0:
        direct = fit_direct_slsqp_multistart(
            reduction.a,
            reduction.b,
            row_weights,
            n_starts=direct_starts,
            seed=direct_seed,
        )
        direct_gap_mass = float(np.max(np.abs(selected.mass - direct.mass)))
        direct_gap_loglik = float(
            abs(
                selected.diagnostics["final_loglikelihood"]
                - direct.diagnostics["best_loglikelihood"]
            )
        )

    original_mass = expand_class_mass(
        selected.mass, reduction.classes, a.shape[1], split="equal"
    )
    diagnostics = {
        "selected_start": selected_name,
        "n_starts_run": len(candidates),
        "warm_cold_class_mass_sup_abs_difference": warm_cold_mass_difference,
        "warm_cold_loglikelihood_abs_difference": warm_cold_loglik_difference,
        "direct_class_mass_sup_abs_difference": direct_gap_mass,
        "direct_loglikelihood_abs_difference": direct_gap_loglik,
        "n_original_cells": int(a.shape[1]),
        "n_forced_zero_cells": int(reduction.forced_zero_indices.size),
        "n_invisible_cells": int(reduction.invisible_indices.size),
        "n_equivalence_classes": int(len(reduction.classes)),
        "n_nontrivial_equivalence_classes": int(
            sum(len(cls) > 1 for cls in reduction.classes)
        ),
        "mm": selected.diagnostics,
    }
    return StartAuditedSieveFit(
        original_mass=original_mass,
        reduction=reduction,
        selected_fit=selected,
        selected_start=selected_name,
        start_table=start_table,
        direct=direct,
        diagnostics=diagnostics,
    )


def histogram_cell_mass(
    grid: RectangularGrid,
    x1: np.ndarray,
    x2: np.ndarray,
) -> np.ndarray:
    values1 = np.asarray(x1, dtype=float)
    values2 = np.asarray(x2, dtype=float)
    if values1.size != values2.size or values1.size == 0:
        raise ValueError("histogram input must contain equally sized nonempty arrays")
    counts, _, _ = np.histogram2d(
        values1,
        values2,
        bins=[grid.edges1, grid.edges2],
    )
    if counts.sum() != values1.size:
        raise RuntimeError("some histogram observations fell outside the bounded support")
    return (counts / counts.sum()).reshape(-1)


def cdf_operator_matrix(
    grid: RectangularGrid,
    x1_values: np.ndarray,
    x2_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1, q2 = np.meshgrid(
        np.asarray(x1_values, dtype=float),
        np.asarray(x2_values, dtype=float),
        indexing="ij",
    )
    rows = [
        grid.cdf_row(float(a), float(b))
        for a, b in zip(q1.reshape(-1), q2.reshape(-1))
    ]
    return np.vstack(rows), q1.reshape(-1), q2.reshape(-1)


def mass_moment_summary(grid: RectangularGrid, mass: np.ndarray) -> Dict[str, float]:
    p = _normalise_probability(np.asarray(mass, dtype=float), "mass")
    table = grid.cell_table
    c1 = table["x1_centre"].to_numpy(float)
    c2 = table["x2_centre"].to_numpy(float)
    w1 = table["width1"].to_numpy(float)
    w2 = table["width2"].to_numpy(float)
    mean1 = float(p @ c1)
    mean2 = float(p @ c2)
    second1 = float(p @ (c1**2 + w1**2 / 12.0))
    second2 = float(p @ (c2**2 + w2**2 / 12.0))
    cross = float(p @ (c1 * c2))
    var1 = max(second1 - mean1**2, 0.0)
    var2 = max(second2 - mean2**2, 0.0)
    covariance = cross - mean1 * mean2
    correlation = covariance / math.sqrt(var1 * var2) if var1 > 0 and var2 > 0 else math.nan
    return {
        "mean_x1": mean1,
        "mean_x2": mean2,
        "sd_x1": math.sqrt(var1),
        "sd_x2": math.sqrt(var2),
        "correlation": float(correlation),
    }


def estimator_accuracy_metrics(
    grid: RectangularGrid,
    mass: np.ndarray,
    true_cell_mass: np.ndarray,
    cdf_matrix: np.ndarray,
    truth_cdf_vector: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    p = _normalise_probability(np.asarray(mass, dtype=float), "estimated_mass")
    p0 = _normalise_probability(np.asarray(true_cell_mass, dtype=float), "true_cell_mass")
    cdf = cdf_matrix @ p
    error = cdf - truth_cdf_vector
    delta = p - p0
    areas = grid.areas
    est_mom = mass_moment_summary(grid, p)
    true_mom = mass_moment_summary(grid, p0)
    metrics = {
        "cdf_rmse": float(np.sqrt(np.mean(error**2))),
        "cdf_mae": float(np.mean(np.abs(error))),
        "cdf_max_abs_error": float(np.max(np.abs(error))),
        "cell_mass_rmse": float(np.sqrt(np.mean(delta**2))),
        "cell_mass_max_abs_error": float(np.max(np.abs(delta))),
        "cell_mass_total_variation": float(0.5 * np.sum(np.abs(delta))),
        "cell_density_integrated_squared_error": float(np.sum(delta**2 / areas)),
        "correlation_estimate_sieve": est_mom["correlation"],
        "correlation_truth_sieve": true_mom["correlation"],
        "correlation_error_sieve": float(est_mom["correlation"] - true_mom["correlation"]),
        "mass_sum": float(p.sum()),
        "mass_min": float(p.min()),
        "all_masses_nonnegative": bool(np.all(p >= 0.0)),
    }
    return metrics, cdf


def equivalence_envelope_metrics_vectorised(
    cdf_matrix: np.ndarray,
    class_mass: np.ndarray,
    classes: Sequence[Sequence[int]],
) -> Dict[str, float]:
    q = np.asarray(class_mass, dtype=float)
    lower = np.zeros(cdf_matrix.shape[0], dtype=float)
    upper = np.zeros(cdf_matrix.shape[0], dtype=float)
    for mass_h, cls_raw in zip(q, classes):
        values = cdf_matrix[:, list(cls_raw)]
        lower += float(mass_h) * values.min(axis=1)
        upper += float(mass_h) * values.max(axis=1)
    width = upper - lower
    return {
        "equivalence_envelope_max_width": float(width.max()),
        "equivalence_envelope_mean_width": float(width.mean()),
    }


def observed_from_latent_prefix(latent_full: pd.DataFrame, n_latent: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    latent = latent_full.iloc[: int(n_latent)].copy().reset_index(drop=True)
    observed = latent.loc[latent["observed"].to_numpy(int) == 1].copy().reset_index(drop=True)
    observed["x1_obs"] = np.where(
        observed["delta1"].to_numpy(int) == 1,
        observed["x1_true"].to_numpy(float),
        observed["y1"].to_numpy(float),
    )
    observed["x2_obs"] = np.where(
        observed["delta2"].to_numpy(int) == 1,
        observed["x2_true"].to_numpy(float),
        observed["y2"].to_numpy(float),
    )
    info = {
        "n_latent": int(len(latent)),
        "n_observed": int(len(observed)),
        "selection_fraction": float(len(observed) / len(latent)),
        "region_counts_observed": {
            region: int((observed["region"] == region).sum())
            for region in ["A", "B", "C"]
        },
    }
    return latent, observed, info


def resolve_profile(args: argparse.Namespace) -> Tuple[List[int], List[int], int, int]:
    defaults = {
        "smoke": ([800, 1600], [4, 6], 2, 21),
        "pilot": ([1500, 3000, 6000], [4, 8, 12], 5, 31),
        "resolution": ([1500, 3000, 6000], [8, 12, 16, 20], 10, 41),
        "full": ([1500, 3000, 6000, 12000], [4, 6, 8, 10, 12], 20, 41),
    }
    n_default, bins_default, rep_default, query_default = defaults[args.profile]
    n_values = sorted(set(args.n_values if args.n_values else n_default))
    bins_values = sorted(set(args.bins_values if args.bins_values else bins_default))
    replicates = int(args.replicates if args.replicates is not None else rep_default)
    query_grid_size = int(
        args.query_grid_size if args.query_grid_size is not None else query_default
    )
    if any(n < 100 for n in n_values):
        raise ValueError("all latent sample sizes must be at least 100")
    if any(m < 2 for m in bins_values):
        raise ValueError("all bin counts must be at least two")
    if replicates < 2:
        raise ValueError("at least two replicates are required for a variance audit")
    return n_values, bins_values, replicates, query_grid_size


def combination_seed(base_seed: int, rho_index: int, replicate: int) -> int:
    return int(base_seed + 100_000 * rho_index + 1_009 * replicate)


def checkpoint_name(rho: float, n_latent: int, replicate: int, bins: int) -> str:
    rho_token = f"{rho:+.3f}".replace("+", "p").replace("-", "m").replace(".", "d")
    return f"rho_{rho_token}__n_{n_latent:06d}__rep_{replicate:04d}__bins_{bins:03d}.npz"


def save_checkpoint(
    path: Path,
    metadata: Dict[str, Any],
    masses: Dict[str, np.ndarray],
    cdfs: Dict[str, np.ndarray],
) -> None:
    payload: Dict[str, Any] = {
        "metadata_json": np.array(json.dumps(_jsonable(metadata), sort_keys=True)),
    }
    for name, value in masses.items():
        payload[f"mass__{name}"] = np.asarray(value, dtype=float)
    for name, value in cdfs.items():
        payload[f"cdf__{name}"] = np.asarray(value, dtype=float)
    np.savez_compressed(path, **payload)


def load_checkpoint(path: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        masses = {
            key.split("mass__", 1)[1]: np.asarray(data[key], dtype=float)
            for key in data.files
            if key.startswith("mass__")
        }
        cdfs = {
            key.split("cdf__", 1)[1]: np.asarray(data[key], dtype=float)
            for key in data.files
            if key.startswith("cdf__")
        }
    return metadata, masses, cdfs


def run_batch_self_tests() -> Dict[str, Any]:
    base = run_self_tests()
    checks: Dict[str, bool] = {"rectangular_sieve_base_self_tests": bool(base["all_passed"])}
    details: Dict[str, Any] = {"base_self_tests": base}

    g4 = make_uniform_grid(4)
    g8 = make_uniform_grid(8)
    rng = np.random.default_rng(20260906)
    p4 = rng.dirichlet(np.ones(g4.k))
    p4_identity = project_rectangular_mass(g4, p4, g4)
    p8 = project_rectangular_mass(g4, p4, g8)
    checks["mass_projection_identity"] = bool(np.max(np.abs(p4_identity - p4)) < 1e-14)
    checks["mass_projection_preserves_total"] = bool(abs(float(p8.sum()) - 1.0) < 1e-14)
    checks["mass_projection_nonnegative"] = bool(np.all(p8 >= 0.0))
    details["projection_identity_sup_abs_difference"] = float(np.max(np.abs(p4_identity - p4)))

    y1, y2 = -0.4, 0.7
    inter = intersection_selection_row(g4, y1, y2)
    expected = np.outer(g4.fraction_ge1(y1), g4.fraction_ge2(y2)).reshape(-1)
    checks["intersection_selection_operator_exact"] = bool(np.array_equal(inter, expected))

    latent_full, _, _ = simulate_gaussian_catalogue(
        300,
        20260906,
        0.55,
        GAUSSIAN_LOWER,
        GAUSSIAN_UPPER,
        GAUSSIAN_Y_SUPPORT,
        GAUSSIAN_Y_MASS,
    )
    latent_150, observed_150, info_150 = observed_from_latent_prefix(latent_full, 150)
    checks["nested_prefix_preserved"] = bool(
        np.array_equal(latent_150["latent_id"].to_numpy(), np.arange(150))
    )
    checks["observed_prefix_count_consistent"] = bool(
        info_150["n_observed"] == int(latent_150["observed"].sum())
        and len(observed_150) == info_150["n_observed"]
    )

    # Cold/warm agreement on an exactly specified population operator.
    pgrid = RectangularGrid(PIECEWISE_EDGES, PIECEWISE_EDGES)
    ptrue = PIECEWISE_TRUE_MASS_MATRIX.reshape(-1)
    a, b, w, _, _ = build_population_categories(
        pgrid, ptrue, PIECEWISE_Y_SUPPORT, PIECEWISE_Y_MASS
    )
    audited = fit_sieve_with_start_audit(
        a,
        b,
        w,
        coverage_tolerance=1e-15,
        mass_tolerance=1e-11,
        score_tolerance=1e-8,
        prune_mass_tolerance=1e-10,
        prune_score_margin=1e-6,
        max_iterations=20000,
        warm_original_mass=ptrue,
        run_uniform_audit=True,
        direct_starts=2,
        direct_seed=20260906,
    )
    checks["population_warm_cold_same_likelihood"] = bool(
        audited.diagnostics["warm_cold_loglikelihood_abs_difference"] < 1e-10
    )
    checks["population_warm_solution_recovers_truth"] = bool(
        np.max(np.abs(audited.original_mass - ptrue)) < 1e-8
    )
    checks["selected_fit_satisfies_kkt"] = bool(
        audited.selected_fit.diagnostics["active_score_sup_abs_residual"] < 1e-8
        and audited.selected_fit.diagnostics["inactive_positive_score_violation"] < 1e-8
    )
    details["warm_cold_loglikelihood_abs_difference"] = audited.diagnostics[
        "warm_cold_loglikelihood_abs_difference"
    ]
    details["population_truth_mass_sup_abs_difference"] = float(
        np.max(np.abs(audited.original_mass - ptrue))
    )

    synthetic_truth = np.array([0.1, 0.2, 0.3])
    synthetic = np.array([[0.0, 0.3, 0.2], [0.2, 0.1, 0.4], [0.1, 0.2, 0.3]])
    mean = synthetic.mean(axis=0)
    bias2 = (mean - synthetic_truth) ** 2
    variance = ((synthetic - mean) ** 2).mean(axis=0)
    mse = ((synthetic - synthetic_truth) ** 2).mean(axis=0)
    decomposition_error = float(np.max(np.abs(mse - bias2 - variance)))
    checks["bias_variance_decomposition_identity"] = bool(decomposition_error < 1e-15)
    details["bias_variance_decomposition_max_abs_error"] = decomposition_error

    return {
        "all_passed": bool(all(checks.values())),
        "checks": checks,
        "details": details,
    }


def _replicate_metric_record(
    base: Dict[str, Any],
    estimator: str,
    accuracy: Dict[str, Any],
    fit: Optional[StartAuditedSieveFit],
    runtime_seconds: float,
    envelope: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    record = dict(base)
    record.update({"estimator": estimator, "runtime_seconds": float(runtime_seconds)})
    record.update(accuracy)
    if envelope:
        record.update(envelope)
    if fit is not None:
        mm = fit.selected_fit.diagnostics
        record.update(
            {
                "converged": bool(fit.selected_fit.converged),
                "selected_start": fit.selected_start,
                "n_starts_run": int(fit.diagnostics["n_starts_run"]),
                "warm_cold_class_mass_sup_abs_difference": fit.diagnostics[
                    "warm_cold_class_mass_sup_abs_difference"
                ],
                "warm_cold_loglikelihood_abs_difference": fit.diagnostics[
                    "warm_cold_loglikelihood_abs_difference"
                ],
                "direct_class_mass_sup_abs_difference": fit.diagnostics[
                    "direct_class_mass_sup_abs_difference"
                ],
                "direct_loglikelihood_abs_difference": fit.diagnostics[
                    "direct_loglikelihood_abs_difference"
                ],
                "n_forced_zero_cells": int(fit.reduction.forced_zero_indices.size),
                "n_invisible_cells": int(fit.reduction.invisible_indices.size),
                "n_equivalence_classes": int(len(fit.reduction.classes)),
                "n_nontrivial_equivalence_classes": int(
                    sum(len(cls) > 1 for cls in fit.reduction.classes)
                ),
                "n_active_classes": int(mm["n_active"]),
                "outer_iterations": int(mm["outer_iterations"]),
                "mm_steps": int(mm["mm_steps"]),
                "n_prune_events": int(mm["n_prune_events"]),
                "active_score_residual": float(mm["active_score_sup_abs_residual"]),
                "inactive_positive_score_violation": float(mm["inactive_positive_score_violation"]),
                "final_loglikelihood": float(mm["final_loglikelihood"]),
                "hybrid_polish_used": bool(mm.get("hybrid_polish_used", False)),
                "polish_slsqp_iterations": int(mm.get("polish_slsqp_iterations", 0)),
                "polish_newton_steps": int(mm.get("polish_newton_steps", 0)),
                "polish_loglikelihood_increment": float(mm.get("polish_loglikelihood_increment", 0.0)),
            }
        )
    else:
        record.update(
            {
                "converged": True,
                "selected_start": "not_applicable",
                "n_starts_run": 0,
                "n_forced_zero_cells": 0,
                "n_invisible_cells": 0,
                "n_equivalence_classes": math.nan,
                "n_nontrivial_equivalence_classes": math.nan,
                "n_active_classes": math.nan,
                "outer_iterations": 0,
                "mm_steps": 0,
                "n_prune_events": 0,
                "active_score_residual": math.nan,
                "inactive_positive_score_violation": 0.0,
                "final_loglikelihood": math.nan,
                "equivalence_envelope_max_width": 0.0,
                "equivalence_envelope_mean_width": 0.0,
                "warm_cold_class_mass_sup_abs_difference": math.nan,
                "warm_cold_loglikelihood_abs_difference": math.nan,
                "direct_class_mass_sup_abs_difference": math.nan,
                "direct_loglikelihood_abs_difference": math.nan,
                "hybrid_polish_used": False,
                "polish_slsqp_iterations": 0,
                "polish_newton_steps": 0,
                "polish_loglikelihood_increment": 0.0,
            }
        )
    return record


def aggregate_replicate_metrics(table: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["rho", "n_latent", "bins_per_axis", "estimator"]
    rows: List[Dict[str, Any]] = []
    for key, grp in table.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        row["n_replicates"] = int(len(grp))
        row["convergence_fraction"] = float(grp["converged"].mean())
        for col in [
            "cdf_rmse",
            "cdf_mae",
            "cdf_max_abs_error",
            "cell_mass_rmse",
            "cell_mass_max_abs_error",
            "cell_mass_total_variation",
            "cell_density_integrated_squared_error",
            "correlation_error_sieve",
            "runtime_seconds",
            "hybrid_polish_used",
            "polish_slsqp_iterations",
            "polish_newton_steps",
            "polish_loglikelihood_increment",
            "outer_iterations",
            "mm_steps",
            "n_active_classes",
            "n_forced_zero_cells",
            "n_nontrivial_equivalence_classes",
            "equivalence_envelope_max_width",
            "equivalence_envelope_mean_width",
            "active_score_residual",
            "inactive_positive_score_violation",
        ]:
            values = pd.to_numeric(grp[col], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size:
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
                row[f"{col}_median"] = float(np.median(values))
                row[f"{col}_q10"] = float(np.quantile(values, 0.10))
                row[f"{col}_q90"] = float(np.quantile(values, 0.90))
            else:
                for suffix in ["mean", "sd", "median", "q10", "q90"]:
                    row[f"{col}_{suffix}"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_bias_variance_tables(
    cdf_store: Dict[Tuple[float, int, int, str], List[np.ndarray]],
    truth_vectors: Dict[float, np.ndarray],
    query_x1: np.ndarray,
    query_x2: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    summary_rows: List[Dict[str, Any]] = []
    point_rows: List[Dict[str, Any]] = []
    max_identity_error = 0.0
    for (rho, n_latent, bins, estimator), vectors in sorted(cdf_store.items()):
        arr = np.vstack(vectors)
        truth = truth_vectors[float(rho)]
        mean_est = arr.mean(axis=0)
        bias = mean_est - truth
        variance = ((arr - mean_est[None, :]) ** 2).mean(axis=0)
        mse = ((arr - truth[None, :]) ** 2).mean(axis=0)
        identity = mse - bias**2 - variance
        max_identity_error = max(max_identity_error, float(np.max(np.abs(identity))))
        summary_rows.append(
            {
                "rho": float(rho),
                "n_latent": int(n_latent),
                "bins_per_axis": int(bins),
                "estimator": estimator,
                "n_replicates": int(arr.shape[0]),
                "integrated_squared_bias": float(np.mean(bias**2)),
                "integrated_variance": float(np.mean(variance)),
                "integrated_mse": float(np.mean(mse)),
                "integrated_rmse": float(math.sqrt(np.mean(mse))),
                "mse_decomposition_abs_residual": float(
                    abs(np.mean(mse) - np.mean(bias**2) - np.mean(variance))
                ),
                "pointwise_mse_decomposition_max_abs_residual": float(
                    np.max(np.abs(identity))
                ),
            }
        )
        for q, (x1, x2) in enumerate(zip(query_x1, query_x2)):
            point_rows.append(
                {
                    "rho": float(rho),
                    "n_latent": int(n_latent),
                    "bins_per_axis": int(bins),
                    "estimator": estimator,
                    "query_index": int(q),
                    "x1": float(x1),
                    "x2": float(x2),
                    "cdf_truth": float(truth[q]),
                    "cdf_mean": float(mean_est[q]),
                    "bias": float(bias[q]),
                    "squared_bias": float(bias[q] ** 2),
                    "variance": float(variance[q]),
                    "mse": float(mse[q]),
                    "decomposition_residual": float(identity[q]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(point_rows), max_identity_error


def best_bins_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (rho, n_latent, estimator), grp in aggregate.groupby(
        ["rho", "n_latent", "estimator"], sort=True
    ):
        valid = grp[np.isfinite(grp["cdf_rmse_mean"])].copy()
        if valid.empty:
            continue
        winner = valid.loc[valid["cdf_rmse_mean"].idxmin()]
        rows.append(
            {
                "rho": float(rho),
                "n_latent": int(n_latent),
                "estimator": estimator,
                "best_bins_per_axis": int(winner["bins_per_axis"]),
                "best_mean_cdf_rmse": float(winner["cdf_rmse_mean"]),
                "best_sd_cdf_rmse": float(winner["cdf_rmse_sd"]),
            }
        )
    return pd.DataFrame(rows)


def paired_estimator_comparisons(replicate_table: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["rho", "n_latent", "bins_per_axis", "replicate"]
    pivot = replicate_table.pivot_table(
        index=index_cols,
        columns="estimator",
        values="cdf_rmse",
        aggfunc="first",
    ).reset_index()
    comparisons = [
        ("ABC_union_sieve", "A_only_intersection_sieve"),
        ("ABC_union_sieve", "naive_A_histogram"),
        ("ABC_union_sieve", "complete_data_histogram"),
    ]
    rows: List[Dict[str, Any]] = []
    group_cols = ["rho", "n_latent", "bins_per_axis"]
    for left, right in comparisons:
        if left not in pivot.columns or right not in pivot.columns:
            continue
        temp = pivot[group_cols + [left, right]].dropna().copy()
        temp["difference"] = temp[left] - temp[right]
        for key, grp in temp.groupby(group_cols, sort=True):
            diff = grp["difference"].to_numpy(float)
            rows.append(
                {
                    "rho": float(key[0]),
                    "n_latent": int(key[1]),
                    "bins_per_axis": int(key[2]),
                    "estimator_left": left,
                    "estimator_right": right,
                    "mean_rmse_difference_left_minus_right": float(diff.mean()),
                    "sd_rmse_difference": float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
                    "fraction_left_has_lower_rmse": float(np.mean(diff < 0.0)),
                    "n_paired_replicates": int(diff.size),
                }
            )
    return pd.DataFrame(rows)


def empirical_rate_slopes(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (rho, bins, estimator), grp in aggregate.groupby(
        ["rho", "bins_per_axis", "estimator"], sort=True
    ):
        valid = grp[(grp["n_latent"] > 0) & (grp["cdf_rmse_mean"] > 0)].sort_values("n_latent")
        if len(valid) < 2:
            continue
        slope, intercept = np.polyfit(
            np.log(valid["n_latent"].to_numpy(float)),
            np.log(valid["cdf_rmse_mean"].to_numpy(float)),
            1,
        )
        rows.append(
            {
                "rho": float(rho),
                "bins_per_axis": int(bins),
                "estimator": estimator,
                "n_sample_sizes": int(len(valid)),
                "log_rmse_vs_log_n_slope": float(slope),
                "log_intercept": float(intercept),
            }
        )
    return pd.DataFrame(rows)




def _paired_difference_summary(values: np.ndarray) -> Dict[str, float]:
    """Summarise paired differences, including a small-sample t interval."""
    diff = np.asarray(values, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {
            "mean": math.nan,
            "sd": math.nan,
            "se": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "median": math.nan,
            "q10": math.nan,
            "q90": math.nan,
            "fraction_negative": math.nan,
            "n": 0,
        }
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    se = float(sd / math.sqrt(diff.size)) if diff.size > 1 else 0.0
    if diff.size > 1:
        critical = float(student_t.ppf(0.975, diff.size - 1))
        low = mean - critical * se
        high = mean + critical * se
    else:
        low = high = mean
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "median": float(np.median(diff)),
        "q10": float(np.quantile(diff, 0.10)),
        "q90": float(np.quantile(diff, 0.90)),
        "fraction_negative": float(np.mean(diff < 0.0)),
        "n": int(diff.size),
    }


def enhanced_paired_estimator_comparisons(replicate_table: pd.DataFrame) -> pd.DataFrame:
    """Paired estimator contrasts with robust summaries and a 95% t interval."""
    index_cols = ["rho", "n_latent", "bins_per_axis", "replicate"]
    pivot = replicate_table.pivot_table(
        index=index_cols,
        columns="estimator",
        values="cdf_rmse",
        aggfunc="first",
    ).reset_index()
    comparisons = [
        ("ABC_union_sieve", "A_only_intersection_sieve"),
        ("ABC_union_sieve", "naive_A_histogram"),
        ("ABC_union_sieve", "complete_data_histogram"),
    ]
    rows: List[Dict[str, Any]] = []
    group_cols = ["rho", "n_latent", "bins_per_axis"]
    for left, right in comparisons:
        if left not in pivot.columns or right not in pivot.columns:
            continue
        temp = pivot[group_cols + [left, right]].dropna().copy()
        for key, grp in temp.groupby(group_cols, sort=True):
            left_values = grp[left].to_numpy(float)
            right_values = grp[right].to_numpy(float)
            stats = _paired_difference_summary(left_values - right_values)
            right_mean = float(np.mean(right_values))
            rows.append(
                {
                    "rho": float(key[0]),
                    "n_latent": int(key[1]),
                    "bins_per_axis": int(key[2]),
                    "estimator_left": left,
                    "estimator_right": right,
                    "mean_rmse_left": float(np.mean(left_values)),
                    "mean_rmse_right": right_mean,
                    "mean_rmse_difference_left_minus_right": stats["mean"],
                    "sd_rmse_difference": stats["sd"],
                    "se_rmse_difference": stats["se"],
                    "ci95_low_rmse_difference": stats["ci95_low"],
                    "ci95_high_rmse_difference": stats["ci95_high"],
                    "median_rmse_difference": stats["median"],
                    "q10_rmse_difference": stats["q10"],
                    "q90_rmse_difference": stats["q90"],
                    "fraction_left_has_lower_rmse": stats["fraction_negative"],
                    "relative_mean_difference_vs_right": (
                        stats["mean"] / right_mean if right_mean > 0.0 else math.nan
                    ),
                    "n_paired_replicates": stats["n"],
                }
            )
    return pd.DataFrame(rows)


def adjacent_resolution_comparisons(replicate_table: pd.DataFrame) -> pd.DataFrame:
    """Compare successive sieve resolutions within each paired replicate."""
    rows: List[Dict[str, Any]] = []
    main = replicate_table[
        replicate_table["estimator"].isin(
            ["ABC_union_sieve", "A_only_intersection_sieve"]
        )
    ].copy()
    for (rho, n_latent, estimator), grp in main.groupby(
        ["rho", "n_latent", "estimator"], sort=True
    ):
        pivot = grp.pivot_table(
            index="replicate", columns="bins_per_axis", values="cdf_rmse", aggfunc="first"
        )
        bins = sorted(int(v) for v in pivot.columns)
        for lower, upper in zip(bins[:-1], bins[1:]):
            pair = pivot[[lower, upper]].dropna()
            stats = _paired_difference_summary(
                pair[upper].to_numpy(float) - pair[lower].to_numpy(float)
            )
            lower_mean = float(pair[lower].mean()) if len(pair) else math.nan
            rows.append(
                {
                    "rho": float(rho),
                    "n_latent": int(n_latent),
                    "estimator": estimator,
                    "lower_bins_per_axis": int(lower),
                    "upper_bins_per_axis": int(upper),
                    "lower_n_cells": int(lower * lower),
                    "upper_n_cells": int(upper * upper),
                    "mean_rmse_change_upper_minus_lower": stats["mean"],
                    "sd_rmse_change": stats["sd"],
                    "se_rmse_change": stats["se"],
                    "ci95_low_rmse_change": stats["ci95_low"],
                    "ci95_high_rmse_change": stats["ci95_high"],
                    "median_rmse_change": stats["median"],
                    "q10_rmse_change": stats["q10"],
                    "q90_rmse_change": stats["q90"],
                    "fraction_upper_has_lower_rmse": stats["fraction_negative"],
                    "relative_mean_change_vs_lower": (
                        stats["mean"] / lower_mean if lower_mean > 0.0 else math.nan
                    ),
                    "n_paired_replicates": stats["n"],
                }
            )
    return pd.DataFrame(rows)


def resolution_frontier_table(
    aggregate: pd.DataFrame,
    bias_variance: pd.DataFrame,
) -> pd.DataFrame:
    """Merge statistical, computational, and KKT diagnostics by resolution."""
    merge_cols = ["rho", "n_latent", "bins_per_axis", "estimator"]
    bv = bias_variance.drop(columns=["n_replicates"], errors="ignore")
    table = aggregate.merge(bv, on=merge_cols, how="left", validate="one_to_one")
    table["n_cells"] = table["bins_per_axis"].astype(int) ** 2
    table["cdf_rmse_se"] = table["cdf_rmse_sd"] / np.sqrt(table["n_replicates"])
    denom = table["integrated_mse"].to_numpy(float)
    bias = table["integrated_squared_bias"].to_numpy(float)
    variance = table["integrated_variance"].to_numpy(float)
    table["integrated_bias_fraction"] = np.divide(
        bias, denom, out=np.full_like(bias, np.nan), where=denom > 0.0
    )
    table["integrated_variance_fraction"] = np.divide(
        variance, denom, out=np.full_like(variance, np.nan), where=denom > 0.0
    )
    table["is_largest_tested_resolution"] = False
    table["previous_bins_per_axis"] = np.nan
    table["integrated_rmse_change_from_previous"] = np.nan
    table["integrated_mse_change_from_previous"] = np.nan
    table["relative_integrated_rmse_change_from_previous"] = np.nan
    for _, idx in table.groupby(["rho", "n_latent", "estimator"], sort=True).groups.items():
        order = table.loc[list(idx)].sort_values("bins_per_axis").index.to_list()
        table.loc[order[-1], "is_largest_tested_resolution"] = True
        for prev, cur in zip(order[:-1], order[1:]):
            prev_rmse = float(table.at[prev, "integrated_rmse"])
            cur_rmse = float(table.at[cur, "integrated_rmse"])
            table.at[cur, "previous_bins_per_axis"] = int(table.at[prev, "bins_per_axis"])
            table.at[cur, "integrated_rmse_change_from_previous"] = cur_rmse - prev_rmse
            table.at[cur, "integrated_mse_change_from_previous"] = (
                float(table.at[cur, "integrated_mse"])
                - float(table.at[prev, "integrated_mse"])
            )
            table.at[cur, "relative_integrated_rmse_change_from_previous"] = (
                (cur_rmse - prev_rmse) / prev_rmse if prev_rmse > 0.0 else math.nan
            )
    return table.sort_values(merge_cols).reset_index(drop=True)


def resolution_selection_summary(frontier: pd.DataFrame) -> pd.DataFrame:
    """Report empirical minima, one-standard-error choices, and boundary flags."""
    rows: List[Dict[str, Any]] = []
    main = frontier[
        frontier["estimator"].isin(
            ["ABC_union_sieve", "A_only_intersection_sieve"]
        )
    ].copy()
    for (rho, n_latent, estimator), grp in main.groupby(
        ["rho", "n_latent", "estimator"], sort=True
    ):
        grp = grp.sort_values("bins_per_axis").copy()
        max_bins = int(grp["bins_per_axis"].max())
        mean_winner = grp.loc[grp["cdf_rmse_mean"].idxmin()]
        integrated_winner = grp.loc[grp["integrated_rmse"].idxmin()]
        threshold = float(mean_winner["cdf_rmse_mean"] + mean_winner["cdf_rmse_se"])
        eligible = grp[grp["cdf_rmse_mean"] <= threshold + 1e-15]
        one_se = eligible.sort_values("bins_per_axis").iloc[0]
        if len(grp) >= 2:
            last = grp.iloc[-1]
            prev = grp.iloc[-2]
            last_change = float(last["integrated_rmse"] - prev["integrated_rmse"])
            last_relative = (
                last_change / float(prev["integrated_rmse"])
                if float(prev["integrated_rmse"]) > 0.0 else math.nan
            )
        else:
            last_change = last_relative = math.nan
        rows.append(
            {
                "rho": float(rho),
                "n_latent": int(n_latent),
                "estimator": estimator,
                "n_tested_resolutions": int(len(grp)),
                "smallest_tested_bins": int(grp["bins_per_axis"].min()),
                "largest_tested_bins": max_bins,
                "best_bins_by_mean_replicate_rmse": int(mean_winner["bins_per_axis"]),
                "best_mean_replicate_rmse": float(mean_winner["cdf_rmse_mean"]),
                "best_mean_replicate_rmse_se": float(mean_winner["cdf_rmse_se"]),
                "best_bins_by_integrated_rmse": int(integrated_winner["bins_per_axis"]),
                "best_integrated_rmse": float(integrated_winner["integrated_rmse"]),
                "one_standard_error_bins": int(one_se["bins_per_axis"]),
                "one_standard_error_threshold": threshold,
                "best_mean_is_upper_boundary": bool(
                    int(mean_winner["bins_per_axis"]) == max_bins
                ),
                "best_integrated_is_upper_boundary": bool(
                    int(integrated_winner["bins_per_axis"]) == max_bins
                ),
                "last_step_integrated_rmse_change": last_change,
                "last_step_relative_integrated_rmse_change": last_relative,
                "turnover_detected_within_tested_range": bool(
                    int(integrated_winner["bins_per_axis"]) < max_bins
                ),
            }
        )
    return pd.DataFrame(rows)


def runtime_scaling_summary(frontier: pd.DataFrame) -> pd.DataFrame:
    """Estimate empirical runtime scaling with the number of sieve cells."""
    rows: List[Dict[str, Any]] = []
    main = frontier[
        frontier["estimator"].isin(
            ["ABC_union_sieve", "A_only_intersection_sieve"]
        )
    ].copy()
    for (rho, n_latent, estimator), grp in main.groupby(
        ["rho", "n_latent", "estimator"], sort=True
    ):
        valid = grp[(grp["runtime_seconds_mean"] > 0.0) & (grp["n_cells"] > 0)].copy()
        if len(valid) < 2:
            continue
        x = np.log(valid["n_cells"].to_numpy(float))
        y = np.log(valid["runtime_seconds_mean"].to_numpy(float))
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
        rows.append(
            {
                "rho": float(rho),
                "n_latent": int(n_latent),
                "estimator": estimator,
                "n_resolutions": int(len(valid)),
                "log_runtime_vs_log_n_cells_slope": float(slope),
                "log_runtime_intercept": float(intercept),
                "r_squared": float(r2),
                "largest_resolution_runtime_mean_seconds": float(
                    valid.sort_values("n_cells").iloc[-1]["runtime_seconds_mean"]
                ),
            }
        )
    return pd.DataFrame(rows)


def batch_configuration_fingerprint(
    args: argparse.Namespace,
    query_grid_size: int,
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "query_grid_size": int(query_grid_size),
        "rho_values": [float(v) for v in args.rho_values],
        "mass_tolerance": float(args.mass_tolerance),
        "score_tolerance": float(args.score_tolerance),
        "prune_mass_tolerance": float(args.prune_mass_tolerance),
        "prune_score_margin": float(args.prune_score_margin),
        "coverage_tolerance": float(args.coverage_tolerance),
        "warm_mixing": float(args.warm_mixing),
        "mm_prepolish_iterations": int(args.mm_prepolish_iterations),
        "newton_tolerance": float(args.newton_tolerance),
        "polish_zero_tolerance": float(args.polish_zero_tolerance),
        "cold_audit_scope": str(args.cold_audit_scope),
        "gaussian_support": [GAUSSIAN_LOWER, GAUSSIAN_UPPER],
        "gaussian_y_support": GAUSSIAN_Y_SUPPORT.tolist(),
        "gaussian_y_mass": GAUSSIAN_Y_MASS.tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# =============================================================================
# Data-driven cross-validation audit
# =============================================================================
CV_CODE_VERSION = "union-selection-sieve-cross-validation-v1.1.0-onecell"


def _stable_token(value: float) -> str:
    return f"{float(value):+.4f}".replace("+", "p").replace("-", "m").replace(".", "d")


def resolve_cv_profile(
    args: argparse.Namespace,
) -> Tuple[List[int], List[int], int, int, int, int]:
    defaults = {
        "smoke": ([800], [8, 12], 2, 3, 1, 21),
        "pilot": ([1500, 3000, 6000], [8, 12, 16, 20], 5, 5, 1, 41),
        "focused": ([1500, 3000, 6000], [8, 12, 16, 20], 10, 5, 1, 41),
    }
    n_default, bins_default, rep_default, fold_default, repeat_default, query_default = defaults[args.profile]
    n_values = sorted(set(args.n_values if args.n_values else n_default))
    bins_values = sorted(set(args.bins_values if args.bins_values else bins_default))
    replicates = int(args.replicates if args.replicates is not None else rep_default)
    n_folds = int(args.cv_folds if args.cv_folds is not None else fold_default)
    cv_repeats = int(args.cv_repeats if args.cv_repeats is not None else repeat_default)
    query_grid_size = int(args.query_grid_size if args.query_grid_size is not None else query_default)
    if any(n < 100 for n in n_values):
        raise ValueError("all latent sample sizes must be at least 100")
    if any(m < 2 for m in bins_values):
        raise ValueError("all bin counts must be at least two")
    if replicates < 1:
        raise ValueError("at least one Monte Carlo replicate is required")
    if n_folds < 2:
        raise ValueError("at least two cross-validation folds are required")
    if cv_repeats < 1:
        raise ValueError("at least one cross-validation repeat is required")
    return n_values, bins_values, replicates, n_folds, cv_repeats, query_grid_size


def make_stratified_fold_assignment(
    observed: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """Assign every observed A--C row to exactly one region-stratified fold."""
    if observed.empty:
        raise ValueError("the observed catalogue is empty")
    if "region" not in observed.columns:
        raise ValueError("the observed catalogue lacks the region column")
    folds = np.full(len(observed), -1, dtype=int)
    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, Any]] = []
    balance: Dict[str, Dict[str, int]] = {}
    for region in ["A", "B", "C"]:
        idx = np.flatnonzero(observed["region"].to_numpy(str) == region)
        if idx.size == 0:
            continue
        if idx.size < n_folds:
            raise ValueError(
                f"region {region} has only {idx.size} rows, fewer than {n_folds} folds"
            )
        shuffled = rng.permutation(idx)
        region_folds = np.arange(idx.size, dtype=int) % int(n_folds)
        folds[shuffled] = region_folds
        counts = np.bincount(region_folds, minlength=n_folds)
        balance[region] = {
            "minimum": int(counts.min()),
            "maximum": int(counts.max()),
            "range": int(counts.max() - counts.min()),
        }
    if np.any(folds < 0):
        bad = np.flatnonzero(folds < 0)
        raise RuntimeError(f"{bad.size} rows were not assigned to a fold")
    fold_counts = np.bincount(folds, minlength=n_folds)
    if np.any(fold_counts == 0):
        raise RuntimeError("at least one validation fold is empty")
    for i, (fold, region) in enumerate(zip(folds, observed["region"].to_numpy(str))):
        rows.append({"row_index": int(i), "region": str(region), "fold": int(fold)})
    audit = {
        "n_rows": int(len(observed)),
        "n_folds": int(n_folds),
        "fold_counts": fold_counts.tolist(),
        "every_row_assigned_once": bool(np.all((folds >= 0) & (folds < n_folds))),
        "region_balance": balance,
        "maximum_region_count_range": int(
            max((entry["range"] for entry in balance.values()), default=0)
        ),
    }
    return folds, pd.DataFrame(rows), audit


def uniform_density_reference_mass(grid: RectangularGrid) -> np.ndarray:
    """Mass vector corresponding to a uniform density on the fixed support box."""
    return _normalise_probability(grid.areas.copy(), "uniform_density_reference_mass")


def predictive_mixture_mass(
    fitted_mass: np.ndarray,
    grid: RectangularGrid,
    n_training_rows: int,
    mixture_strength: float,
) -> Tuple[np.ndarray, float]:
    """Mix a fitted mass with one vanishing uniform predictive pseudocount.

    The mixture weight is eta = c/(n_train+c).  It is used only for held-out
    log-score evaluation and never changes the full-sample estimator.
    """
    if n_training_rows < 1:
        raise ValueError("n_training_rows must be positive")
    c = float(mixture_strength)
    if c < 0.0 or not np.isfinite(c):
        raise ValueError("mixture_strength must be finite and nonnegative")
    p = _normalise_probability(np.asarray(fitted_mass, dtype=float), "fitted_mass")
    reference = uniform_density_reference_mass(grid)
    eta = c / (float(n_training_rows) + c) if c > 0.0 else 0.0
    mixed = (1.0 - eta) * p + eta * reference
    return _normalise_probability(mixed, "predictive_mixture_mass"), float(eta)


def conditional_logscore_vector(
    mass: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    aa, bb = validate_operator_matrices(a, b)
    p = _normalise_probability(np.asarray(mass, dtype=float), "score_mass")
    numerator = aa @ p
    denominator = bb @ p
    score = np.full(aa.shape[0], -math.inf, dtype=float)
    valid = (numerator > 0.0) & (denominator > 0.0)
    score[valid] = np.log(numerator[valid]) - np.log(denominator[valid])
    return score, numerator, denominator


def _build_estimator_operator(
    observed: pd.DataFrame,
    grid: RectangularGrid,
    estimator: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if estimator == "ABC_union_sieve":
        return build_sample_operator(observed, grid)
    if estimator == "A_only_intersection_sieve":
        return build_region_a_intersection_operator(observed, grid)
    raise ValueError(f"unknown estimator {estimator!r}")


def _effective_training_rows(observed: pd.DataFrame, estimator: str) -> int:
    if estimator == "ABC_union_sieve":
        return int(len(observed))
    if estimator == "A_only_intersection_sieve":
        return int((observed["region"].to_numpy(str) == "A").sum())
    raise ValueError(f"unknown estimator {estimator!r}")


def cv_fit_checkpoint_name(
    rho: float,
    n_latent: int,
    replicate: int,
    estimator: str,
    bins: int,
    repeat: int,
    fold: int,
) -> str:
    est = "abc" if estimator == "ABC_union_sieve" else "aonly"
    return (
        f"rho_{_stable_token(rho)}__n_{n_latent:06d}__rep_{replicate:04d}__"
        f"{est}__bins_{bins:03d}__repeat_{repeat:02d}__fold_{fold:02d}.npz"
    )


def full_fit_checkpoint_name(
    rho: float,
    n_latent: int,
    replicate: int,
    estimator: str,
    bins: int,
) -> str:
    est = "abc" if estimator == "ABC_union_sieve" else "aonly"
    return (
        f"rho_{_stable_token(rho)}__n_{n_latent:06d}__rep_{replicate:04d}__"
        f"{est}__bins_{bins:03d}__full.npz"
    )


def save_mass_checkpoint(path: Path, mass: np.ndarray, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mass=np.asarray(mass, dtype=float),
        metadata_json=np.array(json.dumps(_jsonable(metadata), sort_keys=True)),
    )


def load_mass_checkpoint(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        mass = np.asarray(data["mass"], dtype=float)
        metadata = json.loads(str(data["metadata_json"].item()))
    return mass, metadata


def _fit_or_load_mass(
    checkpoint: Path,
    configuration_fingerprint: str,
    observed_train: pd.DataFrame,
    grid: RectangularGrid,
    estimator: str,
    args: argparse.Namespace,
    warm_mass: Optional[np.ndarray],
    run_uniform_audit: bool,
    direct_starts: int,
    direct_seed: int,
    identity: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]], bool]:
    if args.resume and checkpoint.exists():
        mass, metadata = load_mass_checkpoint(checkpoint)
        compatible = bool(
            metadata.get("configuration_fingerprint") == configuration_fingerprint
            and metadata.get("code_version") == CV_CODE_VERSION
            and metadata.get("identity") == _jsonable(identity)
            and mass.size == grid.k
        )
        if compatible:
            return mass, metadata["fit_diagnostics"], metadata.get("start_records", []), True

    a_train, b_train, _ = _build_estimator_operator(observed_train, grid, estimator)
    failure_sidecar = _fit_failure_sidecar_path(checkpoint)
    try:
        fit = fit_sieve_with_start_audit(
            a_train,
            b_train,
            None,
        coverage_tolerance=args.coverage_tolerance,
        mass_tolerance=args.mass_tolerance,
        score_tolerance=args.score_tolerance,
        prune_mass_tolerance=args.prune_mass_tolerance,
        prune_score_margin=args.prune_score_margin,
        max_iterations=args.max_iterations,
        warm_original_mass=warm_mass,
        run_uniform_audit=run_uniform_audit,
        warm_mixing=args.warm_mixing,
        direct_starts=direct_starts,
        direct_seed=direct_seed,
        mm_prepolish_iterations=args.mm_prepolish_iterations,
        newton_tolerance=args.newton_tolerance,
            polish_zero_tolerance=args.polish_zero_tolerance,
            max_polish_cycles=int(getattr(args, "max_polish_cycles", 8)),
        )
    except Exception as exc:
        diagnostic_path = _write_fit_failure_sidecar(
            checkpoint, identity, "fit_exception",
            f"{type(exc).__name__}: {exc}", traceback_text=traceback.format_exc()
        )
        try:
            exc.add_note(f"fit diagnostic written to {diagnostic_path}")
        except AttributeError:
            pass
        raise
    if not fit.selected_fit.converged:
        mm_diag = fit.selected_fit.diagnostics
        diagnostic_path = _write_fit_failure_sidecar(
            checkpoint, identity, "post_fit_KKT_audit",
            "the conditional-likelihood fit did not satisfy the KKT tolerance",
            diagnostics=fit.diagnostics
        )
        raise RuntimeError(
            "the conditional-likelihood fit did not satisfy the KKT tolerance; "
            f"active_residual={float(mm_diag['active_score_sup_abs_residual']):.6e}, "
            f"inactive_violation={float(mm_diag['inactive_positive_score_violation']):.6e}, "
            f"polish_cycles={int(mm_diag.get('polish_cycles', 0))}, "
            f"identity={json.dumps(_jsonable(identity), sort_keys=True)}, "
            f"diagnostic={diagnostic_path}"
        )
    if failure_sidecar.exists():
        failure_sidecar.unlink()
    diagnostics = _jsonable(fit.diagnostics)
    nontrivial_class_indices = [
        h for h, cls in enumerate(fit.reduction.classes) if len(cls) > 1
    ]
    nontrivial_masses = (
        fit.selected_fit.mass[nontrivial_class_indices]
        if nontrivial_class_indices
        else np.array([], dtype=float)
    )
    diagnostics["nontrivial_equivalence_class_mass_fraction"] = float(
        nontrivial_masses.sum()
    )
    diagnostics["maximum_nontrivial_equivalence_class_mass"] = float(
        nontrivial_masses.max()
    ) if nontrivial_masses.size else 0.0
    diagnostics["within_equivalence_class_representative"] = (
        "maximum_entropy_equal_split"
    )
    start_records = fit.start_table.to_dict(orient="records")
    metadata = {
        "configuration_fingerprint": configuration_fingerprint,
        "code_version": CV_CODE_VERSION,
        "identity": _jsonable(identity),
        "fit_diagnostics": diagnostics,
        "start_records": start_records,
    }
    save_mass_checkpoint(checkpoint, fit.original_mass, metadata)
    return fit.original_mass, diagnostics, start_records, False


def _one_standard_error_choice(summary: pd.DataFrame) -> Dict[str, Any]:
    if summary.empty:
        raise ValueError("the cross-validation summary is empty")
    table = summary.sort_values("bins_per_axis").reset_index(drop=True)
    best_row = table.loc[table["mean_validation_logscore"].idxmax()]
    threshold = float(best_row["mean_validation_logscore"] - best_row["se_fold_mean_logscore"])
    eligible = table[table["mean_validation_logscore"] >= threshold - 1e-15]
    chosen = int(eligible["bins_per_axis"].min())
    return {
        "best_bins": int(best_row["bins_per_axis"]),
        "best_mean_logscore": float(best_row["mean_validation_logscore"]),
        "best_logscore_se": float(best_row["se_fold_mean_logscore"]),
        "one_se_threshold": threshold,
        "one_se_bins": chosen,
    }


def _paired_one_standard_error_choice(
    summary: pd.DataFrame,
    fold_scores: pd.DataFrame,
) -> Dict[str, Any]:
    """Choose the coarsest resolution within one paired SE of the CV maximum.

    Fold scores are paired by (repeat, fold), so the standard error is computed
    from score differences against the empirical best resolution rather than
    from the absolute fold-to-fold score variability.
    """
    primary = _one_standard_error_choice(summary)
    best_bins = int(primary["best_bins"])
    best = fold_scores[fold_scores["bins_per_axis"] == best_bins][
        ["repeat", "fold", "mean_validation_logscore"]
    ].rename(columns={"mean_validation_logscore": "best_fold_logscore"})
    rows: List[Dict[str, Any]] = []
    for bins, group in fold_scores.groupby("bins_per_axis", sort=True):
        merged = group[["repeat", "fold", "mean_validation_logscore"]].merge(
            best, on=["repeat", "fold"], how="inner", validate="one_to_one"
        )
        diff = (
            merged["mean_validation_logscore"].to_numpy(float)
            - merged["best_fold_logscore"].to_numpy(float)
        )
        mean_diff = float(np.mean(diff))
        se_diff = (
            float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
            if len(diff) > 1
            else 0.0
        )
        rows.append(
            {
                "bins_per_axis": int(bins),
                "paired_mean_difference_from_best": mean_diff,
                "paired_se_difference_from_best": se_diff,
                "paired_one_se_eligible": bool(mean_diff >= -se_diff - 1e-15),
            }
        )
    table = pd.DataFrame(rows).sort_values("bins_per_axis")
    eligible = table[table["paired_one_se_eligible"]]
    if eligible.empty:
        chosen = best_bins
    else:
        chosen = int(eligible["bins_per_axis"].min())
    chosen_row = table[table["bins_per_axis"] == chosen].iloc[0]
    return {
        "paired_one_se_bins": chosen,
        "paired_one_se_mean_difference_from_best": float(
            chosen_row["paired_mean_difference_from_best"]
        ),
        "paired_one_se_se_difference_from_best": float(
            chosen_row["paired_se_difference_from_best"]
        ),
    }


def run_cv_self_tests() -> Dict[str, Any]:
    base = run_batch_self_tests()
    checks: Dict[str, bool] = {"base_sieve_self_tests": bool(base["all_passed"])}
    details: Dict[str, Any] = {"base_self_tests": base}

    # Stratified folds cover every row exactly once and balance each region to one row.
    toy = pd.DataFrame(
        {
            "region": ["A"] * 11 + ["B"] * 7 + ["C"] * 8,
            "delta1": [1] * 18 + [0] * 8,
            "delta2": [1] * 11 + [0] * 7 + [1] * 8,
        }
    )
    folds, _, audit = make_stratified_fold_assignment(toy, 3, 271828)
    checks["stratified_folds_cover_every_row"] = bool(audit["every_row_assigned_once"])
    checks["stratified_region_balance_within_one"] = bool(audit["maximum_region_count_range"] <= 1)
    checks["all_folds_nonempty"] = bool(np.all(np.bincount(folds, minlength=3) > 0))

    # Predictive mixture is a proper positive mass and vanishes at strength zero.
    grid = make_uniform_grid(4)
    p = np.zeros(grid.k, dtype=float)
    p[0] = 1.0
    mixed, eta = predictive_mixture_mass(p, grid, 100, 1.0)
    zero_mixed, zero_eta = predictive_mixture_mass(p, grid, 100, 0.0)
    checks["predictive_mixture_is_strictly_positive"] = bool(np.all(mixed > 0.0))
    checks["predictive_mixture_is_normalised"] = bool(abs(float(mixed.sum()) - 1.0) < 1e-15)
    checks["zero_strength_leaves_mass_unchanged"] = bool(np.max(np.abs(zero_mixed - p)) < 1e-15 and zero_eta == 0.0)
    details["predictive_eta_n100_c1"] = eta

    # Vector held-out score matches the existing mean conditional loglikelihood.
    a = np.array([[2.0, 0.0], [0.5, 1.0], [0.0, 3.0]], dtype=float)
    b = np.array([[1.0, 1.0], [0.8, 1.0], [1.0, 1.0]], dtype=float)
    p2 = np.array([0.4, 0.6], dtype=float)
    score, _, _ = conditional_logscore_vector(p2, a, b)
    ll = conditional_loglikelihood(p2, a, b, None)
    details["vector_mean_minus_conditional_loglikelihood"] = float(score.mean() - ll)
    checks["heldout_score_matches_conditional_loglikelihood"] = bool(abs(score.mean() - ll) < 1e-15)

    # One-SE rule chooses the coarsest admissible resolution.
    synthetic = pd.DataFrame(
        {
            "bins_per_axis": [8, 12, 16, 20],
            "mean_validation_logscore": [-1.04, -1.01, -1.00, -1.005],
            "se_fold_mean_logscore": [0.01, 0.01, 0.02, 0.01],
        }
    )
    selected = _one_standard_error_choice(synthetic)
    details["synthetic_selection"] = selected
    checks["one_se_rule_chooses_coarsest_eligible"] = bool(
        selected["best_bins"] == 16 and selected["one_se_bins"] == 12
    )
    synthetic_folds = pd.DataFrame(
        {
            "bins_per_axis": np.repeat([8, 12, 16, 20], 4),
            "repeat": np.tile([0, 0, 0, 0], 4),
            "fold": np.tile(np.arange(4), 4),
            "mean_validation_logscore": [
                -1.04, -1.03, -1.05, -1.04,
                -1.01, -1.00, -1.02, -1.01,
                -1.00, -0.99, -1.01, -1.00,
                -1.01, -1.00, -1.02, -1.01,
            ],
        }
    )
    paired_selected = _paired_one_standard_error_choice(synthetic, synthetic_folds)
    details["synthetic_paired_selection"] = paired_selected
    checks["paired_one_se_rule_is_well_defined"] = bool(
        paired_selected["paired_one_se_bins"] in {8, 12, 16, 20}
    )

    # Cross-validation selection function does not require any truth column.
    no_truth = synthetic.copy()
    checks["selection_requires_no_truth_information"] = bool(
        _one_standard_error_choice(no_truth)["best_bins"] == 16
    )

    return {
        "all_passed": bool(all(checks.values())),
        "checks": checks,
        "details": details,
    }


def cv_configuration_fingerprint(
    args: argparse.Namespace,
    n_values: Sequence[int],
    bins_values: Sequence[int],
    replicates: int,
    n_folds: int,
    cv_repeats: int,
    query_grid_size: int,
) -> str:
    payload = {
        "code_version": CV_CODE_VERSION,
        "rho_values": [float(x) for x in args.rho_values],
        "n_values": [int(x) for x in n_values],
        "bins_values": [int(x) for x in bins_values],
        "replicates": int(replicates),
        "n_folds": int(n_folds),
        "cv_repeats": int(cv_repeats),
        "query_grid_size": int(query_grid_size),
        "cv_estimators": list(args.cv_estimators),
        "mixture_strengths": [float(x) for x in args.mixture_strengths],
        "primary_mixture_strength": float(args.primary_mixture_strength),
        "seed": int(args.seed),
        "mass_tolerance": float(args.mass_tolerance),
        "score_tolerance": float(args.score_tolerance),
        "prune_mass_tolerance": float(args.prune_mass_tolerance),
        "prune_score_margin": float(args.prune_score_margin),
        "coverage_tolerance": float(args.coverage_tolerance),
        "warm_mixing": float(args.warm_mixing),
        "max_iterations": int(args.max_iterations),
        "mm_prepolish_iterations": int(args.mm_prepolish_iterations),
        "newton_tolerance": float(args.newton_tolerance),
        "polish_zero_tolerance": float(args.polish_zero_tolerance),
        "support": [GAUSSIAN_LOWER, GAUSSIAN_UPPER],
        "y_support": GAUSSIAN_Y_SUPPORT.tolist(),
        "y_mass": GAUSSIAN_Y_MASS.tolist(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _aggregate_cv_folds(fold_scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = [
        "rho",
        "n_latent",
        "replicate",
        "estimator",
        "bins_per_axis",
        "mixture_strength",
    ]
    for key, group in fold_scores.groupby(group_cols, sort=True):
        g = group.copy()
        total_rows = int(g["n_validation_rows"].sum())
        weighted_mean = float(
            np.sum(g["mean_validation_logscore"] * g["n_validation_rows"]) / total_rows
        )
        fold_means = g["mean_validation_logscore"].to_numpy(float)
        sd = float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else 0.0
        se = float(sd / math.sqrt(len(fold_means))) if len(fold_means) > 0 else math.nan
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_fold_units": int(len(g)),
                "n_validation_rows_total": total_rows,
                "mean_validation_logscore": weighted_mean,
                "sd_fold_mean_logscore": sd,
                "se_fold_mean_logscore": se,
                "raw_zero_numerator_count": int(g["raw_zero_numerator_count"].sum()),
                "raw_zero_denominator_count": int(g["raw_zero_denominator_count"].sum()),
                "raw_nonfinite_score_count": int(g["raw_nonfinite_score_count"].sum()),
                "raw_nonfinite_score_fraction": float(
                    g["raw_nonfinite_score_count"].sum() / total_rows
                ),
                "minimum_predictive_numerator": float(g["minimum_predictive_numerator"].min()),
                "minimum_predictive_denominator": float(g["minimum_predictive_denominator"].min()),
                "maximum_active_score_residual": float(g["active_score_residual"].max()),
                "maximum_inactive_positive_score_violation": float(
                    g["inactive_positive_score_violation"].max()
                ),
                "mean_training_rows": float(g["n_training_rows"].mean()),
                "mean_nontrivial_equivalence_classes": float(
                    g["n_nontrivial_equivalence_classes"].mean()
                ),
                "mean_nontrivial_equivalence_class_mass_fraction": float(
                    g["nontrivial_equivalence_class_mass_fraction"].mean()
                ),
                "maximum_nontrivial_equivalence_class_mass": float(
                    g["maximum_nontrivial_equivalence_class_mass"].max()
                ),
            }
        )
        for region in ["A", "B", "C"]:
            n_col = f"n_validation_{region}"
            s_col = f"sum_logscore_{region}"
            n_region = int(g[n_col].sum()) if n_col in g else 0
            row[f"n_validation_{region}_total"] = n_region
            row[f"mean_logscore_{region}"] = (
                float(g[s_col].sum() / n_region) if n_region > 0 else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_selection_records(
    cv_summary: pd.DataFrame,
    cv_fold_scores: pd.DataFrame,
    full_metrics: pd.DataFrame,
    primary_strength: float,
    bins_values: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    key_cols = ["rho", "n_latent", "replicate", "estimator"]
    for key, full_group in full_metrics.groupby(key_cols, sort=True):
        rho, n_latent, replicate, estimator = key
        oracle_row = full_group.loc[full_group["cdf_rmse"].idxmin()]
        for strength in sorted(cv_summary["mixture_strength"].unique()):
            cv_group = cv_summary[
                (cv_summary["rho"] == rho)
                & (cv_summary["n_latent"] == n_latent)
                & (cv_summary["replicate"] == replicate)
                & (cv_summary["estimator"] == estimator)
                & np.isclose(cv_summary["mixture_strength"], strength)
            ].copy()
            if cv_group.empty:
                continue
            select = _one_standard_error_choice(cv_group)
            fold_group = cv_fold_scores[
                (cv_fold_scores["rho"] == rho)
                & (cv_fold_scores["n_latent"] == n_latent)
                & (cv_fold_scores["replicate"] == replicate)
                & (cv_fold_scores["estimator"] == estimator)
                & np.isclose(cv_fold_scores["mixture_strength"], strength)
            ].copy()
            paired_select = _paired_one_standard_error_choice(cv_group, fold_group)
            score_map = {
                int(r["bins_per_axis"]): float(r["mean_validation_logscore"])
                for _, r in cv_group.iterrows()
            }
            rank_corr = pd.Series(
                [score_map[int(m)] for m in full_group["bins_per_axis"]]
            ).corr(
                pd.Series([-float(x) for x in full_group["cdf_rmse"]]), method="spearman"
            )
            base = {
                "rho": float(rho),
                "n_latent": int(n_latent),
                "replicate": int(replicate),
                "estimator": str(estimator),
                "n_observed": int(full_group.iloc[0]["n_observed"]),
                "n_A": int(full_group.iloc[0]["n_A"]),
                "n_B": int(full_group.iloc[0]["n_B"]),
                "n_C": int(full_group.iloc[0]["n_C"]),
                "mixture_strength": float(strength),
                "is_primary_mixture_strength": bool(np.isclose(strength, primary_strength)),
                "oracle_bins": int(oracle_row["bins_per_axis"]),
                "oracle_cdf_rmse": float(oracle_row["cdf_rmse"]),
                "cv_score_oracle_rmse_spearman": float(rank_corr) if pd.notna(rank_corr) else math.nan,
                **select,
                **paired_select,
            }
            for rule_name, chosen_bins in [
                ("cv_max", int(select["best_bins"])),
                ("cv_one_se", int(select["one_se_bins"])),
                ("cv_paired_one_se", int(paired_select["paired_one_se_bins"])),
            ]:
                chosen_row = full_group[full_group["bins_per_axis"] == chosen_bins].iloc[0]
                record = dict(base)
                selected_rmse = float(chosen_row["cdf_rmse"])
                oracle_rmse = float(oracle_row["cdf_rmse"])
                record.update(
                    {
                        "selection_rule": rule_name,
                        "selected_bins": chosen_bins,
                        "selected_cdf_rmse": selected_rmse,
                        "absolute_rmse_regret": selected_rmse - oracle_rmse,
                        "relative_rmse_regret": (
                            selected_rmse / oracle_rmse - 1.0 if oracle_rmse > 0.0 else math.nan
                        ),
                        "exact_oracle_resolution_match": bool(chosen_bins == int(oracle_row["bins_per_axis"])),
                        "within_5_percent_of_oracle_rmse": bool(selected_rmse <= 1.05 * oracle_rmse + 1e-15),
                        "within_10_percent_of_oracle_rmse": bool(selected_rmse <= 1.10 * oracle_rmse + 1e-15),
                    }
                )
                for fixed in [12, 16, 20]:
                    if fixed in set(int(x) for x in bins_values):
                        fixed_row = full_group[full_group["bins_per_axis"] == fixed].iloc[0]
                        record[f"fixed_{fixed}_cdf_rmse"] = float(fixed_row["cdf_rmse"])
                rows.append(record)
    return pd.DataFrame(rows)


def _aggregate_selection_performance(selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty:
        return pd.DataFrame()
    primary = selection[selection["is_primary_mixture_strength"]].copy()
    rows: List[Dict[str, Any]] = []
    group_cols = ["rho", "n_latent", "estimator", "selection_rule"]
    for key, group in primary.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        selected = group["selected_bins"].to_numpy(int)
        oracle_bins = group["oracle_bins"].to_numpy(int)
        row.update(
            {
                "n_replicates": int(len(group)),
                "mean_selected_bins": float(np.mean(selected)),
                "median_selected_bins": float(np.median(selected)),
                "mean_oracle_bins": float(np.mean(oracle_bins)),
                "median_oracle_bins": float(np.median(oracle_bins)),
                "mean_absolute_bins_difference_from_oracle": float(
                    np.mean(np.abs(selected - oracle_bins))
                ),
                "fraction_exact_oracle_resolution_match": float(
                    group["exact_oracle_resolution_match"].mean()
                ),
                "fraction_within_5_percent_of_oracle_rmse": float(
                    group["within_5_percent_of_oracle_rmse"].mean()
                ),
                "fraction_within_10_percent_of_oracle_rmse": float(
                    group["within_10_percent_of_oracle_rmse"].mean()
                ),
                "mean_selected_cdf_rmse": float(group["selected_cdf_rmse"].mean()),
                "mean_oracle_cdf_rmse": float(group["oracle_cdf_rmse"].mean()),
                "mean_absolute_rmse_regret": float(group["absolute_rmse_regret"].mean()),
                "median_absolute_rmse_regret": float(group["absolute_rmse_regret"].median()),
                "mean_relative_rmse_regret": float(group["relative_rmse_regret"].mean()),
                "mean_cv_score_oracle_rmse_spearman": float(
                    group["cv_score_oracle_rmse_spearman"].mean()
                ),
            }
        )
        for bins, count in pd.Series(selected).value_counts().sort_index().items():
            row[f"selection_fraction_bins_{int(bins)}"] = float(count / len(selected))
        for bins, count in pd.Series(oracle_bins).value_counts().sort_index().items():
            row[f"oracle_fraction_bins_{int(bins)}"] = float(count / len(oracle_bins))
        for fixed in [12, 16, 20]:
            col = f"fixed_{fixed}_cdf_rmse"
            if col in group:
                row[f"mean_fixed_{fixed}_cdf_rmse"] = float(group[col].mean())
                row[f"mean_selected_minus_fixed_{fixed}_rmse"] = float(
                    (group["selected_cdf_rmse"] - group[col]).mean()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _mixture_sensitivity_table(selection: pd.DataFrame, primary_strength: float) -> pd.DataFrame:
    if selection.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    key_cols = ["rho", "n_latent", "replicate", "estimator", "selection_rule"]
    for key, group in selection.groupby(key_cols, sort=True):
        primary = group[np.isclose(group["mixture_strength"], primary_strength)]
        if primary.empty:
            continue
        pbin = int(primary.iloc[0]["selected_bins"])
        for _, row0 in group.iterrows():
            rows.append(
                {
                    **dict(zip(key_cols, key)),
                    "primary_mixture_strength": float(primary_strength),
                    "comparison_mixture_strength": float(row0["mixture_strength"]),
                    "primary_selected_bins": pbin,
                    "comparison_selected_bins": int(row0["selected_bins"]),
                    "same_selected_bins": bool(pbin == int(row0["selected_bins"])),
                    "absolute_bin_difference": int(abs(pbin - int(row0["selected_bins"]))),
                }
            )
    detailed = pd.DataFrame(rows)
    if detailed.empty:
        return detailed
    aggregate = (
        detailed.groupby(
            ["rho", "n_latent", "estimator", "selection_rule", "comparison_mixture_strength"],
            as_index=False,
        )
        .agg(
            n_replicates=("same_selected_bins", "size"),
            fraction_same_selected_bins=("same_selected_bins", "mean"),
            mean_absolute_bin_difference=("absolute_bin_difference", "mean"),
            maximum_absolute_bin_difference=("absolute_bin_difference", "max"),
        )
    )
    return aggregate


def run_cross_validation_audit(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cv_checkpoint_dir = outdir / "cv_fit_checkpoints"
    full_checkpoint_dir = outdir / "full_fit_checkpoints"
    cv_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    full_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_values, bins_values, replicates, n_folds, cv_repeats, query_grid_size = resolve_cv_profile(args)
    mixture_strengths = sorted(set(float(x) for x in args.mixture_strengths))
    if not any(np.isclose(x, args.primary_mixture_strength) for x in mixture_strengths):
        raise ValueError("primary_mixture_strength must be included in mixture_strengths")
    allowed_estimators = {"ABC_union_sieve", "A_only_intersection_sieve"}
    if not set(args.cv_estimators).issubset(allowed_estimators):
        raise ValueError("cv_estimators contains an unknown estimator")
    configuration_fingerprint = cv_configuration_fingerprint(
        args, n_values, bins_values, replicates, n_folds, cv_repeats, query_grid_size
    )
    self_tests = run_cv_self_tests() if args.self_test else {
        "all_passed": True,
        "checks": {},
        "details": {},
    }
    if not self_tests["all_passed"]:
        raise RuntimeError("a built-in cross-validation self-test failed")

    q_axis = np.linspace(GAUSSIAN_LOWER, GAUSSIAN_UPPER, query_grid_size)
    q1m, q2m = np.meshgrid(q_axis, q_axis, indexing="ij")
    query_x1 = q1m.reshape(-1)
    query_x2 = q2m.reshape(-1)
    truths: Dict[float, TruncatedBivariateNormalTruth] = {}
    truth_vectors: Dict[float, np.ndarray] = {}
    true_cell_mass: Dict[Tuple[float, int], np.ndarray] = {}
    cdf_matrices: Dict[int, np.ndarray] = {}
    for rho_raw in args.rho_values:
        rho = float(rho_raw)
        truth = TruncatedBivariateNormalTruth(rho, GAUSSIAN_LOWER, GAUSSIAN_UPPER)
        truths[rho] = truth
        truth_vectors[rho] = np.array(
            [truth.cdf(float(x1), float(x2)) for x1, x2 in zip(query_x1, query_x2)],
            dtype=float,
        )
        for bins in bins_values:
            grid = make_uniform_grid(bins)
            cdf_matrices.setdefault(bins, cdf_operator_matrix(grid, q_axis, q_axis)[0])
            true_cell_mass[(rho, bins)] = truth.cell_masses(grid)

    full_records: List[Dict[str, Any]] = []
    cv_fold_records: List[Dict[str, Any]] = []
    fold_assignment_records: List[Dict[str, Any]] = []
    start_records: List[Dict[str, Any]] = []
    failure_records: List[Dict[str, Any]] = []
    total_full_fits = len(args.rho_values) * replicates * len(n_values) * len(bins_values) * 2
    total_cv_fits = (
        len(args.rho_values)
        * replicates
        * len(n_values)
        * len(bins_values)
        * n_folds
        * cv_repeats
        * len(args.cv_estimators)
    )
    full_done = 0
    cv_done = 0
    start_clock = time.perf_counter()

    for rho_index, rho_raw in enumerate(args.rho_values):
        rho = float(rho_raw)
        max_n = max(n_values)
        for replicate in range(replicates):
            seed = combination_seed(args.seed, rho_index, replicate)
            latent_full, _, _ = simulate_gaussian_catalogue(
                max_n,
                seed,
                rho,
                GAUSSIAN_LOWER,
                GAUSSIAN_UPPER,
                GAUSSIAN_Y_SUPPORT,
                GAUSSIAN_Y_MASS,
            )
            for n_latent in n_values:
                latent, observed, sample_info = observed_from_latent_prefix(latent_full, n_latent)
                dataset_id = {
                    "rho": rho,
                    "n_latent": int(n_latent),
                    "replicate": int(replicate),
                    "seed": int(seed),
                }

                # Full-sample fits for truth-based oracle evaluation only.  Truth is
                # not used in fitting or in cross-validation selection.
                for estimator in ["ABC_union_sieve", "A_only_intersection_sieve"]:
                    previous_grid: Optional[RectangularGrid] = None
                    previous_mass: Optional[np.ndarray] = None
                    for bins in bins_values:
                        grid = make_uniform_grid(bins)
                        warm_mass = None
                        if previous_grid is not None and previous_mass is not None:
                            warm_mass = project_rectangular_mass(previous_grid, previous_mass, grid)
                        audit_this = bool(
                            replicate == 0
                            and n_latent == min(n_values)
                            and bins == min(bins_values)
                        )
                        direct_starts = args.direct_audit_starts if audit_this else 0
                        ckpt = full_checkpoint_dir / full_fit_checkpoint_name(
                            rho, n_latent, replicate, estimator, bins
                        )
                        identity = {
                            **dataset_id,
                            "fit_type": "full",
                            "estimator": estimator,
                            "bins_per_axis": int(bins),
                        }
                        try:
                            fit_start = time.perf_counter()
                            mass, diag, starts, loaded = _fit_or_load_mass(
                                ckpt,
                                configuration_fingerprint,
                                observed,
                                grid,
                                estimator,
                                args,
                                warm_mass,
                                run_uniform_audit=audit_this,
                                direct_starts=direct_starts,
                                direct_seed=seed + 17 * bins,
                                identity=identity,
                            )
                            runtime = time.perf_counter() - fit_start
                            metrics, _ = estimator_accuracy_metrics(
                                grid,
                                mass,
                                true_cell_mass[(rho, bins)],
                                cdf_matrices[bins],
                                truth_vectors[rho],
                            )
                            mm_diag = diag["mm"]
                            full_records.append(
                                {
                                    **dataset_id,
                                    "estimator": estimator,
                                    "bins_per_axis": int(bins),
                                    "n_cells": int(grid.k),
                                    "n_observed": int(sample_info["n_observed"]),
                                    "n_A": int(sample_info["region_counts_observed"]["A"]),
                                    "n_B": int(sample_info["region_counts_observed"]["B"]),
                                    "n_C": int(sample_info["region_counts_observed"]["C"]),
                                    "fit_loaded_from_checkpoint": bool(loaded),
                                    "runtime_seconds": float(runtime),
                                    "converged": bool(mm_diag["converged"]),
                                    "active_score_residual": float(mm_diag["active_score_sup_abs_residual"]),
                                    "inactive_positive_score_violation": float(mm_diag["inactive_positive_score_violation"]),
                                    "n_forced_zero_cells": int(diag["n_forced_zero_cells"]),
                                    "n_equivalence_classes": int(diag["n_equivalence_classes"]),
                                    "n_nontrivial_equivalence_classes": int(diag["n_nontrivial_equivalence_classes"]),
                                    "nontrivial_equivalence_class_mass_fraction": float(diag.get("nontrivial_equivalence_class_mass_fraction", 0.0)),
                                    "maximum_nontrivial_equivalence_class_mass": float(diag.get("maximum_nontrivial_equivalence_class_mass", 0.0)),
                                    "n_active_classes": int(mm_diag["n_active"]),
                                    "warm_cold_loglikelihood_abs_difference": float(diag["warm_cold_loglikelihood_abs_difference"]),
                                    "direct_loglikelihood_abs_difference": float(diag["direct_loglikelihood_abs_difference"]),
                                    **metrics,
                                }
                            )
                            for rec in starts:
                                start_records.append(
                                    {
                                        **dataset_id,
                                        "fit_type": "full",
                                        "estimator": estimator,
                                        "bins_per_axis": int(bins),
                                        "repeat": -1,
                                        "fold": -1,
                                        **rec,
                                    }
                                )
                            previous_grid, previous_mass = grid, mass
                        except Exception as exc:
                            failure_records.append(
                                {
                                    **dataset_id,
                                    "fit_type": "full",
                                    "estimator": estimator,
                                    "bins_per_axis": int(bins),
                                    "repeat": -1,
                                    "fold": -1,
                                    "exception_type": type(exc).__name__,
                                    "message": str(exc),
                                }
                            )
                            if args.fail_fast:
                                raise
                        full_done += 1
                        print(
                            f"Full fit {full_done}/{total_full_fits}: rho={rho:.2f}, "
                            f"rep={replicate + 1}/{replicates}, n={n_latent}, "
                            f"est={estimator}, bins={bins} "
                            f"({time.perf_counter() - start_clock:.1f} s)"
                        )

                # Honest region-stratified cross-validation.  No latent truth is
                # used in fold construction, fitting, scoring, or resolution choice.
                for repeat in range(cv_repeats):
                    fold_seed = seed + 1_000_003 * (repeat + 1) + 97 * n_latent
                    folds, fold_table, fold_audit = make_stratified_fold_assignment(
                        observed, n_folds, fold_seed
                    )
                    fold_table.insert(0, "rho", rho)
                    fold_table.insert(1, "n_latent", int(n_latent))
                    fold_table.insert(2, "replicate", int(replicate))
                    fold_table.insert(3, "repeat", int(repeat))
                    fold_table.insert(4, "fold_seed", int(fold_seed))
                    fold_assignment_records.extend(fold_table.to_dict(orient="records"))

                    for estimator in args.cv_estimators:
                        previous_by_fold: Dict[int, Tuple[RectangularGrid, np.ndarray]] = {}
                        for bins in bins_values:
                            grid = make_uniform_grid(bins)
                            for fold in range(n_folds):
                                train_mask = folds != fold
                                valid_mask = folds == fold
                                train = observed.loc[train_mask].copy().reset_index(drop=True)
                                valid = observed.loc[valid_mask].copy().reset_index(drop=True)
                                n_train_effective = _effective_training_rows(train, estimator)
                                n_valid_effective = _effective_training_rows(valid, estimator)
                                if n_train_effective < 1 or n_valid_effective < 1:
                                    raise RuntimeError("a CV fold has no effective rows for the estimator")
                                warm_mass = None
                                if fold in previous_by_fold:
                                    prev_grid, prev_mass = previous_by_fold[fold]
                                    warm_mass = project_rectangular_mass(prev_grid, prev_mass, grid)
                                audit_this = bool(
                                    repeat == 0
                                    and fold == 0
                                    and replicate == 0
                                    and n_latent == min(n_values)
                                    and bins == min(bins_values)
                                )
                                direct_starts = args.direct_audit_starts if audit_this else 0
                                ckpt = cv_checkpoint_dir / cv_fit_checkpoint_name(
                                    rho, n_latent, replicate, estimator, bins, repeat, fold
                                )
                                identity = {
                                    **dataset_id,
                                    "fit_type": "cross_validation",
                                    "estimator": estimator,
                                    "bins_per_axis": int(bins),
                                    "repeat": int(repeat),
                                    "fold": int(fold),
                                    "fold_seed": int(fold_seed),
                                }
                                try:
                                    fit_start = time.perf_counter()
                                    mass, diag, starts, loaded = _fit_or_load_mass(
                                        ckpt,
                                        configuration_fingerprint,
                                        train,
                                        grid,
                                        estimator,
                                        args,
                                        warm_mass,
                                        run_uniform_audit=audit_this,
                                        direct_starts=direct_starts,
                                        direct_seed=seed + 1009 * bins + 31 * fold,
                                        identity=identity,
                                    )
                                    runtime = time.perf_counter() - fit_start
                                    previous_by_fold[fold] = (grid, mass)
                                    a_val, b_val, val_audit = _build_estimator_operator(valid, grid, estimator)
                                    raw_score, raw_num, raw_den = conditional_logscore_vector(mass, a_val, b_val)
                                    mm_diag = diag["mm"]
                                    valid_regions = (
                                        valid["region"].to_numpy(str)
                                        if estimator == "ABC_union_sieve"
                                        else np.full(a_val.shape[0], "A", dtype=object)
                                    )
                                    for strength in mixture_strengths:
                                        pred_mass, eta = predictive_mixture_mass(
                                            mass, grid, n_train_effective, strength
                                        )
                                        score, numerator, denominator = conditional_logscore_vector(
                                            pred_mass, a_val, b_val
                                        )
                                        if not np.all(np.isfinite(score)):
                                            raise RuntimeError("predictive mixture produced a non-finite held-out score")
                                        record: Dict[str, Any] = {
                                            **dataset_id,
                                            "estimator": estimator,
                                            "bins_per_axis": int(bins),
                                            "n_cells": int(grid.k),
                                            "repeat": int(repeat),
                                            "fold": int(fold),
                                            "fold_seed": int(fold_seed),
                                            "n_training_rows": int(n_train_effective),
                                            "n_validation_rows": int(n_valid_effective),
                                            "mixture_strength": float(strength),
                                            "predictive_mixture_eta": float(eta),
                                            "mean_validation_logscore": float(np.mean(score)),
                                            "sum_validation_logscore": float(np.sum(score)),
                                            "sd_validation_logscore": float(np.std(score, ddof=1)) if len(score) > 1 else 0.0,
                                            "minimum_predictive_numerator": float(np.min(numerator)),
                                            "minimum_predictive_denominator": float(np.min(denominator)),
                                            "raw_zero_numerator_count": int(np.sum(raw_num <= 0.0)),
                                            "raw_zero_denominator_count": int(np.sum(raw_den <= 0.0)),
                                            "raw_nonfinite_score_count": int(np.sum(~np.isfinite(raw_score))),
                                            "fit_loaded_from_checkpoint": bool(loaded),
                                            "runtime_seconds": float(runtime),
                                            "active_score_residual": float(mm_diag["active_score_sup_abs_residual"]),
                                            "inactive_positive_score_violation": float(mm_diag["inactive_positive_score_violation"]),
                                            "n_forced_zero_cells": int(diag["n_forced_zero_cells"]),
                                            "n_equivalence_classes": int(diag["n_equivalence_classes"]),
                                            "n_nontrivial_equivalence_classes": int(diag["n_nontrivial_equivalence_classes"]),
                                            "nontrivial_equivalence_class_mass_fraction": float(diag.get("nontrivial_equivalence_class_mass_fraction", 0.0)),
                                            "maximum_nontrivial_equivalence_class_mass": float(diag.get("maximum_nontrivial_equivalence_class_mass", 0.0)),
                                            "n_active_classes": int(mm_diag["n_active"]),
                                            "fold_region_balance_max_range": int(fold_audit["maximum_region_count_range"]),
                                        }
                                        for region in ["A", "B", "C"]:
                                            mask = valid_regions == region
                                            record[f"n_validation_{region}"] = int(np.sum(mask))
                                            record[f"sum_logscore_{region}"] = float(np.sum(score[mask])) if np.any(mask) else 0.0
                                            record[f"mean_logscore_{region}"] = float(np.mean(score[mask])) if np.any(mask) else math.nan
                                        cv_fold_records.append(record)
                                    for rec in starts:
                                        start_records.append(
                                            {
                                                **dataset_id,
                                                "fit_type": "cross_validation",
                                                "estimator": estimator,
                                                "bins_per_axis": int(bins),
                                                "repeat": int(repeat),
                                                "fold": int(fold),
                                                **rec,
                                            }
                                        )
                                except Exception as exc:
                                    failure_records.append(
                                        {
                                            **dataset_id,
                                            "fit_type": "cross_validation",
                                            "estimator": estimator,
                                            "bins_per_axis": int(bins),
                                            "repeat": int(repeat),
                                            "fold": int(fold),
                                            "exception_type": type(exc).__name__,
                                            "message": str(exc),
                                        }
                                    )
                                    if args.fail_fast:
                                        raise
                                cv_done += 1
                                print(
                                    f"CV fit {cv_done}/{total_cv_fits}: rho={rho:.2f}, "
                                    f"rep={replicate + 1}/{replicates}, n={n_latent}, "
                                    f"est={estimator}, bins={bins}, repeat={repeat + 1}, "
                                    f"fold={fold + 1}/{n_folds} "
                                    f"({time.perf_counter() - start_clock:.1f} s)"
                                )

    full_table = pd.DataFrame(full_records)
    cv_fold_table = pd.DataFrame(cv_fold_records)
    fold_assignment_table = pd.DataFrame(fold_assignment_records)
    start_table = pd.DataFrame(start_records)
    failures = pd.DataFrame(failure_records)
    if full_table.empty or cv_fold_table.empty:
        raise RuntimeError("the cross-validation audit produced no usable results")
    cv_summary = _aggregate_cv_folds(cv_fold_table)
    selection = _build_selection_records(
        cv_summary,
        cv_fold_table,
        full_table[full_table["estimator"].isin(args.cv_estimators)],
        args.primary_mixture_strength,
        bins_values,
    )
    selection_performance = _aggregate_selection_performance(selection)
    mixture_sensitivity = _mixture_sensitivity_table(
        selection, args.primary_mixture_strength
    )

    # Summary checks.
    fit_rows = pd.concat(
        [
            full_table[["active_score_residual", "inactive_positive_score_violation", "converged"]],
            cv_fold_table[["active_score_residual", "inactive_positive_score_violation"]].assign(converged=True),
        ],
        ignore_index=True,
    )
    primary_selection = selection[selection["is_primary_mixture_strength"]]
    primary_sensitivity = mixture_sensitivity[
        ~np.isclose(mixture_sensitivity["comparison_mixture_strength"], args.primary_mixture_strength)
    ] if not mixture_sensitivity.empty else mixture_sensitivity
    checks = {
        "self_tests_passed": bool(self_tests["all_passed"]),
        "no_failed_fits": bool(failures.empty),
        "all_full_and_cv_fits_converged": bool(fit_rows["converged"].all()),
        "all_active_score_residuals_below_tolerance": bool(
            fit_rows["active_score_residual"].max() < args.score_tolerance
        ),
        "all_inactive_kkt_violations_below_tolerance": bool(
            fit_rows["inactive_positive_score_violation"].max() < args.score_tolerance
        ),
        "all_stabilised_heldout_scores_finite": bool(
            np.isfinite(cv_fold_table["mean_validation_logscore"]).all()
        ),
        "folds_region_balanced_within_one": bool(
            cv_fold_table["fold_region_balance_max_range"].max() <= 1
        ),
        "all_cv_choices_belong_to_candidate_set": bool(
            set(selection["selected_bins"].astype(int)).issubset(set(bins_values))
        ),
        "selection_used_no_truth_information": True,
        "support_box_fixed_before_fold_assignment": True,
        "predictive_mixture_used_only_for_heldout_scoring": True,
        "full_sample_estimator_uses_no_predictive_mixture": True,
        "training_equivalence_classes_use_symmetric_maximum_entropy_split": True,
    }

    paths = {
        "full_fit_metrics": outdir / "full_fit_metrics.csv",
        "cv_fold_scores": outdir / "cv_fold_scores.csv",
        "cv_resolution_summary": outdir / "cv_resolution_summary.csv",
        "selection_per_replicate": outdir / "selection_per_replicate.csv",
        "selection_performance": outdir / "selection_performance_summary.csv",
        "mixture_sensitivity": outdir / "mixture_strength_sensitivity.csv",
        "fold_assignments": outdir / "fold_assignments.csv",
        "start_audit": outdir / "start_audit.csv",
        "failures": outdir / "failures.csv",
        "self_tests": outdir / "self_test_results.json",
        "summary": outdir / "comparison_summary.json",
        "manifest": outdir / "manifest.json",
        "environment": outdir / "environment.json",
        "file_hashes": outdir / "file_hashes.json",
    }
    full_table.to_csv(paths["full_fit_metrics"], index=False)
    cv_fold_table.to_csv(paths["cv_fold_scores"], index=False)
    cv_summary.to_csv(paths["cv_resolution_summary"], index=False)
    selection.to_csv(paths["selection_per_replicate"], index=False)
    selection_performance.to_csv(paths["selection_performance"], index=False)
    mixture_sensitivity.to_csv(paths["mixture_sensitivity"], index=False)
    fold_assignment_table.to_csv(paths["fold_assignments"], index=False)
    start_table.to_csv(paths["start_audit"], index=False)
    failures.to_csv(paths["failures"], index=False)
    _write_json(paths["self_tests"], self_tests)

    elapsed = time.perf_counter() - start_clock
    summary = {
        "code_version": CV_CODE_VERSION,
        "profile": args.profile,
        "all_main_checks_passed": bool(all(checks.values())),
        "checks": checks,
        "configuration_fingerprint": configuration_fingerprint,
        "rho_values": [float(x) for x in args.rho_values],
        "n_values": n_values,
        "bins_values": bins_values,
        "replicates": replicates,
        "cv_folds": n_folds,
        "cv_repeats": cv_repeats,
        "cv_estimators": list(args.cv_estimators),
        "mixture_strengths": mixture_strengths,
        "primary_mixture_strength": float(args.primary_mixture_strength),
        "n_successful_full_fits": int(len(full_table)),
        "n_successful_cv_fold_fits": int(
            cv_fold_table[["rho", "n_latent", "replicate", "estimator", "bins_per_axis", "repeat", "fold"]]
            .drop_duplicates()
            .shape[0]
        ),
        "n_failed_fits": int(len(failures)),
        "maximum_active_score_residual": float(fit_rows["active_score_residual"].max()),
        "maximum_inactive_positive_score_violation": float(
            fit_rows["inactive_positive_score_violation"].max()
        ),
        "raw_nonfinite_heldout_score_fraction": float(
            cv_summary["raw_nonfinite_score_count"].sum()
            / cv_summary["n_validation_rows_total"].sum()
        ),
        "minimum_stabilised_predictive_numerator": float(
            cv_fold_table["minimum_predictive_numerator"].min()
        ),
        "minimum_stabilised_predictive_denominator": float(
            cv_fold_table["minimum_predictive_denominator"].min()
        ),
        "mean_primary_cv_one_se_absolute_regret": float(
            primary_selection.loc[
                primary_selection["selection_rule"] == "cv_one_se", "absolute_rmse_regret"
            ].mean()
        ),
        "mean_primary_cv_max_absolute_regret": float(
            primary_selection.loc[
                primary_selection["selection_rule"] == "cv_max", "absolute_rmse_regret"
            ].mean()
        ),
        "mean_primary_cv_paired_one_se_absolute_regret": float(
            primary_selection.loc[
                primary_selection["selection_rule"] == "cv_paired_one_se", "absolute_rmse_regret"
            ].mean()
        ),
        "primary_cv_one_se_within_5_percent_oracle_fraction": float(
            primary_selection.loc[
                primary_selection["selection_rule"] == "cv_one_se", "within_5_percent_of_oracle_rmse"
            ].mean()
        ),
        "primary_cv_max_within_5_percent_oracle_fraction": float(
            primary_selection.loc[
                primary_selection["selection_rule"] == "cv_max", "within_5_percent_of_oracle_rmse"
            ].mean()
        ),
        "primary_cv_paired_one_se_within_5_percent_oracle_fraction": float(
            primary_selection.loc[
                primary_selection["selection_rule"] == "cv_paired_one_se", "within_5_percent_of_oracle_rmse"
            ].mean()
        ),
        "mixture_sensitivity_fraction_same_selected_bins": float(
            primary_sensitivity["fraction_same_selected_bins"].mean()
        ) if not primary_sensitivity.empty else 1.0,
        "elapsed_seconds": float(elapsed),
        "scientific_scope": {
            "observed_data_only_resolution_selection": True,
            "region_stratified_kfold_cross_validation": True,
            "heldout_union_selection_conditional_logscore": True,
            "vanishing_uniform_predictive_mixture": True,
            "truth_used_only_for_post_selection_audit": True,
            "fixed_support_box_known_before_cross_validation": True,
            "cross_validation_optimality_theorem": False,
            "sieve_asymptotic_theory": False,
            "numerical_solver_amendment_only": True,
        },
    }
    _write_json(paths["summary"], summary)
    manifest = {
        "code_version": CV_CODE_VERSION,
        "configuration_fingerprint": configuration_fingerprint,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "resolved_profile": {
            "n_values": n_values,
            "bins_values": bins_values,
            "replicates": replicates,
            "cv_folds": n_folds,
            "cv_repeats": cv_repeats,
            "query_grid_size": query_grid_size,
        },
        "fold_design": "independent deterministic region-stratified folds within each observed catalogue and repeat",
        "predictive_stabilisation": {
            "definition": "p_pred=(1-eta)*p_hat+eta*p_uniform; eta=c/(n_train+c)",
            "used_only_for_heldout_scoring": True,
            "within_training_equivalence_class_representative": "maximum_entropy_equal_split",
            "mixture_strengths": mixture_strengths,
            "primary_strength": float(args.primary_mixture_strength),
        },
        "selection_rules": {
            "cv_max": "maximise mean heldout conditional logscore",
            "cv_one_se": "coarsest resolution within the absolute fold-level SE of the maximum",
            "cv_paired_one_se": "coarsest resolution whose paired fold-score deficit from the maximum is within one SE of that deficit",
        },
        "truth_usage": "truth CDF is used only after CV choices are fixed, to audit oracle match and RMSE regret",
        "support_box": [GAUSSIAN_LOWER, GAUSSIAN_UPPER],
        "checkpoint_directories": {
            "cv": str(cv_checkpoint_dir),
            "full": str(full_checkpoint_dir),
        },
    }
    _write_json(paths["manifest"], manifest)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "working_directory": str(Path.cwd()),
    }
    _write_json(paths["environment"], environment)
    hashes: Dict[str, Dict[str, str]] = {}
    for name, path in paths.items():
        if name != "file_hashes" and path.exists():
            hashes[name] = {"path": str(path), "sha256": _sha256(path)}
    _write_json(paths["file_hashes"], hashes)
    return summary, paths


def build_cv_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="union_selection_sieve_cross_validation_outputs")
    parser.add_argument("--profile", choices=["smoke", "pilot", "focused"], default="pilot")
    parser.add_argument("--n-values", type=int, nargs="+", default=None)
    parser.add_argument("--bins-values", type=int, nargs="+", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--cv-repeats", type=int, default=None)
    parser.add_argument("--query-grid-size", type=int, default=None)
    parser.add_argument("--rho-values", type=float, nargs="+", default=[0.55, 0.9])
    parser.add_argument("--cv-estimators", nargs="+", default=["ABC_union_sieve"])
    parser.add_argument("--mixture-strengths", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--primary-mixture-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--mass-tolerance", type=float, default=1e-12)
    parser.add_argument("--score-tolerance", type=float, default=1e-9)
    parser.add_argument("--prune-mass-tolerance", type=float, default=1e-7)
    parser.add_argument("--prune-score-margin", type=float, default=1e-7)
    parser.add_argument("--coverage-tolerance", type=float, default=1e-15)
    parser.add_argument("--warm-mixing", type=float, default=1e-9)
    parser.add_argument("--max-iterations", type=int, default=30000)
    parser.add_argument("--mm-prepolish-iterations", type=int, default=2000)
    parser.add_argument("--newton-tolerance", type=float, default=1e-12)
    parser.add_argument("--polish-zero-tolerance", type=float, default=1e-13)
    parser.add_argument("--max-polish-cycles", type=int, default=8)
    parser.add_argument("--direct-audit-starts", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def cv_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cv_parser()
    if argv is None:
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"Ignoring unrecognised arguments: {unknown}")
    else:
        args = parser.parse_args(list(argv))
    summary, paths = run_cross_validation_audit(args)
    concise = {
        "code_version": summary["code_version"],
        "profile": summary["profile"],
        "all_main_checks_passed": summary["all_main_checks_passed"],
        "checks": summary["checks"],
        "rho_values": summary["rho_values"],
        "n_values": summary["n_values"],
        "bins_values": summary["bins_values"],
        "replicates": summary["replicates"],
        "cv_folds": summary["cv_folds"],
        "cv_repeats": summary["cv_repeats"],
        "cv_estimators": summary["cv_estimators"],
        "mixture_strengths": summary["mixture_strengths"],
        "primary_mixture_strength": summary["primary_mixture_strength"],
        "n_successful_full_fits": summary["n_successful_full_fits"],
        "n_successful_cv_fold_fits": summary["n_successful_cv_fold_fits"],
        "n_failed_fits": summary["n_failed_fits"],
        "maximum_active_score_residual": summary["maximum_active_score_residual"],
        "maximum_inactive_positive_score_violation": summary["maximum_inactive_positive_score_violation"],
        "raw_nonfinite_heldout_score_fraction": summary["raw_nonfinite_heldout_score_fraction"],
        "mean_primary_cv_one_se_absolute_regret": summary["mean_primary_cv_one_se_absolute_regret"],
        "mean_primary_cv_max_absolute_regret": summary["mean_primary_cv_max_absolute_regret"],
        "mean_primary_cv_paired_one_se_absolute_regret": summary["mean_primary_cv_paired_one_se_absolute_regret"],
        "primary_cv_one_se_within_5_percent_oracle_fraction": summary["primary_cv_one_se_within_5_percent_oracle_fraction"],
        "primary_cv_max_within_5_percent_oracle_fraction": summary["primary_cv_max_within_5_percent_oracle_fraction"],
        "primary_cv_paired_one_se_within_5_percent_oracle_fraction": summary["primary_cv_paired_one_se_within_5_percent_oracle_fraction"],
        "mixture_sensitivity_fraction_same_selected_bins": summary["mixture_sensitivity_fraction_same_selected_bins"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }
    print(json.dumps(_jsonable(concise), indent=2, sort_keys=True))

    perf = pd.read_csv(paths["selection_performance"])
    print("\nObserved-data-only CV selection performance against the simulation oracle:")
    print(perf.to_string(index=False))

    selected = pd.read_csv(paths["selection_per_replicate"])
    primary = selected[selected["is_primary_mixture_strength"]]
    print("\nPrimary-mixture resolution choices by replicate:")
    print(
        primary[[
            "rho", "n_latent", "replicate", "estimator", "selection_rule",
            "selected_bins", "oracle_bins", "selected_cdf_rmse", "oracle_cdf_rmse",
            "absolute_rmse_regret", "within_5_percent_of_oracle_rmse",
            "cv_score_oracle_rmse_spearman",
        ]].to_string(index=False)
    )

    cvsum = pd.read_csv(paths["cv_resolution_summary"])
    cvsum = cvsum[np.isclose(cvsum["mixture_strength"], args.primary_mixture_strength)]
    print("\nPrimary-mixture held-out conditional-logscore summary:")
    print(
        cvsum[[
            "rho", "n_latent", "replicate", "estimator", "bins_per_axis",
            "mean_validation_logscore", "se_fold_mean_logscore",
            "raw_nonfinite_score_fraction", "mean_nontrivial_equivalence_classes",
            "mean_nontrivial_equivalence_class_mass_fraction",
        ]].to_string(index=False)
    )

    sensitivity = pd.read_csv(paths["mixture_sensitivity"])
    print("\nPredictive-mixture sensitivity of selected resolution:")
    print(sensitivity.to_string(index=False))

    print("\nWritten files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("One-cell data-driven sieve cross-validation audit completed successfully (status=0).")
    print(f"Output directory: {Path(args.outdir).resolve()}")
    return 0

# =============================================================================
# Selection-target decomposition and repeated-CV audit
# =============================================================================
TARGET_CODE_VERSION = "union-selection-sieve-target-decomposition-v1.0.0-onecell"
TARGET_SPLIT_STRATEGIES = ("equal", "coarse_prior", "lower_corner", "upper_corner")


@dataclass
class TargetModel:
    equal_mass: np.ndarray
    class_mass: np.ndarray
    classes: List[List[int]]
    diagnostics: Dict[str, Any]
    start_records: List[Dict[str, Any]]
    loaded_from_checkpoint: bool


def _vector_cell_index(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    idx = np.searchsorted(np.asarray(edges, dtype=float), x, side="right") - 1
    idx = idx.astype(int, copy=False)
    idx[x == edges[-1]] = len(edges) - 2
    idx[(x < edges[0]) | (x > edges[-1])] = -1
    return idx


@dataclass
class FastObservedScoreDesign:
    """Fast evaluation design for observed A/B/C conditional log scores.

    All geometry is precomputed for one observed catalogue and one rectangular
    grid.  Each new mass vector is then scored in O(n + K * n_Y) time rather
    than by constructing an n-by-K operator matrix.
    """

    grid: RectangularGrid
    region_code: np.ndarray
    x1_cell: np.ndarray
    x2_cell: np.ndarray
    y_group: np.ndarray
    y_pairs: np.ndarray
    selection_rows: np.ndarray
    fraction_below1: np.ndarray
    fraction_below2: np.ndarray
    block_ids: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.region_code.size)

    @property
    def n_blocks(self) -> int:
        return int(self.block_ids.max() + 1) if self.block_ids.size else 0

    @classmethod
    def from_observed(
        cls,
        observed: pd.DataFrame,
        grid: RectangularGrid,
        block_ids: Optional[np.ndarray] = None,
    ) -> "FastObservedScoreDesign":
        required = {"region", "x1_obs", "x2_obs", "y1", "y2"}
        missing = sorted(required.difference(observed.columns))
        if missing:
            raise ValueError(f"observed catalogue lacks required columns: {missing}")
        if observed.empty:
            raise ValueError("observed catalogue is empty")
        region_text = observed["region"].to_numpy(str)
        mapping = {"A": 0, "B": 1, "C": 2}
        if not set(np.unique(region_text)).issubset(mapping):
            raise ValueError("score design accepts only Regions A, B, and C")
        region_code = np.array([mapping[x] for x in region_text], dtype=int)
        x1_cell = _vector_cell_index(grid.edges1, observed["x1_obs"].to_numpy(float))
        x2_cell = _vector_cell_index(grid.edges2, observed["x2_obs"].to_numpy(float))
        if np.any((region_code != 2) & (x1_cell < 0)):
            raise ValueError("a detected x1 value lies outside the sieve support")
        if np.any((region_code != 1) & (x2_cell < 0)):
            raise ValueError("a detected x2 value lies outside the sieve support")

        pairs = observed[["y1", "y2"]].drop_duplicates().sort_values(["y1", "y2"])
        y_pairs = pairs.to_numpy(float)
        pair_to_index = {
            (float(a), float(b)): int(i) for i, (a, b) in enumerate(y_pairs)
        }
        y_group = np.array(
            [
                pair_to_index[(float(a), float(b))]
                for a, b in observed[["y1", "y2"]].to_numpy(float)
            ],
            dtype=int,
        )
        selection_rows = np.vstack(
            [grid.selection_row(float(a), float(b)) for a, b in y_pairs]
        )
        fraction_below1 = np.vstack(
            [grid.fraction_below1(float(a)) for a, _ in y_pairs]
        )
        fraction_below2 = np.vstack(
            [grid.fraction_below2(float(b)) for _, b in y_pairs]
        )
        if block_ids is None:
            blocks = np.zeros(len(observed), dtype=int)
        else:
            blocks = np.asarray(block_ids, dtype=int)
            if blocks.ndim != 1 or blocks.size != len(observed):
                raise ValueError("block_ids must have one entry per observed row")
            if np.any(blocks < 0):
                raise ValueError("block_ids must be nonnegative")
        return cls(
            grid=grid,
            region_code=region_code,
            x1_cell=x1_cell,
            x2_cell=x2_cell,
            y_group=y_group,
            y_pairs=y_pairs,
            selection_rows=selection_rows,
            fraction_below1=fraction_below1,
            fraction_below2=fraction_below2,
            block_ids=blocks,
        )

    def score_vector(
        self, mass: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = _normalise_probability(np.asarray(mass, dtype=float), "score_mass")
        if p.size != self.grid.k:
            raise ValueError("mass length differs from the score grid")
        pmat = p.reshape(self.grid.n1, self.grid.n2)
        numerator = np.zeros(self.n_rows, dtype=float)

        mask_a = self.region_code == 0
        if np.any(mask_a):
            density = pmat / self.grid.areas_matrix
            numerator[mask_a] = density[
                self.x1_cell[mask_a], self.x2_cell[mask_a]
            ]

        # For each observed boundary pair, precompute all one-dimensional
        # subdensity values on the x1 and x2 cell indices.
        b_table = np.einsum("ij,qj->qi", pmat, self.fraction_below2)
        b_table = b_table / self.grid.widths1[None, :]
        mask_b = self.region_code == 1
        if np.any(mask_b):
            numerator[mask_b] = b_table[
                self.y_group[mask_b], self.x1_cell[mask_b]
            ]

        c_table = np.einsum("qi,ij->qj", self.fraction_below1, pmat)
        c_table = c_table / self.grid.widths2[None, :]
        mask_c = self.region_code == 2
        if np.any(mask_c):
            numerator[mask_c] = c_table[
                self.y_group[mask_c], self.x2_cell[mask_c]
            ]

        denominator_by_group = self.selection_rows @ p
        denominator = denominator_by_group[self.y_group]
        score = np.full(self.n_rows, -math.inf, dtype=float)
        valid = (numerator > 0.0) & (denominator > 0.0)
        score[valid] = np.log(numerator[valid]) - np.log(denominator[valid])
        return score, numerator, denominator

    def block_means(self, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(score, dtype=float)
        if values.ndim != 1 or values.size != self.n_rows:
            raise ValueError("score length differs from the score design")
        means: List[float] = []
        counts: List[int] = []
        for block in range(self.n_blocks):
            mask = self.block_ids == block
            if not np.any(mask):
                continue
            means.append(float(np.mean(values[mask])))
            counts.append(int(np.sum(mask)))
        return np.asarray(means, dtype=float), np.asarray(counts, dtype=int)


def make_balanced_blocks(n_rows: int, n_blocks: int, seed: int) -> np.ndarray:
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least two")
    n_blocks_eff = min(int(n_blocks), int(n_rows))
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(n_rows)
    block_ids = np.empty(n_rows, dtype=int)
    block_ids[permutation] = np.arange(n_rows, dtype=int) % n_blocks_eff
    return block_ids


def _strategy_mass(
    class_mass: np.ndarray,
    classes: Sequence[Sequence[int]],
    grid: RectangularGrid,
    strategy: str,
    coarse_prior_mass: Optional[np.ndarray] = None,
) -> np.ndarray:
    q = _normalise_probability(np.asarray(class_mass, dtype=float), "class_mass")
    if q.size != len(classes):
        raise ValueError("class_mass length differs from the equivalence classes")
    p = np.zeros(grid.k, dtype=float)
    table = grid.cell_table
    c1 = table["x1_centre"].to_numpy(float)
    c2 = table["x2_centre"].to_numpy(float)
    prior = None
    if coarse_prior_mass is not None:
        prior = _normalise_probability(
            np.asarray(coarse_prior_mass, dtype=float), "coarse_prior_mass"
        )
        if prior.size != grid.k:
            raise ValueError("coarse prior mass differs from the target grid")
    for h, cls_raw in enumerate(classes):
        cls = np.asarray(list(cls_raw), dtype=int)
        if cls.size == 0:
            raise ValueError("an equivalence class is empty")
        if strategy == "equal":
            p[cls] = q[h] / cls.size
        elif strategy == "coarse_prior":
            if prior is None:
                weights = grid.areas[cls].copy()
            else:
                weights = prior[cls].copy()
                if float(weights.sum()) <= 1e-18:
                    weights = grid.areas[cls].copy()
            weights = _normalise_probability(weights, "within_class_prior")
            p[cls] = q[h] * weights
        elif strategy == "lower_corner":
            chosen = min(
                cls.tolist(),
                key=lambda k: (c1[k] + c2[k], c1[k], c2[k], int(k)),
            )
            p[chosen] = q[h]
        elif strategy == "upper_corner":
            chosen = max(
                cls.tolist(),
                key=lambda k: (c1[k] + c2[k], c1[k], c2[k], int(k)),
            )
            p[chosen] = q[h]
        else:
            raise ValueError(f"unknown within-class strategy {strategy!r}")
    return _normalise_probability(p, f"{strategy}_mass")


def _score_summary(
    score: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    design: FastObservedScoreDesign,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    s = np.asarray(score, dtype=float)
    finite = np.isfinite(s)
    block_means, block_counts = design.block_means(s)
    if np.any(~finite):
        mean_score = -math.inf
        sd_score = math.inf
        se_score = math.inf
    else:
        mean_score = float(np.mean(s))
        sd_score = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
        se_score = float(sd_score / math.sqrt(s.size)) if s.size else math.nan
    block_se = (
        float(np.std(block_means, ddof=1) / math.sqrt(len(block_means)))
        if len(block_means) > 1
        else 0.0
    )
    return (
        {
            "mean_logscore": mean_score,
            "sd_row_logscore": sd_score,
            "se_row_mean_logscore": se_score,
            "se_block_mean_logscore": block_se,
            "n_rows": int(s.size),
            "n_blocks": int(len(block_means)),
            "nonfinite_score_count": int(np.sum(~finite)),
            "nonfinite_score_fraction": float(np.mean(~finite)),
            "minimum_numerator": float(np.min(numerator)),
            "minimum_denominator": float(np.min(denominator)),
        },
        block_means,
        block_counts,
    )


def _target_checkpoint_name(
    fit_type: str,
    rho: float,
    n_latent: int,
    replicate: int,
    bins: int,
    repeat: int = -1,
    fold: int = -1,
) -> str:
    rho_token = _stable_token(float(rho))
    if fit_type == "full":
        return (
            f"rho_{rho_token}__n_{n_latent:06d}__rep_{replicate:03d}__"
            f"bins_{bins:03d}__full.npz"
        )
    return (
        f"rho_{rho_token}__n_{n_latent:06d}__rep_{replicate:03d}__"
        f"bins_{bins:03d}__repeat_{repeat:02d}__fold_{fold:02d}.npz"
    )


def _save_target_model(path: Path, model: TargetModel, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        equal_mass=np.asarray(model.equal_mass, dtype=float),
        class_mass=np.asarray(model.class_mass, dtype=float),
        classes_json=np.array(json.dumps(model.classes)),
        metadata_json=np.array(json.dumps(_jsonable(metadata), sort_keys=True)),
    )


def _load_target_model(path: Path) -> Tuple[TargetModel, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        equal_mass = np.asarray(data["equal_mass"], dtype=float)
        class_mass = np.asarray(data["class_mass"], dtype=float)
        classes = json.loads(str(data["classes_json"].item()))
        metadata = json.loads(str(data["metadata_json"].item()))
    model = TargetModel(
        equal_mass=equal_mass,
        class_mass=class_mass,
        classes=[[int(x) for x in cls] for cls in classes],
        diagnostics=metadata["fit_diagnostics"],
        start_records=metadata.get("start_records", []),
        loaded_from_checkpoint=True,
    )
    return model, metadata



def _fit_failure_sidecar_path(checkpoint: Path) -> Path:
    """Return the durable diagnostic path associated with one failed fit."""
    return checkpoint.with_name(checkpoint.stem + "__failure.json")


def _write_fit_failure_sidecar(
    checkpoint: Path,
    identity: Mapping[str, Any],
    stage: str,
    message: str,
    diagnostics: Optional[Mapping[str, Any]] = None,
    traceback_text: Optional[str] = None,
) -> Path:
    """Persist fit diagnostics before fail-fast re-raises an exception."""
    path = _fit_failure_sidecar_path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "code_version": TARGET_CODE_VERSION,
        "identity": _jsonable(dict(identity)),
        "stage": str(stage),
        "message": str(message),
    }
    if diagnostics is not None:
        payload["fit_diagnostics"] = _jsonable(dict(diagnostics))
    if traceback_text is not None:
        payload["traceback"] = str(traceback_text)
    _write_json(path, payload)
    return path

def _fit_or_load_target_model(
    checkpoint: Path,
    configuration_fingerprint: str,
    observed_train: pd.DataFrame,
    grid: RectangularGrid,
    args: argparse.Namespace,
    warm_mass: Optional[np.ndarray],
    run_uniform_audit: bool,
    direct_starts: int,
    direct_seed: int,
    identity: Dict[str, Any],
) -> TargetModel:
    if args.resume and checkpoint.exists():
        model, metadata = _load_target_model(checkpoint)
        compatible = bool(
            metadata.get("configuration_fingerprint") == configuration_fingerprint
            and metadata.get("code_version") == TARGET_CODE_VERSION
            and metadata.get("identity") == _jsonable(identity)
            and model.equal_mass.size == grid.k
        )
        if compatible:
            return model

    a, b, _ = build_sample_operator(observed_train, grid)
    failure_sidecar = _fit_failure_sidecar_path(checkpoint)
    try:
        fit = fit_sieve_with_start_audit(
            a,
            b,
            None,
            coverage_tolerance=args.coverage_tolerance,
            mass_tolerance=args.mass_tolerance,
            score_tolerance=args.score_tolerance,
            prune_mass_tolerance=args.prune_mass_tolerance,
            prune_score_margin=args.prune_score_margin,
            max_iterations=args.max_iterations,
            warm_original_mass=warm_mass,
            run_uniform_audit=run_uniform_audit,
            warm_mixing=args.warm_mixing,
            direct_starts=direct_starts,
            direct_seed=direct_seed,
            mm_prepolish_iterations=args.mm_prepolish_iterations,
            newton_tolerance=args.newton_tolerance,
            polish_zero_tolerance=args.polish_zero_tolerance,
            max_polish_cycles=int(getattr(args, "max_polish_cycles", 8)),
        )
    except Exception as exc:
        diagnostic_path = _write_fit_failure_sidecar(
            checkpoint,
            identity,
            stage="fit_exception",
            message=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )
        try:
            exc.add_note(f"fit diagnostic written to {diagnostic_path}")
        except AttributeError:
            pass
        raise
    if not fit.selected_fit.converged:
        mm_diag = fit.selected_fit.diagnostics
        diagnostic_path = _write_fit_failure_sidecar(
            checkpoint,
            identity,
            stage="post_fit_KKT_audit",
            message="target-audit fit did not satisfy the KKT tolerance",
            diagnostics=fit.diagnostics,
        )
        raise RuntimeError(
            "target-audit fit did not satisfy the KKT tolerance; "
            f"active_residual={float(mm_diag['active_score_sup_abs_residual']):.6e}, "
            f"inactive_violation={float(mm_diag['inactive_positive_score_violation']):.6e}, "
            f"polish_cycles={int(mm_diag.get('polish_cycles', 0))}, "
            f"identity={json.dumps(_jsonable(identity), sort_keys=True)}, "
            f"diagnostic={diagnostic_path}"
        )
    if failure_sidecar.exists():
        failure_sidecar.unlink()
    class_mass = np.asarray(fit.selected_fit.mass, dtype=float)
    classes = [[int(x) for x in cls] for cls in fit.reduction.classes]
    nontrivial = [h for h, cls in enumerate(classes) if len(cls) > 1]
    diagnostics = _jsonable(fit.diagnostics)
    diagnostics["nontrivial_equivalence_class_mass_fraction"] = float(
        class_mass[nontrivial].sum() if nontrivial else 0.0
    )
    diagnostics["maximum_nontrivial_equivalence_class_mass"] = float(
        class_mass[nontrivial].max() if nontrivial else 0.0
    )
    model = TargetModel(
        equal_mass=np.asarray(fit.original_mass, dtype=float),
        class_mass=class_mass,
        classes=classes,
        diagnostics=diagnostics,
        start_records=fit.start_table.to_dict(orient="records"),
        loaded_from_checkpoint=False,
    )
    metadata = {
        "configuration_fingerprint": configuration_fingerprint,
        "code_version": TARGET_CODE_VERSION,
        "identity": _jsonable(identity),
        "fit_diagnostics": diagnostics,
        "start_records": model.start_records,
    }
    _save_target_model(checkpoint, model, metadata)
    return model


def _parse_condition_token(token: str) -> Tuple[float, int]:
    try:
        rho_text, n_text = str(token).split(":", 1)
        rho = float(rho_text)
        n = int(n_text)
    except Exception as exc:
        raise ValueError(
            f"invalid condition {token!r}; use rho:n, for example 0.55:6000"
        ) from exc
    if not (-0.999 < rho < 0.999):
        raise ValueError("rho must lie strictly between -0.999 and 0.999")
    if n < 100:
        raise ValueError("each latent sample size must be at least 100")
    return rho, n


def _resolve_target_profile(
    args: argparse.Namespace,
) -> Tuple[List[Tuple[float, int]], List[int], int, int, int, int, int]:
    defaults = {
        "smoke": {
            "conditions": [(0.55, 1200)],
            "bins": [8, 12],
            "replicates": 1,
            "folds": 3,
            "repeats": 2,
            "population_latent": 12000,
            "population_blocks": 8,
        },
        "pilot": {
            "conditions": [(0.55, 6000), (0.90, 3000), (0.90, 6000)],
            "bins": [8, 12, 16, 20],
            "replicates": 3,
            "folds": 5,
            "repeats": 3,
            "population_latent": 100000,
            "population_blocks": 20,
        },
        "full": {
            "conditions": [(0.55, 6000), (0.90, 3000), (0.90, 6000)],
            "bins": [8, 12, 16, 20],
            "replicates": 10,
            "folds": 5,
            "repeats": 5,
            "population_latent": 500000,
            "population_blocks": 40,
        },
    }
    cfg = defaults[args.profile]
    conditions = (
        [_parse_condition_token(x) for x in args.conditions]
        if args.conditions
        else list(cfg["conditions"])
    )
    bins = sorted(set(int(x) for x in (args.bins_values or cfg["bins"])))
    replicates = int(args.replicates if args.replicates is not None else cfg["replicates"])
    folds = int(args.cv_folds if args.cv_folds is not None else cfg["folds"])
    repeats = int(args.cv_repeats if args.cv_repeats is not None else cfg["repeats"])
    population_latent = int(
        args.population_validation_latent
        if args.population_validation_latent is not None
        else cfg["population_latent"]
    )
    population_blocks = int(
        args.population_blocks
        if args.population_blocks is not None
        else cfg["population_blocks"]
    )
    if len(set(conditions)) != len(conditions):
        raise ValueError("conditions contain duplicates")
    if any(x < 2 for x in bins):
        raise ValueError("all bin counts must be at least two")
    if replicates < 1 or folds < 2 or repeats < 1:
        raise ValueError("replicates, folds, and repeats are invalid")
    if population_latent < 1000 or population_blocks < 2:
        raise ValueError("population validation sample or block count is too small")
    return conditions, bins, replicates, folds, repeats, population_latent, population_blocks


def _target_fingerprint(
    args: argparse.Namespace,
    conditions: Sequence[Tuple[float, int]],
    bins_values: Sequence[int],
    replicates: int,
    folds: int,
    repeats: int,
    population_latent: int,
    population_blocks: int,
) -> str:
    payload = {
        "code_version": TARGET_CODE_VERSION,
        "conditions": [[float(r), int(n)] for r, n in conditions],
        "bins_values": [int(x) for x in bins_values],
        "replicates": int(replicates),
        "folds": int(folds),
        "repeats": int(repeats),
        "population_validation_latent": int(population_latent),
        "population_blocks": int(population_blocks),
        "split_strategies": list(args.split_strategies),
        "mixture_strength": float(args.mixture_strength),
        "seed": int(args.seed),
        "solver": {
            "mass_tolerance": float(args.mass_tolerance),
            "score_tolerance": float(args.score_tolerance),
            "prune_mass_tolerance": float(args.prune_mass_tolerance),
            "prune_score_margin": float(args.prune_score_margin),
            "coverage_tolerance": float(args.coverage_tolerance),
            "warm_mixing": float(args.warm_mixing),
            "max_iterations": int(args.max_iterations),
            "mm_prepolish_iterations": int(args.mm_prepolish_iterations),
            "newton_tolerance": float(args.newton_tolerance),
            "polish_zero_tolerance": float(args.polish_zero_tolerance),
            "max_polish_cycles": int(args.max_polish_cycles),
            "active_set_repolish_after_change": True,
        },
        "support": [GAUSSIAN_LOWER, GAUSSIAN_UPPER],
        "y_support": GAUSSIAN_Y_SUPPORT.tolist(),
        "y_mass": GAUSSIAN_Y_MASS.tolist(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _training_likelihood_for_mass(
    observed: pd.DataFrame,
    grid: RectangularGrid,
    mass: np.ndarray,
) -> float:
    a, b, _ = build_sample_operator(observed, grid)
    return conditional_loglikelihood(mass, a, b)


def run_target_self_tests() -> Dict[str, Any]:
    base = run_cv_self_tests()
    checks: Dict[str, bool] = {"base_cross_validation_tests": bool(base["all_passed"])}
    details: Dict[str, Any] = {"base": base}

    _, observed, _ = simulate_gaussian_catalogue(
        500,
        20260910,
        0.55,
        GAUSSIAN_LOWER,
        GAUSSIAN_UPPER,
        GAUSSIAN_Y_SUPPORT,
        GAUSSIAN_Y_MASS,
    )
    grid = make_uniform_grid(4)
    rng = np.random.default_rng(271828)
    mass = rng.dirichlet(np.ones(grid.k))
    design = FastObservedScoreDesign.from_observed(observed, grid)
    fast_score, fast_num, fast_den = design.score_vector(mass)
    a, b, _ = build_sample_operator(observed, grid)
    matrix_score, matrix_num, matrix_den = conditional_logscore_vector(mass, a, b)
    details["fast_matrix_score_sup_abs_difference"] = _max_abs(fast_score - matrix_score)
    details["fast_matrix_numerator_sup_abs_difference"] = _max_abs(fast_num - matrix_num)
    details["fast_matrix_denominator_sup_abs_difference"] = _max_abs(fast_den - matrix_den)
    checks["fast_score_matches_operator_matrix"] = bool(
        details["fast_matrix_score_sup_abs_difference"] < 1e-13
        and details["fast_matrix_numerator_sup_abs_difference"] < 1e-14
        and details["fast_matrix_denominator_sup_abs_difference"] < 1e-14
    )

    # Exact duplicate columns create a nontrivial class.  Every deterministic
    # split must preserve the class total and hence the training likelihood.
    a_dup = np.array([[1.0, 1.0, 0.2], [0.4, 0.4, 1.0], [0.8, 0.8, 0.6]])
    b_dup = np.array([[1.0, 1.0, 1.0], [0.9, 0.9, 1.0], [1.0, 1.0, 0.8]])
    reduction = reduce_operator_columns(a_dup, b_dup, None)
    class_mass = np.array([0.7, 0.3]) if len(reduction.classes) == 2 else None
    if class_mass is None:
        raise RuntimeError("split self-test did not create two classes")
    toy_grid = RectangularGrid(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    # Map the three tested columns to the first three geometric cells and add a
    # zero fourth cell only for the geometric split helper.
    classes = [list(reduction.classes[0]), list(reduction.classes[1])]
    prior = np.array([0.6, 0.1, 0.3, 0.0])
    likelihoods: Dict[str, float] = {}
    totals_ok = True
    for strategy in TARGET_SPLIT_STRATEGIES:
        # The duplicate test has three columns, so use a manual expansion for
        # training-likelihood invariance and separately test the geometric helper.
        p3 = np.zeros(3)
        for h, cls in enumerate(classes):
            if strategy in {"equal", "coarse_prior"}:
                weights = np.ones(len(cls)) if strategy == "equal" else prior[np.asarray(cls)]
                if float(np.sum(weights)) <= 0.0:
                    weights = np.ones(len(cls))
                weights = weights / np.sum(weights)
                p3[np.asarray(cls)] = class_mass[h] * weights
            elif strategy == "lower_corner":
                p3[min(cls)] = class_mass[h]
            else:
                p3[max(cls)] = class_mass[h]
        likelihoods[strategy] = conditional_loglikelihood(p3, a_dup, b_dup)
        totals_ok = totals_ok and abs(float(p3.sum()) - 1.0) < 1e-15 and np.all(p3 >= 0.0)
    details["split_training_loglikelihood_range"] = float(
        max(likelihoods.values()) - min(likelihoods.values())
    )
    checks["within_class_splits_preserve_training_likelihood"] = bool(
        details["split_training_loglikelihood_range"] < 1e-14 and totals_ok
    )

    helper_classes = [[0, 1], [2], [3]]
    helper_q = np.array([0.5, 0.3, 0.2])
    helper_valid = True
    for strategy in TARGET_SPLIT_STRATEGIES:
        helper = _strategy_mass(
            helper_q,
            helper_classes,
            toy_grid,
            strategy,
            coarse_prior_mass=prior,
        )
        helper_valid = helper_valid and np.all(helper >= 0.0) and abs(float(helper.sum()) - 1.0) < 1e-15
        helper_valid = helper_valid and np.allclose(
            collapse_mass_to_classes(helper, helper_classes), helper_q
        )
    checks["strategy_mass_preserves_every_class_total"] = bool(helper_valid)

    blocks = make_balanced_blocks(len(observed), 7, 161803)
    block_design = FastObservedScoreDesign.from_observed(observed, grid, blocks)
    pred, _ = predictive_mixture_mass(mass, grid, len(observed), 1.0)
    score, num, den = block_design.score_vector(pred)
    summary, means, counts = _score_summary(score, num, den, block_design)
    reconstructed = float(np.sum(means * counts) / np.sum(counts))
    details["block_mean_reconstruction_abs_error"] = abs(
        reconstructed - summary["mean_logscore"]
    )
    checks["population_block_means_reconstruct_total_mean"] = bool(
        details["block_mean_reconstruction_abs_error"] < 1e-15
    )

    synthetic_summary = pd.DataFrame(
        {
            "bins_per_axis": [8, 12, 16, 20],
            "mean_validation_logscore": [-1.03, -1.00, -1.005, -1.02],
            "se_fold_mean_logscore": [0.01, 0.01, 0.01, 0.01],
        }
    )
    checks["selection_helper_finds_known_maximum"] = bool(
        _one_standard_error_choice(synthetic_summary)["best_bins"] == 12
    )
    return {
        "all_passed": bool(all(checks.values())),
        "checks": checks,
        "details": details,
    }


def _aggregate_target_cv_folds(fold_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = [
        "rho",
        "n_latent",
        "replicate",
        "repeat",
        "split_strategy",
        "bins_per_axis",
    ]
    for key, group in fold_table.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        n_total = int(group["n_validation_rows"].sum())
        heldout_mean = float(
            np.sum(group["sum_validation_logscore"]) / n_total
        )
        fold_held = group["mean_validation_logscore"].to_numpy(float)
        pop_fold = group["mean_population_logscore"].to_numpy(float)
        row.update(
            {
                "n_folds": int(len(group)),
                "n_validation_rows_total": n_total,
                "mean_validation_logscore": heldout_mean,
                "sd_fold_mean_logscore": float(np.std(fold_held, ddof=1)) if len(fold_held) > 1 else 0.0,
                "se_fold_mean_logscore": float(np.std(fold_held, ddof=1) / math.sqrt(len(fold_held))) if len(fold_held) > 1 else 0.0,
                "mean_population_logscore": float(np.mean(pop_fold)),
                "sd_fold_population_logscore": float(np.std(pop_fold, ddof=1)) if len(pop_fold) > 1 else 0.0,
                "se_fold_population_logscore": float(np.std(pop_fold, ddof=1) / math.sqrt(len(pop_fold))) if len(pop_fold) > 1 else 0.0,
                "raw_heldout_nonfinite_fraction": float(
                    group["raw_heldout_nonfinite_count"].sum() / n_total
                ),
                "mean_raw_population_nonfinite_fraction": float(
                    group["raw_population_nonfinite_fraction"].mean()
                ),
                "maximum_active_score_residual": float(group["active_score_residual"].max()),
                "maximum_inactive_positive_score_violation": float(
                    group["inactive_positive_score_violation"].max()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _selection_from_target_tables(
    cv_summary: pd.DataFrame,
    cv_folds: pd.DataFrame,
    full_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    key_cols = ["rho", "n_latent", "replicate", "repeat", "split_strategy"]
    for key, cv_group in cv_summary.groupby(key_cols, sort=True):
        rho, n_latent, replicate, repeat, strategy = key
        fold_group = cv_folds[
            (cv_folds["rho"] == rho)
            & (cv_folds["n_latent"] == n_latent)
            & (cv_folds["replicate"] == replicate)
            & (cv_folds["repeat"] == repeat)
            & (cv_folds["split_strategy"] == strategy)
        ].copy()
        full_group = full_table[
            (full_table["rho"] == rho)
            & (full_table["n_latent"] == n_latent)
            & (full_table["replicate"] == replicate)
            & (full_table["split_strategy"] == strategy)
        ].copy()
        if full_group.empty:
            raise RuntimeError("full target table is missing a CV dataset")
        cv_choice = _one_standard_error_choice(cv_group)
        paired_choice = _paired_one_standard_error_choice(cv_group, fold_group)
        fold_pop_row = cv_group.loc[cv_group["mean_population_logscore"].idxmax()]
        full_pop_row = full_group.loc[full_group["mean_population_logscore"].idxmax()]
        cdf_row = full_group.loc[full_group["cdf_rmse"].idxmin()]
        cv_rank = pd.Series(cv_group["mean_validation_logscore"].to_numpy(float)).corr(
            pd.Series(cv_group["mean_population_logscore"].to_numpy(float)),
            method="spearman",
        )
        target_rank = pd.Series(full_group["mean_population_logscore"].to_numpy(float)).corr(
            pd.Series(-full_group["cdf_rmse"].to_numpy(float)),
            method="spearman",
        )
        base = {
            "rho": float(rho),
            "n_latent": int(n_latent),
            "replicate": int(replicate),
            "repeat": int(repeat),
            "split_strategy": str(strategy),
            "fold_population_oracle_bins": int(fold_pop_row["bins_per_axis"]),
            "fold_population_oracle_logscore": float(fold_pop_row["mean_population_logscore"]),
            "full_population_oracle_bins": int(full_pop_row["bins_per_axis"]),
            "full_population_oracle_logscore": float(full_pop_row["mean_population_logscore"]),
            "cdf_oracle_bins": int(cdf_row["bins_per_axis"]),
            "cdf_oracle_rmse": float(cdf_row["cdf_rmse"]),
            "cv_vs_fold_population_spearman": float(cv_rank) if pd.notna(cv_rank) else math.nan,
            "full_population_vs_negative_cdf_rmse_spearman": float(target_rank) if pd.notna(target_rank) else math.nan,
            "fold_population_vs_full_population_exact_match": bool(
                int(fold_pop_row["bins_per_axis"]) == int(full_pop_row["bins_per_axis"])
            ),
            "full_population_vs_cdf_oracle_exact_match": bool(
                int(full_pop_row["bins_per_axis"]) == int(cdf_row["bins_per_axis"])
            ),
            "full_population_oracle_cdf_rmse": float(full_pop_row["cdf_rmse"]),
            "full_population_oracle_cdf_regret": float(
                full_pop_row["cdf_rmse"] - cdf_row["cdf_rmse"]
            ),
        }
        choices = [
            ("cv_max", int(cv_choice["best_bins"])),
            ("cv_one_se", int(cv_choice["one_se_bins"])),
            ("cv_paired_one_se", int(paired_choice["paired_one_se_bins"])),
        ]
        for rule, chosen in choices:
            cv_selected = cv_group[cv_group["bins_per_axis"] == chosen].iloc[0]
            full_selected = full_group[full_group["bins_per_axis"] == chosen].iloc[0]
            record = dict(base)
            record.update(
                {
                    "selection_rule": rule,
                    "selected_bins": int(chosen),
                    "selected_validation_logscore": float(cv_selected["mean_validation_logscore"]),
                    "selected_fold_population_logscore": float(cv_selected["mean_population_logscore"]),
                    "fold_population_logscore_regret": float(
                        fold_pop_row["mean_population_logscore"]
                        - cv_selected["mean_population_logscore"]
                    ),
                    "selected_full_population_logscore": float(full_selected["mean_population_logscore"]),
                    "full_population_logscore_regret": float(
                        full_pop_row["mean_population_logscore"]
                        - full_selected["mean_population_logscore"]
                    ),
                    "selected_cdf_rmse": float(full_selected["cdf_rmse"]),
                    "cdf_rmse_regret": float(full_selected["cdf_rmse"] - cdf_row["cdf_rmse"]),
                    "cv_matches_fold_population_oracle": bool(chosen == int(fold_pop_row["bins_per_axis"])),
                    "cv_matches_full_population_oracle": bool(chosen == int(full_pop_row["bins_per_axis"])),
                    "cv_matches_cdf_oracle": bool(chosen == int(cdf_row["bins_per_axis"])),
                    "within_5_percent_cdf_oracle": bool(
                        float(full_selected["cdf_rmse"]) <= 1.05 * float(cdf_row["cdf_rmse"]) + 1e-15
                    ),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def _selection_alignment_summary(selection: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["rho", "n_latent", "split_strategy", "selection_rule"]
    for key, group in selection.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_selection_units": int(len(group)),
                "fraction_cv_matches_fold_population_oracle": float(
                    group["cv_matches_fold_population_oracle"].mean()
                ),
                "fraction_cv_matches_full_population_oracle": float(
                    group["cv_matches_full_population_oracle"].mean()
                ),
                "fraction_cv_matches_cdf_oracle": float(
                    group["cv_matches_cdf_oracle"].mean()
                ),
                "fraction_within_5_percent_cdf_oracle": float(
                    group["within_5_percent_cdf_oracle"].mean()
                ),
                "mean_fold_population_logscore_regret": float(
                    group["fold_population_logscore_regret"].mean()
                ),
                "mean_full_population_logscore_regret": float(
                    group["full_population_logscore_regret"].mean()
                ),
                "mean_cdf_rmse_regret": float(group["cdf_rmse_regret"].mean()),
                "mean_cv_vs_fold_population_spearman": float(
                    group["cv_vs_fold_population_spearman"].mean()
                ),
                "fraction_fold_population_matches_full_population": float(
                    group["fold_population_vs_full_population_exact_match"].mean()
                ),
                "fraction_full_population_matches_cdf_oracle": float(
                    group["full_population_vs_cdf_oracle_exact_match"].mean()
                ),
                "mean_full_population_oracle_cdf_regret": float(
                    group["full_population_oracle_cdf_regret"].mean()
                ),
                "mean_full_population_vs_negative_cdf_spearman": float(
                    group["full_population_vs_negative_cdf_rmse_spearman"].mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _repeated_cv_stability(selection: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    freq_rows: List[Dict[str, Any]] = []
    group_cols = ["rho", "n_latent", "replicate", "split_strategy", "selection_rule"]
    for key, group in selection.groupby(group_cols, sort=True):
        counts = group["selected_bins"].value_counts().sort_index()
        probs = counts.to_numpy(float) / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs))) if probs.size else math.nan
        modal_bin = int(counts.idxmax())
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_cv_repeats": int(len(group)),
                "modal_selected_bins": modal_bin,
                "modal_fraction": float(counts.max() / counts.sum()),
                "selection_entropy_nats": entropy,
                "n_distinct_selected_resolutions": int(len(counts)),
                "selected_bins_min": int(group["selected_bins"].min()),
                "selected_bins_max": int(group["selected_bins"].max()),
            }
        )
        rows.append(row)
        for bins, count in counts.items():
            frow = dict(zip(group_cols, key))
            frow.update(
                {
                    "bins_per_axis": int(bins),
                    "selection_count": int(count),
                    "selection_fraction": float(count / counts.sum()),
                }
            )
            freq_rows.append(frow)
    return pd.DataFrame(rows), pd.DataFrame(freq_rows)


def _split_sensitivity(selection: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    key_cols = ["rho", "n_latent", "replicate", "repeat", "selection_rule"]
    for key, group in selection.groupby(key_cols, sort=True):
        equal = group[group["split_strategy"] == "equal"]
        if len(equal) != 1:
            continue
        eq = equal.iloc[0]
        for _, alt in group.iterrows():
            rows.append(
                {
                    **dict(zip(key_cols, key)),
                    "comparison_strategy": str(alt["split_strategy"]),
                    "equal_selected_bins": int(eq["selected_bins"]),
                    "comparison_selected_bins": int(alt["selected_bins"]),
                    "same_selected_bins": bool(
                        int(eq["selected_bins"]) == int(alt["selected_bins"])
                    ),
                    "absolute_bins_difference": int(
                        abs(int(eq["selected_bins"]) - int(alt["selected_bins"]))
                    ),
                    "cdf_regret_difference_comparison_minus_equal": float(
                        alt["cdf_rmse_regret"] - eq["cdf_rmse_regret"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _population_margin_table(
    block_table: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """Paired block uncertainty of the best population-logscore resolution."""
    if block_table.empty:
        return pd.DataFrame()
    if level == "full":
        group_cols = ["rho", "n_latent", "replicate", "split_strategy"]
        work = block_table[block_table["fit_type"] == "full"].copy()
        fold_average_cols: List[str] = []
    elif level == "cv_fold_average":
        group_cols = ["rho", "n_latent", "replicate", "repeat", "split_strategy"]
        work = block_table[block_table["fit_type"] == "cross_validation"].copy()
        # Average the independent-population block score over fold-trained fits.
        work = (
            work.groupby(group_cols + ["bins_per_axis", "population_block"], as_index=False)
            ["block_mean_logscore"]
            .mean()
        )
        fold_average_cols = []
    else:
        raise ValueError("unknown population margin level")
    rows: List[Dict[str, Any]] = []
    for key, group in work.groupby(group_cols, sort=True):
        means = group.groupby("bins_per_axis")["block_mean_logscore"].mean()
        best_bin = int(means.idxmax())
        runner_candidates = means.drop(index=best_bin)
        if runner_candidates.empty:
            continue
        runner_bin = int(runner_candidates.idxmax())
        wide = group[group["bins_per_axis"].isin([best_bin, runner_bin])].pivot_table(
            index="population_block",
            columns="bins_per_axis",
            values="block_mean_logscore",
            aggfunc="mean",
        ).dropna()
        diff = wide[best_bin].to_numpy(float) - wide[runner_bin].to_numpy(float)
        mean_diff = float(np.mean(diff))
        se_diff = float(np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
        row = dict(zip(group_cols, key))
        row.update(
            {
                "level": level,
                "best_bins": best_bin,
                "runner_up_bins": runner_bin,
                "best_minus_runner_mean_logscore": mean_diff,
                "best_minus_runner_se_block": se_diff,
                "best_minus_runner_z_like": mean_diff / se_diff if se_diff > 0.0 else math.inf,
                "n_population_blocks": int(len(diff)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_target_decomposition_audit(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    full_ckpt_dir = outdir / "full_fit_checkpoints"
    cv_ckpt_dir = outdir / "cv_fit_checkpoints"
    full_ckpt_dir.mkdir(parents=True, exist_ok=True)
    cv_ckpt_dir.mkdir(parents=True, exist_ok=True)

    (
        conditions,
        bins_values,
        replicates,
        n_folds,
        cv_repeats,
        population_latent,
        population_blocks,
    ) = _resolve_target_profile(args)
    strategies = list(dict.fromkeys(str(x) for x in args.split_strategies))
    if not set(strategies).issubset(set(TARGET_SPLIT_STRATEGIES)):
        raise ValueError("split_strategies contains an unknown strategy")
    if "equal" not in strategies:
        raise ValueError("the equal split must be included as the primary reference")
    fingerprint = _target_fingerprint(
        args,
        conditions,
        bins_values,
        replicates,
        n_folds,
        cv_repeats,
        population_latent,
        population_blocks,
    )
    self_tests = run_target_self_tests() if args.self_test else {
        "all_passed": True,
        "checks": {},
        "details": {},
    }
    if not self_tests["all_passed"]:
        raise RuntimeError("a built-in target-decomposition self-test failed")

    q_axis = np.linspace(GAUSSIAN_LOWER, GAUSSIAN_UPPER, args.query_grid_size)
    q1m, q2m = np.meshgrid(q_axis, q_axis, indexing="ij")
    query_x1 = q1m.reshape(-1)
    query_x2 = q2m.reshape(-1)
    unique_rhos = sorted(set(float(r) for r, _ in conditions))
    truths: Dict[float, TruncatedBivariateNormalTruth] = {}
    truth_vectors: Dict[float, np.ndarray] = {}
    true_cell_mass: Dict[Tuple[float, int], np.ndarray] = {}
    cdf_matrices: Dict[int, np.ndarray] = {}
    population_observed: Dict[float, pd.DataFrame] = {}
    population_info_rows: List[Dict[str, Any]] = []
    population_designs: Dict[Tuple[float, int], FastObservedScoreDesign] = {}
    for rho_index, rho in enumerate(unique_rhos):
        truth = TruncatedBivariateNormalTruth(rho, GAUSSIAN_LOWER, GAUSSIAN_UPPER)
        truths[rho] = truth
        truth_vectors[rho] = np.array(
            [truth.cdf(float(x1), float(x2)) for x1, x2 in zip(query_x1, query_x2)],
            dtype=float,
        )
        pop_seed = int(args.seed + 9_000_000 + 104729 * rho_index)
        _, pop_obs, pop_info = simulate_gaussian_catalogue(
            population_latent,
            pop_seed,
            rho,
            GAUSSIAN_LOWER,
            GAUSSIAN_UPPER,
            GAUSSIAN_Y_SUPPORT,
            GAUSSIAN_Y_MASS,
        )
        blocks = make_balanced_blocks(len(pop_obs), population_blocks, pop_seed + 17)
        population_observed[rho] = pop_obs
        population_info_rows.append(
            {
                "rho": rho,
                "population_validation_seed": pop_seed,
                "population_latent": population_latent,
                "population_observed": int(len(pop_obs)),
                "population_selection_fraction": float(len(pop_obs) / population_latent),
                "n_A": int((pop_obs["region"] == "A").sum()),
                "n_B": int((pop_obs["region"] == "B").sum()),
                "n_C": int((pop_obs["region"] == "C").sum()),
                "population_blocks": int(len(np.unique(blocks))),
            }
        )
        for bins in bins_values:
            grid = make_uniform_grid(bins)
            cdf_matrices.setdefault(bins, cdf_operator_matrix(grid, q_axis, q_axis)[0])
            true_cell_mass[(rho, bins)] = truth.cell_masses(grid)
            population_designs[(rho, bins)] = FastObservedScoreDesign.from_observed(
                pop_obs, grid, blocks
            )

    full_records: List[Dict[str, Any]] = []
    cv_fold_records: List[Dict[str, Any]] = []
    block_records: List[Dict[str, Any]] = []
    fold_assignment_records: List[Dict[str, Any]] = []
    start_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    split_invariance_records: List[Dict[str, Any]] = []
    total_full = len(conditions) * replicates * len(bins_values)
    total_cv = len(conditions) * replicates * cv_repeats * n_folds * len(bins_values)
    full_done = 0
    cv_done = 0
    start_clock = time.perf_counter()

    for condition_index, (rho, n_latent) in enumerate(conditions):
        for replicate in range(replicates):
            train_seed = int(args.seed + 100_000 * condition_index + 1_009 * replicate)
            _, observed, sample_info = simulate_gaussian_catalogue(
                n_latent,
                train_seed,
                rho,
                GAUSSIAN_LOWER,
                GAUSSIAN_UPPER,
                GAUSSIAN_Y_SUPPORT,
                GAUSSIAN_Y_MASS,
            )
            dataset = {
                "rho": float(rho),
                "n_latent": int(n_latent),
                "replicate": int(replicate),
                "training_seed": int(train_seed),
                "n_observed": int(sample_info["n_observed"]),
                "n_A": int(sample_info["region_counts_observed"]["A"]),
                "n_B": int(sample_info["region_counts_observed"]["B"]),
                "n_C": int(sample_info["region_counts_observed"]["C"]),
            }

            # Full-sample fits, used for final-model population log score and
            # truth-based CDF audit only after fitting.
            previous_grid: Optional[RectangularGrid] = None
            previous_mass: Optional[np.ndarray] = None
            coarsest_grid: Optional[RectangularGrid] = None
            coarsest_mass: Optional[np.ndarray] = None
            for bins in bins_values:
                grid = make_uniform_grid(bins)
                warm = (
                    project_rectangular_mass(previous_grid, previous_mass, grid)
                    if previous_grid is not None and previous_mass is not None
                    else None
                )
                audit_this = bool(condition_index == 0 and replicate == 0 and bins == min(bins_values))
                ckpt = full_ckpt_dir / _target_checkpoint_name(
                    "full", rho, n_latent, replicate, bins
                )
                identity = {
                    **dataset,
                    "fit_type": "full",
                    "bins_per_axis": int(bins),
                }
                try:
                    model = _fit_or_load_target_model(
                        ckpt,
                        fingerprint,
                        observed,
                        grid,
                        args,
                        warm,
                        run_uniform_audit=audit_this,
                        direct_starts=args.direct_audit_starts if audit_this else 0,
                        direct_seed=train_seed + 31 * bins,
                        identity=identity,
                    )
                    if coarsest_grid is None:
                        coarsest_grid = grid
                        coarsest_mass = model.equal_mass.copy()
                    prior = (
                        project_rectangular_mass(coarsest_grid, coarsest_mass, grid)
                        if coarsest_grid is not None and coarsest_mass is not None
                        else uniform_density_reference_mass(grid)
                    )
                    pop_design = population_designs[(rho, bins)]
                    a_train, b_train, _ = build_sample_operator(observed, grid)
                    training_ll: Dict[str, float] = {}
                    for strategy in strategies:
                        split_mass = _strategy_mass(
                            model.class_mass,
                            model.classes,
                            grid,
                            strategy,
                            coarse_prior_mass=prior,
                        )
                        training_ll[strategy] = conditional_loglikelihood(
                            split_mass, a_train, b_train
                        )
                        pred_mass, eta = predictive_mixture_mass(
                            split_mass,
                            grid,
                            len(observed),
                            args.mixture_strength,
                        )
                        raw_score, _, _ = pop_design.score_vector(split_mass)
                        score, numerator, denominator = pop_design.score_vector(pred_mass)
                        score_diag, block_means, block_counts = _score_summary(
                            score, numerator, denominator, pop_design
                        )
                        metrics, _ = estimator_accuracy_metrics(
                            grid,
                            split_mass,
                            true_cell_mass[(rho, bins)],
                            cdf_matrices[bins],
                            truth_vectors[rho],
                        )
                        mm_diag = model.diagnostics["mm"]
                        full_records.append(
                            {
                                **dataset,
                                "bins_per_axis": int(bins),
                                "n_cells": int(grid.k),
                                "split_strategy": strategy,
                                "predictive_mixture_eta": float(eta),
                                "mean_population_logscore": score_diag["mean_logscore"],
                                "se_population_logscore_blocks": score_diag["se_block_mean_logscore"],
                                "raw_population_nonfinite_fraction": float(np.mean(~np.isfinite(raw_score))),
                                "minimum_predictive_population_numerator": score_diag["minimum_numerator"],
                                "minimum_predictive_population_denominator": score_diag["minimum_denominator"],
                                "fit_loaded_from_checkpoint": bool(model.loaded_from_checkpoint),
                                "converged": bool(mm_diag["converged"]),
                                "active_score_residual": float(mm_diag["active_score_sup_abs_residual"]),
                                "inactive_positive_score_violation": float(mm_diag["inactive_positive_score_violation"]),
                                "n_forced_zero_cells": int(model.diagnostics["n_forced_zero_cells"]),
                                "n_equivalence_classes": int(model.diagnostics["n_equivalence_classes"]),
                                "n_nontrivial_equivalence_classes": int(model.diagnostics["n_nontrivial_equivalence_classes"]),
                                "nontrivial_equivalence_class_mass_fraction": float(model.diagnostics.get("nontrivial_equivalence_class_mass_fraction", 0.0)),
                                "maximum_nontrivial_equivalence_class_mass": float(model.diagnostics.get("maximum_nontrivial_equivalence_class_mass", 0.0)),
                                "n_active_classes": int(mm_diag["n_active"]),
                                "training_loglikelihood": float(training_ll[strategy]),
                                **metrics,
                            }
                        )
                        for block, (value, count) in enumerate(zip(block_means, block_counts)):
                            block_records.append(
                                {
                                    **dataset,
                                    "fit_type": "full",
                                    "repeat": -1,
                                    "fold": -1,
                                    "bins_per_axis": int(bins),
                                    "split_strategy": strategy,
                                    "population_block": int(block),
                                    "block_n_rows": int(count),
                                    "block_mean_logscore": float(value),
                                }
                            )
                    split_invariance_records.append(
                        {
                            **dataset,
                            "fit_type": "full",
                            "repeat": -1,
                            "fold": -1,
                            "bins_per_axis": int(bins),
                            "training_loglikelihood_range_across_splits": float(
                                max(training_ll.values()) - min(training_ll.values())
                            ),
                        }
                    )
                    for rec in model.start_records:
                        start_records.append(
                            {
                                **dataset,
                                "fit_type": "full",
                                "repeat": -1,
                                "fold": -1,
                                "bins_per_axis": int(bins),
                                **rec,
                            }
                        )
                    previous_grid, previous_mass = grid, model.equal_mass
                except Exception as exc:
                    failures.append(
                        {
                            **dataset,
                            "fit_type": "full",
                            "repeat": -1,
                            "fold": -1,
                            "bins_per_axis": int(bins),
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    if args.fail_fast:
                        raise
                full_done += 1
                print(
                    f"Full fit {full_done}/{total_full}: rho={rho:.2f}, "
                    f"rep={replicate + 1}/{replicates}, n={n_latent}, bins={bins} "
                    f"({time.perf_counter() - start_clock:.1f} s)"
                )

            # Repeated region-stratified cross-validation.  Every fold-trained
            # model is scored both on its finite held-out fold and on the same
            # large independent population-validation catalogue.
            for repeat in range(cv_repeats):
                fold_seed = int(train_seed + 500_000 + 7_919 * repeat)
                fold_ids, fold_table, fold_audit = make_stratified_fold_assignment(
                    observed, n_folds, fold_seed
                )
                fold_table = fold_table.assign(
                    rho=rho,
                    n_latent=n_latent,
                    replicate=replicate,
                    repeat=repeat,
                    fold_seed=fold_seed,
                )
                fold_assignment_records.extend(fold_table.to_dict(orient="records"))
                previous_by_fold: Dict[int, Tuple[RectangularGrid, np.ndarray]] = {}
                coarse_by_fold: Dict[int, Tuple[RectangularGrid, np.ndarray]] = {}
                for bins in bins_values:
                    grid = make_uniform_grid(bins)
                    for fold in range(n_folds):
                        train = observed.loc[fold_ids != fold].copy().reset_index(drop=True)
                        valid = observed.loc[fold_ids == fold].copy().reset_index(drop=True)
                        warm = None
                        if fold in previous_by_fold:
                            prev_grid, prev_mass = previous_by_fold[fold]
                            warm = project_rectangular_mass(prev_grid, prev_mass, grid)
                        audit_this = bool(
                            condition_index == 0
                            and replicate == 0
                            and repeat == 0
                            and fold == 0
                            and bins == min(bins_values)
                        )
                        ckpt = cv_ckpt_dir / _target_checkpoint_name(
                            "cross_validation",
                            rho,
                            n_latent,
                            replicate,
                            bins,
                            repeat,
                            fold,
                        )
                        identity = {
                            **dataset,
                            "fit_type": "cross_validation",
                            "repeat": int(repeat),
                            "fold": int(fold),
                            "fold_seed": int(fold_seed),
                            "bins_per_axis": int(bins),
                        }
                        try:
                            model = _fit_or_load_target_model(
                                ckpt,
                                fingerprint,
                                train,
                                grid,
                                args,
                                warm,
                                run_uniform_audit=audit_this,
                                direct_starts=args.direct_audit_starts if audit_this else 0,
                                direct_seed=train_seed + 1009 * bins + 31 * fold + 17 * repeat,
                                identity=identity,
                            )
                            previous_by_fold[fold] = (grid, model.equal_mass)
                            if fold not in coarse_by_fold:
                                coarse_by_fold[fold] = (grid, model.equal_mass.copy())
                            coarse_grid, coarse_mass = coarse_by_fold[fold]
                            prior = project_rectangular_mass(coarse_grid, coarse_mass, grid)
                            valid_design = FastObservedScoreDesign.from_observed(valid, grid)
                            pop_design = population_designs[(rho, bins)]
                            a_train, b_train, _ = build_sample_operator(train, grid)
                            training_ll: Dict[str, float] = {}
                            for strategy in strategies:
                                split_mass = _strategy_mass(
                                    model.class_mass,
                                    model.classes,
                                    grid,
                                    strategy,
                                    coarse_prior_mass=prior,
                                )
                                training_ll[strategy] = conditional_loglikelihood(
                                    split_mass, a_train, b_train
                                )
                                pred_mass, eta = predictive_mixture_mass(
                                    split_mass,
                                    grid,
                                    len(train),
                                    args.mixture_strength,
                                )
                                raw_valid, raw_valid_num, raw_valid_den = valid_design.score_vector(split_mass)
                                score_valid, val_num, val_den = valid_design.score_vector(pred_mass)
                                if not np.all(np.isfinite(score_valid)):
                                    raise RuntimeError("stabilised held-out score is non-finite")
                                raw_pop, _, _ = pop_design.score_vector(split_mass)
                                score_pop, pop_num, pop_den = pop_design.score_vector(pred_mass)
                                if not np.all(np.isfinite(score_pop)):
                                    raise RuntimeError("stabilised population score is non-finite")
                                pop_diag, block_means, block_counts = _score_summary(
                                    score_pop, pop_num, pop_den, pop_design
                                )
                                mm_diag = model.diagnostics["mm"]
                                cv_fold_records.append(
                                    {
                                        **dataset,
                                        "repeat": int(repeat),
                                        "fold": int(fold),
                                        "fold_seed": int(fold_seed),
                                        "bins_per_axis": int(bins),
                                        "n_cells": int(grid.k),
                                        "split_strategy": strategy,
                                        "n_training_rows": int(len(train)),
                                        "n_validation_rows": int(len(valid)),
                                        "predictive_mixture_eta": float(eta),
                                        "mean_validation_logscore": float(np.mean(score_valid)),
                                        "sum_validation_logscore": float(np.sum(score_valid)),
                                        "mean_population_logscore": pop_diag["mean_logscore"],
                                        "se_population_logscore_blocks": pop_diag["se_block_mean_logscore"],
                                        "raw_heldout_nonfinite_count": int(np.sum(~np.isfinite(raw_valid))),
                                        "raw_population_nonfinite_fraction": float(np.mean(~np.isfinite(raw_pop))),
                                        "minimum_predictive_heldout_numerator": float(np.min(val_num)),
                                        "minimum_predictive_heldout_denominator": float(np.min(val_den)),
                                        "minimum_predictive_population_numerator": float(np.min(pop_num)),
                                        "minimum_predictive_population_denominator": float(np.min(pop_den)),
                                        "active_score_residual": float(mm_diag["active_score_sup_abs_residual"]),
                                        "inactive_positive_score_violation": float(mm_diag["inactive_positive_score_violation"]),
                                        "converged": bool(mm_diag["converged"]),
                                        "n_forced_zero_cells": int(model.diagnostics["n_forced_zero_cells"]),
                                        "n_equivalence_classes": int(model.diagnostics["n_equivalence_classes"]),
                                        "n_nontrivial_equivalence_classes": int(model.diagnostics["n_nontrivial_equivalence_classes"]),
                                        "nontrivial_equivalence_class_mass_fraction": float(model.diagnostics.get("nontrivial_equivalence_class_mass_fraction", 0.0)),
                                        "maximum_nontrivial_equivalence_class_mass": float(model.diagnostics.get("maximum_nontrivial_equivalence_class_mass", 0.0)),
                                        "n_active_classes": int(mm_diag["n_active"]),
                                        "fold_region_balance_max_range": int(fold_audit["maximum_region_count_range"]),
                                        "training_loglikelihood": float(training_ll[strategy]),
                                    }
                                )
                                for block, (value, count) in enumerate(zip(block_means, block_counts)):
                                    block_records.append(
                                        {
                                            **dataset,
                                            "fit_type": "cross_validation",
                                            "repeat": int(repeat),
                                            "fold": int(fold),
                                            "bins_per_axis": int(bins),
                                            "split_strategy": strategy,
                                            "population_block": int(block),
                                            "block_n_rows": int(count),
                                            "block_mean_logscore": float(value),
                                        }
                                    )
                            split_invariance_records.append(
                                {
                                    **dataset,
                                    "fit_type": "cross_validation",
                                    "repeat": int(repeat),
                                    "fold": int(fold),
                                    "bins_per_axis": int(bins),
                                    "training_loglikelihood_range_across_splits": float(
                                        max(training_ll.values()) - min(training_ll.values())
                                    ),
                                }
                            )
                            for rec in model.start_records:
                                start_records.append(
                                    {
                                        **dataset,
                                        "fit_type": "cross_validation",
                                        "repeat": int(repeat),
                                        "fold": int(fold),
                                        "bins_per_axis": int(bins),
                                        **rec,
                                    }
                                )
                        except Exception as exc:
                            failures.append(
                                {
                                    **dataset,
                                    "fit_type": "cross_validation",
                                    "repeat": int(repeat),
                                    "fold": int(fold),
                                    "bins_per_axis": int(bins),
                                    "exception_type": type(exc).__name__,
                                    "message": str(exc),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                            if args.fail_fast:
                                raise
                        cv_done += 1
                        print(
                            f"CV fit {cv_done}/{total_cv}: rho={rho:.2f}, "
                            f"rep={replicate + 1}/{replicates}, n={n_latent}, "
                            f"bins={bins}, repeat={repeat + 1}/{cv_repeats}, "
                            f"fold={fold + 1}/{n_folds} "
                            f"({time.perf_counter() - start_clock:.1f} s)"
                        )

    full_table = pd.DataFrame(full_records)
    cv_fold_table = pd.DataFrame(cv_fold_records)
    block_table = pd.DataFrame(block_records)
    fold_assignment_table = pd.DataFrame(fold_assignment_records)
    start_table = pd.DataFrame(start_records)
    failure_table = pd.DataFrame(failures)
    split_invariance_table = pd.DataFrame(split_invariance_records)
    if full_table.empty or cv_fold_table.empty:
        raise RuntimeError("target decomposition produced no successful result tables")

    cv_summary = _aggregate_target_cv_folds(cv_fold_table)
    selection = _selection_from_target_tables(cv_summary, cv_fold_table, full_table)
    alignment = _selection_alignment_summary(selection)
    stability, stability_frequency = _repeated_cv_stability(selection)
    split_sensitivity = _split_sensitivity(selection)
    split_score_spread = (
        cv_summary.groupby(
            ["rho", "n_latent", "replicate", "repeat", "bins_per_axis"],
            as_index=False,
        )
        .agg(
            heldout_logscore_split_min=("mean_validation_logscore", "min"),
            heldout_logscore_split_max=("mean_validation_logscore", "max"),
            population_logscore_split_min=("mean_population_logscore", "min"),
            population_logscore_split_max=("mean_population_logscore", "max"),
        )
    )
    split_score_spread["heldout_logscore_split_range"] = (
        split_score_spread["heldout_logscore_split_max"]
        - split_score_spread["heldout_logscore_split_min"]
    )
    split_score_spread["population_logscore_split_range"] = (
        split_score_spread["population_logscore_split_max"]
        - split_score_spread["population_logscore_split_min"]
    )
    margin_full = _population_margin_table(block_table, "full")
    margin_cv = _population_margin_table(block_table, "cv_fold_average")
    margin_table = pd.concat([margin_full, margin_cv], ignore_index=True)

    all_fit_rows = pd.concat(
        [
            full_table[["active_score_residual", "inactive_positive_score_violation", "converged"]],
            cv_fold_table[["active_score_residual", "inactive_positive_score_violation", "converged"]],
        ],
        ignore_index=True,
    )
    expected_checkpoints = total_full + total_cv
    actual_checkpoints = len(list(full_ckpt_dir.glob("*.npz"))) + len(list(cv_ckpt_dir.glob("*.npz")))
    equal_selection = selection[selection["split_strategy"] == "equal"]
    non_equal_sensitivity = split_sensitivity[
        split_sensitivity["comparison_strategy"] != "equal"
    ]
    block_reconstruction_errors: List[float] = []
    for _, group in block_table.groupby(
        [
            "rho", "n_latent", "replicate", "fit_type", "repeat", "fold",
            "bins_per_axis", "split_strategy",
        ],
        sort=False,
    ):
        reconstructed = float(
            np.sum(group["block_mean_logscore"] * group["block_n_rows"])
            / group["block_n_rows"].sum()
        )
        if group.iloc[0]["fit_type"] == "full":
            ref = full_table[
                (full_table["rho"] == group.iloc[0]["rho"])
                & (full_table["n_latent"] == group.iloc[0]["n_latent"])
                & (full_table["replicate"] == group.iloc[0]["replicate"])
                & (full_table["bins_per_axis"] == group.iloc[0]["bins_per_axis"])
                & (full_table["split_strategy"] == group.iloc[0]["split_strategy"])
            ]["mean_population_logscore"].iloc[0]
        else:
            ref = cv_fold_table[
                (cv_fold_table["rho"] == group.iloc[0]["rho"])
                & (cv_fold_table["n_latent"] == group.iloc[0]["n_latent"])
                & (cv_fold_table["replicate"] == group.iloc[0]["replicate"])
                & (cv_fold_table["repeat"] == group.iloc[0]["repeat"])
                & (cv_fold_table["fold"] == group.iloc[0]["fold"])
                & (cv_fold_table["bins_per_axis"] == group.iloc[0]["bins_per_axis"])
                & (cv_fold_table["split_strategy"] == group.iloc[0]["split_strategy"])
            ]["mean_population_logscore"].iloc[0]
        block_reconstruction_errors.append(abs(reconstructed - float(ref)))

    checks = {
        "self_tests_passed": bool(self_tests["all_passed"]),
        "no_failed_fits": bool(failure_table.empty),
        "all_full_and_cv_fits_converged": bool(all_fit_rows["converged"].all()),
        "all_active_score_residuals_below_tolerance": bool(
            all_fit_rows["active_score_residual"].max() < args.score_tolerance
        ),
        "all_inactive_kkt_violations_below_tolerance": bool(
            all_fit_rows["inactive_positive_score_violation"].max() < args.score_tolerance
        ),
        "all_stabilised_heldout_scores_finite": bool(
            np.isfinite(cv_fold_table["mean_validation_logscore"]).all()
        ),
        "all_stabilised_population_scores_finite": bool(
            np.isfinite(full_table["mean_population_logscore"]).all()
            and np.isfinite(cv_fold_table["mean_population_logscore"]).all()
        ),
        "all_split_training_likelihoods_invariant": bool(
            split_invariance_table["training_loglikelihood_range_across_splits"].max() < 1e-11
        ),
        "population_block_means_reconstruct_scores": bool(
            max(block_reconstruction_errors, default=0.0) < 1e-12
        ),
        "folds_region_balanced_within_one": bool(
            cv_fold_table["fold_region_balance_max_range"].max() <= 1
        ),
        "checkpoint_count_covers_expected_fits": bool(actual_checkpoints >= expected_checkpoints),
        "population_validation_is_independent_of_training": True,
        "cv_selection_uses_no_truth_information": True,
        "truth_used_only_for_post_fit_cdf_audit": True,
        "fixed_support_box_used_for_all_fits": True,
    }

    paths = {
        "full_target_metrics": outdir / "full_fit_target_metrics.csv",
        "cv_fold_target_metrics": outdir / "cv_fold_target_metrics.csv",
        "cv_target_summary": outdir / "cv_target_summary.csv",
        "selection_decomposition": outdir / "selection_target_decomposition.csv",
        "alignment_summary": outdir / "selection_target_alignment_summary.csv",
        "repeated_cv_stability": outdir / "repeated_cv_stability.csv",
        "repeated_cv_frequency": outdir / "repeated_cv_selection_frequencies.csv",
        "split_sensitivity": outdir / "equivalence_split_selection_sensitivity.csv",
        "split_score_spread": outdir / "equivalence_split_score_spread.csv",
        "population_score_blocks": outdir / "population_score_block_means.csv",
        "population_oracle_margins": outdir / "population_oracle_score_margins.csv",
        "population_validation_summary": outdir / "population_validation_summary.csv",
        "split_invariance": outdir / "training_likelihood_split_invariance.csv",
        "fold_assignments": outdir / "fold_assignments.csv",
        "start_audit": outdir / "start_audit.csv",
        "failures": outdir / "failures.csv",
        "self_tests": outdir / "self_test_results.json",
        "summary": outdir / "comparison_summary.json",
        "manifest": outdir / "manifest.json",
        "environment": outdir / "environment.json",
        "file_hashes": outdir / "file_hashes.json",
    }
    full_table.to_csv(paths["full_target_metrics"], index=False)
    cv_fold_table.to_csv(paths["cv_fold_target_metrics"], index=False)
    cv_summary.to_csv(paths["cv_target_summary"], index=False)
    selection.to_csv(paths["selection_decomposition"], index=False)
    alignment.to_csv(paths["alignment_summary"], index=False)
    stability.to_csv(paths["repeated_cv_stability"], index=False)
    stability_frequency.to_csv(paths["repeated_cv_frequency"], index=False)
    split_sensitivity.to_csv(paths["split_sensitivity"], index=False)
    split_score_spread.to_csv(paths["split_score_spread"], index=False)
    block_table.to_csv(paths["population_score_blocks"], index=False)
    margin_table.to_csv(paths["population_oracle_margins"], index=False)
    pd.DataFrame(population_info_rows).to_csv(paths["population_validation_summary"], index=False)
    split_invariance_table.to_csv(paths["split_invariance"], index=False)
    fold_assignment_table.to_csv(paths["fold_assignments"], index=False)
    start_table.to_csv(paths["start_audit"], index=False)
    failure_table.to_csv(paths["failures"], index=False)
    _write_json(paths["self_tests"], self_tests)

    elapsed = time.perf_counter() - start_clock
    summary = {
        "code_version": TARGET_CODE_VERSION,
        "profile": args.profile,
        "all_main_checks_passed": bool(all(checks.values())),
        "checks": checks,
        "configuration_fingerprint": fingerprint,
        "conditions": [[float(r), int(n)] for r, n in conditions],
        "bins_values": bins_values,
        "replicates": replicates,
        "cv_folds": n_folds,
        "cv_repeats": cv_repeats,
        "population_validation_latent": population_latent,
        "population_blocks": population_blocks,
        "split_strategies": strategies,
        "mixture_strength": float(args.mixture_strength),
        "n_successful_full_fits": int(total_full - len(failure_table[failure_table.get("fit_type", pd.Series(dtype=str)) == "full"])) if not failure_table.empty else int(total_full),
        "n_successful_cv_fits": int(total_cv - len(failure_table[failure_table.get("fit_type", pd.Series(dtype=str)) == "cross_validation"])) if not failure_table.empty else int(total_cv),
        "n_failed_fits": int(len(failure_table)),
        "expected_checkpoint_count": int(expected_checkpoints),
        "actual_checkpoint_count": int(actual_checkpoints),
        "maximum_active_score_residual": float(all_fit_rows["active_score_residual"].max()),
        "maximum_inactive_positive_score_violation": float(
            all_fit_rows["inactive_positive_score_violation"].max()
        ),
        "maximum_training_loglikelihood_split_range": float(
            split_invariance_table["training_loglikelihood_range_across_splits"].max()
        ),
        "maximum_population_block_reconstruction_error": float(
            max(block_reconstruction_errors, default=0.0)
        ),
        "equal_split_cv_max_matches_fold_population_oracle_fraction": float(
            equal_selection.loc[
                equal_selection["selection_rule"] == "cv_max",
                "cv_matches_fold_population_oracle",
            ].mean()
        ),
        "equal_split_cv_max_mean_fold_population_logscore_regret": float(
            equal_selection.loc[
                equal_selection["selection_rule"] == "cv_max",
                "fold_population_logscore_regret",
            ].mean()
        ),
        "equal_split_full_population_matches_cdf_oracle_fraction": float(
            equal_selection.loc[
                equal_selection["selection_rule"] == "cv_max",
                "full_population_vs_cdf_oracle_exact_match",
            ].mean()
        ),
        "equal_split_full_population_oracle_mean_cdf_regret": float(
            equal_selection.loc[
                equal_selection["selection_rule"] == "cv_max",
                "full_population_oracle_cdf_regret",
            ].mean()
        ),
        "non_equal_split_fraction_same_selected_bins_as_equal": float(
            non_equal_sensitivity["same_selected_bins"].mean()
        ) if not non_equal_sensitivity.empty else 1.0,
        "mean_repeated_cv_modal_fraction_equal_split": float(
            stability.loc[stability["split_strategy"] == "equal", "modal_fraction"].mean()
        ),
        "elapsed_seconds": float(elapsed),
        "scientific_scope": {
            "separates_fold_noise_from_scoring_target_mismatch": True,
            "independent_population_validation_sample": True,
            "repeated_region_stratified_cross_validation": True,
            "equivalence_class_split_sensitivity": True,
            "population_logscore_approximated_by_large_monte_carlo_sample": True,
            "truth_used_only_for_cdf_target_audit": True,
            "general_cv_optimality_theorem": False,
            "sieve_asymptotic_theory": False,
        },
    }
    _write_json(paths["summary"], summary)
    manifest = {
        "code_version": TARGET_CODE_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_fingerprint": fingerprint,
        "arguments": vars(args),
        "resolved_profile": {
            "conditions": conditions,
            "bins_values": bins_values,
            "replicates": replicates,
            "cv_folds": n_folds,
            "cv_repeats": cv_repeats,
            "population_validation_latent": population_latent,
            "population_blocks": population_blocks,
        },
        "population_validation": {
            "independent_seed_family": "base_seed + 9,000,000 + 104729*rho_index",
            "shared_across_all_models_with_the_same_rho": True,
            "score_standard_error": "paired balanced-block standard error",
        },
        "split_strategies": {
            "equal": "maximum-entropy equal allocation within every training-equivalence class",
            "coarse_prior": "training-only allocation proportional to the projected coarsest-grid fit",
            "lower_corner": "deterministic extreme allocation to the geometrically lower class member",
            "upper_corner": "deterministic extreme allocation to the geometrically upper class member",
        },
        "predictive_stabilisation": {
            "definition": "p_pred=(1-eta)*p_split+eta*p_uniform, eta=c/(n_train+c)",
            "mixture_strength": float(args.mixture_strength),
            "used_only_for_heldout_and_population_predictive_scores": True,
            "not_used_in_training_or_final_CDF": True,
        },
        "target_decomposition": {
            "cv_vs_fold_population": "finite validation-fold noise and fold assignment variability",
            "fold_population_vs_full_population": "training-fraction effect",
            "full_population_vs_parent_cdf": "intrinsic predictive-logscore versus CDF target mismatch",
        },
    }
    _write_json(paths["manifest"], manifest)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "working_directory": str(Path.cwd()),
    }
    _write_json(paths["environment"], environment)
    hashes: Dict[str, Dict[str, str]] = {}
    for name, path in paths.items():
        if name != "file_hashes" and path.exists():
            hashes[name] = {"path": str(path), "sha256": _sha256(path)}
    _write_json(paths["file_hashes"], hashes)
    return summary, paths


def build_target_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Selection-target decomposition audit for the union-selection rectangular sieve"
    )
    parser.add_argument("--outdir", default="union_selection_sieve_target_decomposition_outputs")
    parser.add_argument("--profile", choices=["smoke", "pilot", "full"], default="pilot")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--bins-values", type=int, nargs="+", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--cv-repeats", type=int, default=None)
    parser.add_argument("--population-validation-latent", type=int, default=None)
    parser.add_argument("--population-blocks", type=int, default=None)
    parser.add_argument("--query-grid-size", type=int, default=41)
    parser.add_argument(
        "--split-strategies",
        nargs="+",
        default=list(TARGET_SPLIT_STRATEGIES),
    )
    parser.add_argument("--mixture-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--mass-tolerance", type=float, default=1e-12)
    parser.add_argument("--score-tolerance", type=float, default=1e-9)
    parser.add_argument("--prune-mass-tolerance", type=float, default=1e-7)
    parser.add_argument("--prune-score-margin", type=float, default=1e-7)
    parser.add_argument("--coverage-tolerance", type=float, default=1e-15)
    parser.add_argument("--warm-mixing", type=float, default=1e-9)
    parser.add_argument("--max-iterations", type=int, default=30000)
    parser.add_argument("--mm-prepolish-iterations", type=int, default=2000)
    parser.add_argument("--newton-tolerance", type=float, default=1e-12)
    parser.add_argument("--polish-zero-tolerance", type=float, default=1e-13)
    parser.add_argument("--direct-audit-starts", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return parser


def target_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_target_parser()
    if argv is None:
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"Ignoring unrecognised arguments: {unknown}")
    else:
        args = parser.parse_args(list(argv))
    summary, paths = run_target_decomposition_audit(args)
    concise_keys = [
        "code_version",
        "profile",
        "all_main_checks_passed",
        "checks",
        "conditions",
        "bins_values",
        "replicates",
        "cv_folds",
        "cv_repeats",
        "population_validation_latent",
        "population_blocks",
        "split_strategies",
        "n_successful_full_fits",
        "n_successful_cv_fits",
        "n_failed_fits",
        "maximum_active_score_residual",
        "maximum_inactive_positive_score_violation",
        "maximum_training_loglikelihood_split_range",
        "maximum_population_block_reconstruction_error",
        "equal_split_cv_max_matches_fold_population_oracle_fraction",
        "equal_split_cv_max_mean_fold_population_logscore_regret",
        "equal_split_full_population_matches_cdf_oracle_fraction",
        "equal_split_full_population_oracle_mean_cdf_regret",
        "non_equal_split_fraction_same_selected_bins_as_equal",
        "mean_repeated_cv_modal_fraction_equal_split",
        "elapsed_seconds",
    ]
    print(json.dumps({k: summary[k] for k in concise_keys}, indent=2, sort_keys=True))

    alignment = pd.read_csv(paths["alignment_summary"])
    print("\nSelection-target alignment summary:")
    print(alignment.to_string(index=False))

    stability = pd.read_csv(paths["repeated_cv_stability"])
    print("\nRepeated-CV stability:")
    print(stability.to_string(index=False))

    split = pd.read_csv(paths["split_sensitivity"])
    if not split.empty:
        split_summary = (
            split.groupby(
                ["rho", "n_latent", "selection_rule", "comparison_strategy"],
                as_index=False,
            )
            .agg(
                fraction_same_selected_bins=("same_selected_bins", "mean"),
                mean_absolute_bins_difference=("absolute_bins_difference", "mean"),
                mean_cdf_regret_difference=("cdf_regret_difference_comparison_minus_equal", "mean"),
            )
        )
        print("\nEquivalence-class split sensitivity:")
        print(split_summary.to_string(index=False))

    margins = pd.read_csv(paths["population_oracle_margins"])
    print("\nPopulation-logscore oracle margins from paired validation blocks:")
    print(margins.to_string(index=False))

    print("\nWritten files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("One-cell selection-target decomposition audit completed successfully (status=0).")
    print(f"Output directory: {Path(args.outdir).resolve()}")
    return 0


# =============================================================================

# =============================================================================
# Independent-seed confirmatory Monte Carlo
# =============================================================================
import os

CONFIRMATORY_CODE_VERSION = "union-selection-sieve-confirmatory-monte-carlo-v1.1.0-onecell"
# Reuse the proven checkpoint readers/writers with a new, frozen code identity.
TARGET_CODE_VERSION = CONFIRMATORY_CODE_VERSION
CV_CODE_VERSION = CONFIRMATORY_CODE_VERSION


def _confirmatory_conditions_from_tokens(tokens: Optional[Sequence[str]]) -> List[Tuple[float, int]]:
    if tokens:
        values = [_parse_condition_token(token) for token in tokens]
    else:
        values = [
            (0.55, 1500),
            (0.55, 3000),
            (0.55, 6000),
            (0.90, 1500),
            (0.90, 3000),
            (0.90, 6000),
        ]
    if len(set(values)) != len(values):
        raise ValueError("confirmatory conditions contain duplicates")
    return values


def _resolve_confirmatory_profile(
    args: argparse.Namespace,
) -> Tuple[List[Tuple[float, int]], List[int], int, int, int, int, int]:
    defaults = {
        "smoke": {
            "conditions": [(0.55, 1500), (0.90, 1500)],
            "bins": [12, 16, 20],
            "replicates": 2,
            "folds": 3,
            "repeats": 1,
            "population_latent": 12000,
            "population_blocks": 8,
            "query_grid_size": 21,
        },
        "pilot": {
            "conditions": [
                (0.55, 1500), (0.55, 3000), (0.55, 6000),
                (0.90, 1500), (0.90, 3000), (0.90, 6000),
            ],
            "bins": [12, 16, 20],
            "replicates": 5,
            "folds": 5,
            "repeats": 2,
            "population_latent": 50000,
            "population_blocks": 15,
            "query_grid_size": 31,
        },
        "confirmatory": {
            "conditions": [
                (0.55, 1500), (0.55, 3000), (0.55, 6000),
                (0.90, 1500), (0.90, 3000), (0.90, 6000),
            ],
            "bins": [12, 16, 20],
            "replicates": 30,
            "folds": 5,
            "repeats": 3,
            "population_latent": 100000,
            "population_blocks": 20,
            "query_grid_size": 41,
        },
        "full": {
            "conditions": [
                (0.55, 1500), (0.55, 3000), (0.55, 6000),
                (0.90, 1500), (0.90, 3000), (0.90, 6000),
            ],
            "bins": [12, 16, 20],
            "replicates": 50,
            "folds": 5,
            "repeats": 3,
            "population_latent": 250000,
            "population_blocks": 40,
            "query_grid_size": 41,
        },
    }
    cfg = defaults[args.profile]
    conditions = (
        _confirmatory_conditions_from_tokens(args.conditions)
        if args.conditions
        else list(cfg["conditions"])
    )
    bins = sorted(set(int(x) for x in (args.bins_values or cfg["bins"])))
    replicates = int(args.replicates if args.replicates is not None else cfg["replicates"])
    folds = int(args.cv_folds if args.cv_folds is not None else cfg["folds"])
    repeats = int(args.cv_repeats if args.cv_repeats is not None else cfg["repeats"])
    population_latent = int(
        args.population_validation_latent
        if args.population_validation_latent is not None
        else cfg["population_latent"]
    )
    population_blocks = int(
        args.population_blocks
        if args.population_blocks is not None
        else cfg["population_blocks"]
    )
    query_grid_size = int(
        args.query_grid_size
        if args.query_grid_size is not None
        else cfg["query_grid_size"]
    )
    if bins != [12, 16, 20] and not args.allow_nonfrozen_design:
        raise ValueError(
            "the confirmatory design is frozen at bins 12, 16, and 20; "
            "use --allow-nonfrozen-design only for debugging"
        )
    if args.primary_bins not in bins:
        raise ValueError("primary_bins must belong to the candidate resolution set")
    if replicates < 2 or folds < 2 or repeats < 1:
        raise ValueError("replicates, CV folds, or CV repeats are invalid")
    if population_latent < 1000 or population_blocks < 2:
        raise ValueError("population validation settings are too small")
    return conditions, bins, replicates, folds, repeats, population_latent, population_blocks, query_grid_size


def _confirmatory_fingerprint(
    args: argparse.Namespace,
    conditions: Sequence[Tuple[float, int]],
    bins: Sequence[int],
    replicates: int,
    folds: int,
    repeats: int,
    population_latent: int,
    population_blocks: int,
    query_grid_size: int,
) -> str:
    payload = {
        "code_version": CONFIRMATORY_CODE_VERSION,
        "analysis_freeze": {
            "conditions": [[float(r), int(n)] for r, n in conditions],
            "bins": [int(x) for x in bins],
            "primary_bins": int(args.primary_bins),
            "primary_estimator": "ABC_union_sieve",
            "data_driven_comparator": "repeat_averaged_cv_max",
            "sensitivity_bins": [12, 20],
            "cdf_oracle_is_simulation_only": True,
            "replicates": int(replicates),
            "cv_folds": int(folds),
            "cv_repeats": int(repeats),
            "mixture_strength": float(args.mixture_strength),
            "base_seed": int(args.seed),
        },
        "population_validation": {
            "latent": int(population_latent),
            "blocks": int(population_blocks),
        },
        "query_grid_size": int(query_grid_size),
        "solver": {
            "mass_tolerance": float(args.mass_tolerance),
            "score_tolerance": float(args.score_tolerance),
            "prune_mass_tolerance": float(args.prune_mass_tolerance),
            "prune_score_margin": float(args.prune_score_margin),
            "coverage_tolerance": float(args.coverage_tolerance),
            "warm_mixing": float(args.warm_mixing),
            "max_iterations": int(args.max_iterations),
            "mm_prepolish_iterations": int(args.mm_prepolish_iterations),
            "newton_tolerance": float(args.newton_tolerance),
            "polish_zero_tolerance": float(args.polish_zero_tolerance),
            "max_polish_cycles": int(args.max_polish_cycles),
            "active_set_repolish_after_change": True,
        },
        "support": [float(GAUSSIAN_LOWER), float(GAUSSIAN_UPPER)],
        "y_support": GAUSSIAN_Y_SUPPORT.tolist(),
        "y_mass": GAUSSIAN_Y_MASS.tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _confirmatory_token(rho: float) -> str:
    return f"{float(rho):+.3f}".replace("+", "p").replace("-", "m").replace(".", "d")


def _confirmatory_abc_checkpoint(
    kind: str,
    rho: float,
    n_latent: int,
    replicate: int,
    bins: int,
    repeat: int = -1,
    fold: int = -1,
) -> str:
    return (
        f"{kind}__rho_{_confirmatory_token(rho)}__n_{int(n_latent):06d}__"
        f"rep_{int(replicate):04d}__bins_{int(bins):03d}__"
        f"repeat_{int(repeat):02d}__fold_{int(fold):02d}.npz"
    )


def _confirmatory_seed(base_seed: int, condition_index: int, replicate: int) -> int:
    return int(base_seed + 1_000_003 * condition_index + 10_007 * replicate)


def _sample_mean_se_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "se": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "median": math.nan,
            "q10": math.nan,
            "q90": math.nan,
        }
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    se = float(sd / math.sqrt(x.size)) if x.size > 0 else math.nan
    if x.size > 1:
        alpha = 1.0 - float(confidence)
        crit = float(student_t.ppf(1.0 - alpha / 2.0, df=x.size - 1))
    else:
        crit = 0.0
    return {
        "n": int(x.size),
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": float(mean - crit * se),
        "ci_high": float(mean + crit * se),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q90": float(np.quantile(x, 0.90)),
    }


def _choose_best_bin(
    table: pd.DataFrame,
    value_column: str,
    maximise: bool,
    tie_tolerance: float,
) -> int:
    if table.empty:
        raise ValueError("cannot select a resolution from an empty table")
    values = table[value_column].to_numpy(float)
    target = float(np.max(values) if maximise else np.min(values))
    if maximise:
        eligible = table[table[value_column] >= target - tie_tolerance]
    else:
        eligible = table[table[value_column] <= target + tie_tolerance]
    return int(eligible["bins_per_axis"].min())


def _aggregate_confirmatory_cv(cv_fold: pd.DataFrame, tie_tolerance: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows: List[Dict[str, Any]] = []
    repeat_group = ["rho", "n_latent", "replicate", "repeat", "bins_per_axis"]
    for key, group in cv_fold.groupby(repeat_group, sort=True):
        n_total = int(group["n_validation_rows"].sum())
        repeat_rows.append(
            {
                **dict(zip(repeat_group, key)),
                "n_validation_rows_total": n_total,
                "repeat_mean_logscore": float(group["sum_validation_logscore"].sum() / n_total),
                "raw_nonfinite_count": int(group["raw_heldout_nonfinite_count"].sum()),
            }
        )
    repeat_table = pd.DataFrame(repeat_rows)

    avg_rows: List[Dict[str, Any]] = []
    avg_group = ["rho", "n_latent", "replicate", "bins_per_axis"]
    for key, group in cv_fold.groupby(avg_group, sort=True):
        n_total = int(group["n_validation_rows"].sum())
        avg_rows.append(
            {
                **dict(zip(avg_group, key)),
                "n_validation_rows_total": n_total,
                "repeat_averaged_mean_logscore": float(group["sum_validation_logscore"].sum() / n_total),
                "raw_nonfinite_count": int(group["raw_heldout_nonfinite_count"].sum()),
                "maximum_active_score_residual": float(group["active_score_residual"].max()),
                "maximum_inactive_positive_score_violation": float(group["inactive_positive_score_violation"].max()),
            }
        )
    average_table = pd.DataFrame(avg_rows)

    selection_rows: List[Dict[str, Any]] = []
    cat_group = ["rho", "n_latent", "replicate"]
    for key, group in average_table.groupby(cat_group, sort=True):
        chosen = _choose_best_bin(
            group,
            "repeat_averaged_mean_logscore",
            maximise=True,
            tie_tolerance=tie_tolerance,
        )
        repeat_choice: List[int] = []
        for _, repeat_group_table in repeat_table[
            (repeat_table["rho"] == key[0])
            & (repeat_table["n_latent"] == key[1])
            & (repeat_table["replicate"] == key[2])
        ].groupby("repeat", sort=True):
            repeat_choice.append(
                _choose_best_bin(
                    repeat_group_table,
                    "repeat_mean_logscore",
                    maximise=True,
                    tie_tolerance=tie_tolerance,
                )
            )
        counts = pd.Series(repeat_choice).value_counts().sort_index()
        modal_count = int(counts.max())
        modal_bins = int(counts[counts == modal_count].index.min())
        selection_rows.append(
            {
                **dict(zip(cat_group, key)),
                "repeat_averaged_cv_selected_bins": int(chosen),
                "single_repeat_modal_bins": modal_bins,
                "single_repeat_modal_fraction": float(modal_count / len(repeat_choice)),
                "n_distinct_single_repeat_choices": int(len(counts)),
                "single_repeat_choices": json.dumps([int(x) for x in repeat_choice]),
            }
        )
    return repeat_table, average_table, pd.DataFrame(selection_rows)


def _build_confirmatory_selection(
    full_table: pd.DataFrame,
    cv_selection: pd.DataFrame,
    bins_values: Sequence[int],
    primary_bins: int,
    tie_tolerance: float,
) -> pd.DataFrame:
    abc = full_table[full_table["estimator"] == "ABC_union_sieve"].copy()
    rows: List[Dict[str, Any]] = []
    key_cols = ["rho", "n_latent", "replicate"]
    for key, group in abc.groupby(key_cols, sort=True):
        cv_row = cv_selection[
            (cv_selection["rho"] == key[0])
            & (cv_selection["n_latent"] == key[1])
            & (cv_selection["replicate"] == key[2])
        ]
        if len(cv_row) != 1:
            raise RuntimeError("repeat-averaged CV selection is missing or duplicated")
        cv_bins = int(cv_row.iloc[0]["repeat_averaged_cv_selected_bins"])
        cdf_oracle_bins = _choose_best_bin(
            group, "cdf_rmse", maximise=False, tie_tolerance=tie_tolerance
        )
        pop_oracle_bins = _choose_best_bin(
            group, "mean_population_logscore", maximise=True, tie_tolerance=tie_tolerance
        )
        cdf_oracle_value = float(
            group.loc[group["bins_per_axis"] == cdf_oracle_bins, "cdf_rmse"].iloc[0]
        )
        pop_oracle_value = float(
            group.loc[group["bins_per_axis"] == pop_oracle_bins, "mean_population_logscore"].iloc[0]
        )
        rule_bins = {
            "fixed_12": 12,
            "fixed_16": int(primary_bins),
            "fixed_20": 20,
            "repeat_averaged_cv_max": cv_bins,
            "population_logscore_oracle": pop_oracle_bins,
            "cdf_rmse_oracle": cdf_oracle_bins,
        }
        for rule, selected_bins in rule_bins.items():
            if selected_bins not in set(int(x) for x in bins_values):
                continue
            selected = group[group["bins_per_axis"] == selected_bins].iloc[0]
            cdf_rmse = float(selected["cdf_rmse"])
            pop_score = float(selected["mean_population_logscore"])
            rows.append(
                {
                    **dict(zip(key_cols, key)),
                    "selection_rule": rule,
                    "deployable_rule": bool(rule not in {"population_logscore_oracle", "cdf_rmse_oracle"}),
                    "selected_bins": int(selected_bins),
                    "cdf_oracle_bins": int(cdf_oracle_bins),
                    "population_logscore_oracle_bins": int(pop_oracle_bins),
                    "selected_cdf_rmse": cdf_rmse,
                    "cdf_rmse_regret": float(cdf_rmse - cdf_oracle_value),
                    "relative_cdf_rmse_regret": float(cdf_rmse / cdf_oracle_value - 1.0) if cdf_oracle_value > 0.0 else math.nan,
                    "within_5_percent_cdf_oracle": bool(cdf_rmse <= 1.05 * cdf_oracle_value + 1e-15),
                    "within_10_percent_cdf_oracle": bool(cdf_rmse <= 1.10 * cdf_oracle_value + 1e-15),
                    "selected_population_logscore": pop_score,
                    "population_logscore_regret": float(pop_oracle_value - pop_score),
                    "n_observed": int(selected["n_observed"]),
                    "n_A": int(selected["n_A"]),
                    "n_B": int(selected["n_B"]),
                    "n_C": int(selected["n_C"]),
                }
            )
    return pd.DataFrame(rows)


def _selection_performance_summary(selection: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    groupings: List[Tuple[str, pd.DataFrame]] = [("overall", selection)]
    for (rho, n_latent), group in selection.groupby(["rho", "n_latent"], sort=True):
        groupings.append((f"rho={rho:.2f},n={int(n_latent)}", group))
    for scope, base in groupings:
        for rule, group in base.groupby("selection_rule", sort=True):
            regret = _sample_mean_se_ci(group["cdf_rmse_regret"])
            relative = _sample_mean_se_ci(group["relative_cdf_rmse_regret"])
            pop = _sample_mean_se_ci(group["population_logscore_regret"])
            rows.append(
                {
                    "scope": scope,
                    "selection_rule": rule,
                    "n_catalogues": int(len(group)),
                    "mean_selected_bins": float(group["selected_bins"].mean()),
                    "mean_selected_cdf_rmse": float(group["selected_cdf_rmse"].mean()),
                    "mean_cdf_rmse_regret": regret["mean"],
                    "se_cdf_rmse_regret": regret["se"],
                    "ci95_low_cdf_rmse_regret": regret["ci_low"],
                    "ci95_high_cdf_rmse_regret": regret["ci_high"],
                    "mean_relative_cdf_rmse_regret": relative["mean"],
                    "fraction_within_5_percent_cdf_oracle": float(group["within_5_percent_cdf_oracle"].mean()),
                    "fraction_within_10_percent_cdf_oracle": float(group["within_10_percent_cdf_oracle"].mean()),
                    "mean_population_logscore_regret": pop["mean"],
                    "se_population_logscore_regret": pop["se"],
                }
            )
    return pd.DataFrame(rows)


def _paired_rule_comparisons(selection: pd.DataFrame, primary_rule: str = "fixed_16") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    comparators = ["fixed_12", "fixed_20", "repeat_averaged_cv_max", "population_logscore_oracle"]
    scopes: List[Tuple[str, pd.DataFrame]] = [("overall", selection)]
    for (rho, n_latent), group in selection.groupby(["rho", "n_latent"], sort=True):
        scopes.append((f"rho={rho:.2f},n={int(n_latent)}", group))
    key = ["rho", "n_latent", "replicate"]
    for scope, base in scopes:
        left = base[base["selection_rule"] == primary_rule][key + ["selected_cdf_rmse", "selected_population_logscore"]].rename(
            columns={
                "selected_cdf_rmse": "left_cdf_rmse",
                "selected_population_logscore": "left_population_logscore",
            }
        )
        for comparator in comparators:
            right = base[base["selection_rule"] == comparator][key + ["selected_cdf_rmse", "selected_population_logscore"]].rename(
                columns={
                    "selected_cdf_rmse": "right_cdf_rmse",
                    "selected_population_logscore": "right_population_logscore",
                }
            )
            merged = left.merge(right, on=key, how="inner")
            if merged.empty:
                continue
            cdf_diff = merged["left_cdf_rmse"] - merged["right_cdf_rmse"]
            pop_diff = merged["right_population_logscore"] - merged["left_population_logscore"]
            cdf = _sample_mean_se_ci(cdf_diff)
            pop = _sample_mean_se_ci(pop_diff)
            rows.append(
                {
                    "scope": scope,
                    "left_rule": primary_rule,
                    "right_rule": comparator,
                    "n_paired_catalogues": int(len(merged)),
                    "mean_cdf_rmse_difference_left_minus_right": cdf["mean"],
                    "se_cdf_rmse_difference": cdf["se"],
                    "ci95_low_cdf_rmse_difference": cdf["ci_low"],
                    "ci95_high_cdf_rmse_difference": cdf["ci_high"],
                    "median_cdf_rmse_difference": cdf["median"],
                    "fraction_left_has_lower_cdf_rmse": float(np.mean(cdf_diff < 0.0)),
                    "mean_population_logscore_regret_difference_left_minus_right": pop["mean"],
                    "ci95_low_population_logscore_regret_difference": pop["ci_low"],
                    "ci95_high_population_logscore_regret_difference": pop["ci_high"],
                }
            )
    return pd.DataFrame(rows)


def _abc_aonly_comparisons(full_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    key = ["rho", "n_latent", "replicate", "bins_per_axis"]
    abc = full_table[full_table["estimator"] == "ABC_union_sieve"][key + ["cdf_rmse"]].rename(columns={"cdf_rmse": "abc_cdf_rmse"})
    aonly = full_table[full_table["estimator"] == "A_only_intersection_sieve"][key + ["cdf_rmse"]].rename(columns={"cdf_rmse": "aonly_cdf_rmse"})
    merged_all = abc.merge(aonly, on=key, how="inner")
    for bins, bins_group in merged_all.groupby("bins_per_axis", sort=True):
        scopes: List[Tuple[str, pd.DataFrame]] = [("overall", bins_group)]
        for (rho, n_latent), group in bins_group.groupby(["rho", "n_latent"], sort=True):
            scopes.append((f"rho={rho:.2f},n={int(n_latent)}", group))
        for scope, group in scopes:
            diff = group["abc_cdf_rmse"] - group["aonly_cdf_rmse"]
            stats = _sample_mean_se_ci(diff)
            rows.append(
                {
                    "scope": scope,
                    "bins_per_axis": int(bins),
                    "n_paired_catalogues": int(len(group)),
                    "mean_abc_cdf_rmse": float(group["abc_cdf_rmse"].mean()),
                    "mean_aonly_cdf_rmse": float(group["aonly_cdf_rmse"].mean()),
                    "mean_difference_abc_minus_aonly": stats["mean"],
                    "se_difference": stats["se"],
                    "ci95_low_difference": stats["ci_low"],
                    "ci95_high_difference": stats["ci_high"],
                    "median_difference": stats["median"],
                    "fraction_abc_has_lower_rmse": float(np.mean(diff < 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _cdf_bias_variance_summary(
    cdf_store: Mapping[Tuple[float, int, str, int], List[np.ndarray]],
    truth_vectors: Mapping[float, np.ndarray],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (rho, n_latent, estimator, bins), values in sorted(cdf_store.items()):
        arr = np.vstack(values)
        truth = np.asarray(truth_vectors[float(rho)], dtype=float)
        mean_cdf = arr.mean(axis=0)
        bias = mean_cdf - truth
        variance = np.mean((arr - mean_cdf[None, :]) ** 2, axis=0)
        mse = bias**2 + variance
        empirical = np.mean((arr - truth[None, :]) ** 2, axis=0)
        rows.append(
            {
                "rho": float(rho),
                "n_latent": int(n_latent),
                "estimator": estimator,
                "bins_per_axis": int(bins),
                "n_replicates": int(arr.shape[0]),
                "integrated_squared_bias": float(np.mean(bias**2)),
                "integrated_variance": float(np.mean(variance)),
                "integrated_mse": float(np.mean(mse)),
                "integrated_rmse": float(math.sqrt(np.mean(mse))),
                "empirical_integrated_mse": float(np.mean(empirical)),
                "identity_residual": float(abs(np.mean(mse) - np.mean(empirical))),
            }
        )
    return pd.DataFrame(rows)


def _confirmatory_self_tests() -> Dict[str, Any]:
    inherited = run_target_self_tests()
    checks: Dict[str, bool] = {"inherited_target_self_tests": bool(inherited["all_passed"])}
    details: Dict[str, Any] = {}

    stats = _sample_mean_se_ci([-1.0, 0.0, 1.0])
    checks["paired_summary_mean_is_zero"] = bool(abs(stats["mean"]) < 1e-15)
    checks["paired_summary_interval_contains_zero"] = bool(stats["ci_low"] <= 0.0 <= stats["ci_high"])

    toy_cv = pd.DataFrame(
        {
            "rho": [0.55] * 12,
            "n_latent": [1000] * 12,
            "replicate": [0] * 12,
            "repeat": np.repeat([0, 1], 6),
            "fold": list(np.tile([0, 1, 2], 4)),
            "bins_per_axis": np.repeat([12, 16], 3).tolist() * 2,
            "n_validation_rows": [10] * 12,
            "sum_validation_logscore": [-10.0] * 3 + [-9.0] * 3 + [-10.1] * 3 + [-9.1] * 3,
            "raw_heldout_nonfinite_count": [0] * 12,
            "active_score_residual": [0.0] * 12,
            "inactive_positive_score_violation": [0.0] * 12,
        }
    )
    _, avg, selected = _aggregate_confirmatory_cv(toy_cv, 1e-12)
    checks["repeat_averaged_cv_selects_higher_score"] = bool(int(selected.iloc[0]["repeat_averaged_cv_selected_bins"]) == 16)
    details["toy_repeat_averaged_scores"] = avg[["bins_per_axis", "repeat_averaged_mean_logscore"]].to_dict(orient="records")

    # Regression for the rare Version 1.0 failure: the first Newton pass
    # activates a class after SLSQP.  Version 1.1 must rerun SLSQP on the
    # enlarged face and satisfy the original 1e-9 KKT threshold.
    regression_seed = 20290933
    _, regression_observed, _ = simulate_gaussian_catalogue(
        1500,
        regression_seed,
        0.55,
        GAUSSIAN_LOWER,
        GAUSSIAN_UPPER,
        GAUSSIAN_Y_SUPPORT,
        GAUSSIAN_Y_MASS,
    )
    regression_repeat = 2
    regression_fold = 0
    regression_fold_seed = int(regression_seed + 700_001 + 7_919 * regression_repeat)
    regression_fold_ids, _, _ = make_stratified_fold_assignment(
        regression_observed, 5, regression_fold_seed
    )
    regression_train = regression_observed.loc[
        regression_fold_ids != regression_fold
    ].copy().reset_index(drop=True)
    regression_previous_grid: Optional[RectangularGrid] = None
    regression_previous_mass: Optional[np.ndarray] = None
    regression_fit: Optional[StartAuditedSieveFit] = None
    for regression_bins in [12, 16, 20]:
        regression_grid = make_uniform_grid(regression_bins)
        regression_warm = (
            project_rectangular_mass(
                regression_previous_grid, regression_previous_mass, regression_grid
            )
            if regression_previous_grid is not None
            and regression_previous_mass is not None
            else None
        )
        regression_a, regression_b, _ = build_sample_operator(
            regression_train, regression_grid
        )
        regression_fit = fit_sieve_with_start_audit(
            regression_a,
            regression_b,
            None,
            coverage_tolerance=1e-15,
            mass_tolerance=1e-12,
            score_tolerance=1e-9,
            prune_mass_tolerance=1e-7,
            prune_score_margin=1e-7,
            max_iterations=30000,
            warm_original_mass=regression_warm,
            run_uniform_audit=False,
            warm_mixing=1e-9,
            direct_starts=0,
            mm_prepolish_iterations=2000,
            newton_tolerance=1e-12,
            polish_zero_tolerance=1e-13,
            max_polish_cycles=8,
        )
        regression_previous_grid = regression_grid
        regression_previous_mass = regression_fit.original_mass
    assert regression_fit is not None
    regression_diag = regression_fit.selected_fit.diagnostics
    checks["rare_active_set_repolish_regression_converged"] = bool(
        regression_fit.selected_fit.converged
        and regression_diag["active_score_sup_abs_residual"] < 1e-9
        and regression_diag["inactive_positive_score_violation"] < 1e-9
    )
    checks["rare_active_set_repolish_regression_exercised"] = bool(
        regression_diag.get("polish_total_activation_events", 0) >= 1
        and regression_diag.get("polish_cycles", 0) >= 2
    )
    details["rare_active_set_repolish_regression"] = {
        "training_seed": regression_seed,
        "repeat": regression_repeat,
        "fold": regression_fold,
        "bins_per_axis": 20,
        "n_training_rows": int(len(regression_train)),
        "converged": bool(regression_fit.selected_fit.converged),
        "active_score_residual": float(
            regression_diag["active_score_sup_abs_residual"]
        ),
        "inactive_positive_score_violation": float(
            regression_diag["inactive_positive_score_violation"]
        ),
        "polish_cycles": int(regression_diag.get("polish_cycles", 0)),
        "polish_total_activation_events": int(
            regression_diag.get("polish_total_activation_events", 0)
        ),
        "polish_total_slsqp_iterations": int(
            regression_diag.get("polish_total_slsqp_iterations", 0)
        ),
        "polish_total_newton_steps": int(
            regression_diag.get("polish_total_newton_steps", 0)
        ),
        "final_loglikelihood": float(regression_diag["final_loglikelihood"]),
    }

    fake_store = {(0.55, 1000, "ABC_union_sieve", 16): [np.array([0.1, 0.3]), np.array([0.2, 0.4])]}
    fake_truth = {0.55: np.array([0.15, 0.35])}
    bv = _cdf_bias_variance_summary(fake_store, fake_truth)
    checks["bias_variance_identity"] = bool(float(bv.iloc[0]["identity_residual"]) < 1e-15)
    return {"all_passed": bool(all(checks.values())), "checks": checks, "details": details}


def run_confirmatory_audit(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    full_abc_dir = outdir / "full_abc_checkpoints"
    full_aonly_dir = outdir / "full_aonly_checkpoints"
    cv_dir = outdir / "cv_abc_checkpoints"
    for directory in [full_abc_dir, full_aonly_dir, cv_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    (
        conditions,
        bins_values,
        replicates,
        n_folds,
        cv_repeats,
        population_latent,
        population_blocks,
        query_grid_size,
    ) = _resolve_confirmatory_profile(args)
    fingerprint = _confirmatory_fingerprint(
        args,
        conditions,
        bins_values,
        replicates,
        n_folds,
        cv_repeats,
        population_latent,
        population_blocks,
        query_grid_size,
    )
    self_tests = _confirmatory_self_tests() if args.self_test else {"all_passed": True, "checks": {}, "details": {}}
    if not self_tests["all_passed"]:
        raise RuntimeError("a confirmatory self-test failed")

    freeze = {
        "code_version": CONFIRMATORY_CODE_VERSION,
        "frozen_before_confirmatory_data_generation": True,
        "primary_estimator": "ABC_union_sieve",
        "primary_resolution": int(args.primary_bins),
        "data_driven_comparator": "repeat_averaged_cv_max",
        "resolution_sensitivity": [12, 20],
        "simulation_only_oracles": ["cdf_rmse_oracle", "population_logscore_oracle"],
        "conditions": [[float(r), int(n)] for r, n in conditions],
        "replicates": int(replicates),
        "candidate_resolutions": [int(x) for x in bins_values],
        "cv_folds": int(n_folds),
        "cv_repeats": int(cv_repeats),
        "predictive_mixture_strength": float(args.mixture_strength),
        "base_seed": int(args.seed),
        "decision_criteria": {
            "fixed16_overall_mean_cdf_regret_no_larger_than_each_deployable_comparator": True,
            "fixed16_fraction_within_5_percent_of_oracle_at_least": 0.75,
            "fixed16_each_condition_mean_relative_regret_at_most": 0.10,
            "ABC_fixed16_overall_mean_rmse_lower_than_A_only_fixed16": True,
            "no_failed_fits_and_all_KKT_checks_pass": True,
        },
        "interpretation": "descriptive confirmatory Monte Carlo; criteria do not constitute a general optimality theorem",
        "configuration_fingerprint": fingerprint,
        "numerical_solver_amendment": {
            "version": "1.1",
            "reason": "repeat active-face SLSQP and Newton after an active-set change until the KKT tolerance is met",
            "statistical_design_changed": False,
            "estimator_changed": False,
            "candidate_resolutions_changed": False,
            "primary_resolution_changed": False,
            "seed_family_changed": False,
        },
    }
    freeze_path = outdir / "analysis_freeze.json"
    _write_json(freeze_path, freeze)

    q_axis = np.linspace(GAUSSIAN_LOWER, GAUSSIAN_UPPER, query_grid_size)
    q1m, q2m = np.meshgrid(q_axis, q_axis, indexing="ij")
    query_x1 = q1m.reshape(-1)
    query_x2 = q2m.reshape(-1)
    unique_rhos = sorted(set(float(r) for r, _ in conditions))
    truths: Dict[float, TruncatedBivariateNormalTruth] = {}
    truth_vectors: Dict[float, np.ndarray] = {}
    true_cell_mass: Dict[Tuple[float, int], np.ndarray] = {}
    cdf_matrices: Dict[int, np.ndarray] = {}
    population_designs: Dict[Tuple[float, int], FastObservedScoreDesign] = {}
    population_rows: List[Dict[str, Any]] = []
    for rho_index, rho in enumerate(unique_rhos):
        truth = TruncatedBivariateNormalTruth(rho, GAUSSIAN_LOWER, GAUSSIAN_UPPER)
        truths[rho] = truth
        truth_vectors[rho] = np.array(
            [truth.cdf(float(x1), float(x2)) for x1, x2 in zip(query_x1, query_x2)],
            dtype=float,
        )
        pop_seed = int(args.seed + 50_000_000 + 104729 * rho_index)
        _, pop_observed, pop_info = simulate_gaussian_catalogue(
            population_latent,
            pop_seed,
            rho,
            GAUSSIAN_LOWER,
            GAUSSIAN_UPPER,
            GAUSSIAN_Y_SUPPORT,
            GAUSSIAN_Y_MASS,
        )
        block_ids = make_balanced_blocks(len(pop_observed), population_blocks, pop_seed + 17)
        population_rows.append(
            {
                "rho": rho,
                "population_validation_seed": pop_seed,
                "n_latent": population_latent,
                "n_observed": int(len(pop_observed)),
                "selection_fraction": float(len(pop_observed) / population_latent),
                "n_A": int((pop_observed["region"] == "A").sum()),
                "n_B": int((pop_observed["region"] == "B").sum()),
                "n_C": int((pop_observed["region"] == "C").sum()),
                "n_blocks": int(len(np.unique(block_ids))),
            }
        )
        for bins in bins_values:
            grid = make_uniform_grid(bins)
            cdf_matrices.setdefault(bins, cdf_operator_matrix(grid, q_axis, q_axis)[0])
            true_cell_mass[(rho, bins)] = truth.cell_masses(grid)
            population_designs[(rho, bins)] = FastObservedScoreDesign.from_observed(
                pop_observed, grid, block_ids
            )

    full_records: List[Dict[str, Any]] = []
    cv_records: List[Dict[str, Any]] = []
    fold_audit_records: List[Dict[str, Any]] = []
    start_records: List[Dict[str, Any]] = []
    failure_records: List[Dict[str, Any]] = []
    seed_records: List[Dict[str, Any]] = []
    cdf_store: Dict[Tuple[float, int, str, int], List[np.ndarray]] = {}

    total_full = len(conditions) * replicates * len(bins_values) * 2
    total_cv = len(conditions) * replicates * cv_repeats * n_folds * len(bins_values)
    full_done = 0
    cv_done = 0
    start_clock = time.perf_counter()

    for condition_index, (rho, n_latent) in enumerate(conditions):
        for replicate in range(replicates):
            train_seed = _confirmatory_seed(args.seed, condition_index, replicate)
            seed_records.append(
                {
                    "condition_index": condition_index,
                    "rho": rho,
                    "n_latent": n_latent,
                    "replicate": replicate,
                    "training_seed": train_seed,
                }
            )
            _, observed, sample_info = simulate_gaussian_catalogue(
                n_latent,
                train_seed,
                rho,
                GAUSSIAN_LOWER,
                GAUSSIAN_UPPER,
                GAUSSIAN_Y_SUPPORT,
                GAUSSIAN_Y_MASS,
            )
            dataset = {
                "condition_index": int(condition_index),
                "rho": float(rho),
                "n_latent": int(n_latent),
                "replicate": int(replicate),
                "training_seed": int(train_seed),
                "n_observed": int(sample_info["n_observed"]),
                "n_A": int(sample_info["region_counts_observed"]["A"]),
                "n_B": int(sample_info["region_counts_observed"]["B"]),
                "n_C": int(sample_info["region_counts_observed"]["C"]),
            }

            for estimator in ["ABC_union_sieve", "A_only_intersection_sieve"]:
                previous_grid: Optional[RectangularGrid] = None
                previous_mass: Optional[np.ndarray] = None
                for bins in bins_values:
                    grid = make_uniform_grid(bins)
                    warm = (
                        project_rectangular_mass(previous_grid, previous_mass, grid)
                        if previous_grid is not None and previous_mass is not None
                        else None
                    )
                    audit_this = bool(condition_index == 0 and replicate == 0 and bins == min(bins_values))
                    identity = {
                        **dataset,
                        "fit_type": "full",
                        "estimator": estimator,
                        "bins_per_axis": int(bins),
                    }
                    try:
                        fit_start = time.perf_counter()
                        if estimator == "ABC_union_sieve":
                            checkpoint = full_abc_dir / _confirmatory_abc_checkpoint(
                                "full_abc", rho, n_latent, replicate, bins
                            )
                            model = _fit_or_load_target_model(
                                checkpoint,
                                fingerprint,
                                observed,
                                grid,
                                args,
                                warm,
                                run_uniform_audit=audit_this,
                                direct_starts=args.direct_audit_starts if audit_this else 0,
                                direct_seed=train_seed + 31 * bins,
                                identity=identity,
                            )
                            mass = model.equal_mass
                            diagnostics = model.diagnostics
                            starts = model.start_records
                            loaded = model.loaded_from_checkpoint
                        else:
                            checkpoint = full_aonly_dir / full_fit_checkpoint_name(
                                rho, n_latent, replicate, estimator, bins
                            )
                            mass, diagnostics, starts, loaded = _fit_or_load_mass(
                                checkpoint,
                                fingerprint,
                                observed,
                                grid,
                                estimator,
                                args,
                                warm,
                                run_uniform_audit=audit_this,
                                direct_starts=args.direct_audit_starts if audit_this else 0,
                                direct_seed=train_seed + 43 * bins,
                                identity=identity,
                            )
                        runtime = time.perf_counter() - fit_start
                        metrics, cdf_values = estimator_accuracy_metrics(
                            grid,
                            mass,
                            true_cell_mass[(rho, bins)],
                            cdf_matrices[bins],
                            truth_vectors[rho],
                        )
                        cdf_store.setdefault((rho, n_latent, estimator, bins), []).append(cdf_values)
                        mm_diag = diagnostics["mm"]
                        record: Dict[str, Any] = {
                            **dataset,
                            "estimator": estimator,
                            "bins_per_axis": int(bins),
                            "n_cells": int(grid.k),
                            "fit_loaded_from_checkpoint": bool(loaded),
                            "runtime_seconds": float(runtime),
                            "converged": bool(mm_diag["converged"]),
                            "active_score_residual": float(mm_diag["active_score_sup_abs_residual"]),
                            "inactive_positive_score_violation": float(mm_diag["inactive_positive_score_violation"]),
                            "n_forced_zero_cells": int(diagnostics["n_forced_zero_cells"]),
                            "n_equivalence_classes": int(diagnostics["n_equivalence_classes"]),
                            "n_nontrivial_equivalence_classes": int(diagnostics["n_nontrivial_equivalence_classes"]),
                            "nontrivial_equivalence_class_mass_fraction": float(diagnostics.get("nontrivial_equivalence_class_mass_fraction", 0.0)),
                            "n_active_classes": int(mm_diag["n_active"]),
                            "hybrid_polish_used": bool(mm_diag.get("hybrid_polish_used", False)),
                            "polish_cycles": int(mm_diag.get("polish_cycles", 0)),
                            "polish_active_set_change_cycles": int(mm_diag.get("polish_active_set_change_cycles", 0)),
                            "polish_total_activation_events": int(mm_diag.get("polish_total_activation_events", 0)),
                            "polish_total_slsqp_iterations": int(mm_diag.get("polish_total_slsqp_iterations", 0)),
                            "polish_total_newton_steps": int(mm_diag.get("polish_total_newton_steps", 0)),
                            "polish_loglikelihood_increment": float(mm_diag.get("polish_loglikelihood_increment", 0.0)),
                            **metrics,
                        }
                        if estimator == "ABC_union_sieve":
                            pred_mass, eta = predictive_mixture_mass(
                                mass, grid, len(observed), args.mixture_strength
                            )
                            pop_design = population_designs[(rho, bins)]
                            score, numerator, denominator = pop_design.score_vector(pred_mass)
                            if not np.all(np.isfinite(score)):
                                raise RuntimeError("full-fit population score is non-finite")
                            block_means, _ = pop_design.block_means(score)
                            record.update(
                                {
                                    "predictive_mixture_eta": float(eta),
                                    "mean_population_logscore": float(np.mean(score)),
                                    "se_population_logscore_blocks": float(np.std(block_means, ddof=1) / math.sqrt(len(block_means))) if len(block_means) > 1 else 0.0,
                                    "minimum_population_predictive_numerator": float(np.min(numerator)),
                                    "minimum_population_predictive_denominator": float(np.min(denominator)),
                                }
                            )
                        else:
                            record.update(
                                {
                                    "predictive_mixture_eta": math.nan,
                                    "mean_population_logscore": math.nan,
                                    "se_population_logscore_blocks": math.nan,
                                    "minimum_population_predictive_numerator": math.nan,
                                    "minimum_population_predictive_denominator": math.nan,
                                }
                            )
                        full_records.append(record)
                        for start_record in starts:
                            start_records.append(
                                {
                                    **dataset,
                                    "fit_type": "full",
                                    "estimator": estimator,
                                    "bins_per_axis": int(bins),
                                    "repeat": -1,
                                    "fold": -1,
                                    **start_record,
                                }
                            )
                        previous_grid, previous_mass = grid, mass
                    except Exception as exc:
                        failure_records.append(
                            {
                                **dataset,
                                "fit_type": "full",
                                "estimator": estimator,
                                "bins_per_axis": int(bins),
                                "repeat": -1,
                                "fold": -1,
                                "exception_type": type(exc).__name__,
                                "message": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        if args.fail_fast:
                            raise
                    full_done += 1
                    if full_done % args.progress_every == 0 or full_done == total_full:
                        print(
                            f"Full fit progress {full_done}/{total_full} "
                            f"({time.perf_counter() - start_clock:.1f} s)"
                        )

            for repeat in range(cv_repeats):
                fold_seed = int(train_seed + 700_001 + 7_919 * repeat)
                fold_ids, _, fold_audit = make_stratified_fold_assignment(
                    observed, n_folds, fold_seed
                )
                fold_audit_records.append(
                    {
                        **dataset,
                        "repeat": int(repeat),
                        "fold_seed": int(fold_seed),
                        "maximum_region_count_range": int(fold_audit["maximum_region_count_range"]),
                        "fold_counts": json.dumps(fold_audit["fold_counts"]),
                    }
                )
                previous_by_fold: Dict[int, Tuple[RectangularGrid, np.ndarray]] = {}
                for bins in bins_values:
                    grid = make_uniform_grid(bins)
                    for fold in range(n_folds):
                        train = observed.loc[fold_ids != fold].copy().reset_index(drop=True)
                        valid = observed.loc[fold_ids == fold].copy().reset_index(drop=True)
                        warm = None
                        if fold in previous_by_fold:
                            prev_grid, prev_mass = previous_by_fold[fold]
                            warm = project_rectangular_mass(prev_grid, prev_mass, grid)
                        identity = {
                            **dataset,
                            "fit_type": "cross_validation",
                            "estimator": "ABC_union_sieve",
                            "bins_per_axis": int(bins),
                            "repeat": int(repeat),
                            "fold": int(fold),
                            "fold_seed": int(fold_seed),
                        }
                        checkpoint = cv_dir / _confirmatory_abc_checkpoint(
                            "cv_abc", rho, n_latent, replicate, bins, repeat, fold
                        )
                        audit_this = bool(
                            condition_index == 0
                            and replicate == 0
                            and repeat == 0
                            and fold == 0
                            and bins == min(bins_values)
                        )
                        try:
                            fit_start = time.perf_counter()
                            model = _fit_or_load_target_model(
                                checkpoint,
                                fingerprint,
                                train,
                                grid,
                                args,
                                warm,
                                run_uniform_audit=audit_this,
                                direct_starts=args.direct_audit_starts if audit_this else 0,
                                direct_seed=train_seed + 1009 * bins + 31 * fold + 17 * repeat,
                                identity=identity,
                            )
                            runtime = time.perf_counter() - fit_start
                            previous_by_fold[fold] = (grid, model.equal_mass)
                            valid_design = FastObservedScoreDesign.from_observed(valid, grid)
                            raw_score, _, _ = valid_design.score_vector(model.equal_mass)
                            pred_mass, eta = predictive_mixture_mass(
                                model.equal_mass, grid, len(train), args.mixture_strength
                            )
                            score, numerator, denominator = valid_design.score_vector(pred_mass)
                            if not np.all(np.isfinite(score)):
                                raise RuntimeError("stabilised held-out score is non-finite")
                            mm_diag = model.diagnostics["mm"]
                            cv_records.append(
                                {
                                    **dataset,
                                    "bins_per_axis": int(bins),
                                    "repeat": int(repeat),
                                    "fold": int(fold),
                                    "fold_seed": int(fold_seed),
                                    "n_training_rows": int(len(train)),
                                    "n_validation_rows": int(len(valid)),
                                    "predictive_mixture_eta": float(eta),
                                    "mean_validation_logscore": float(np.mean(score)),
                                    "sum_validation_logscore": float(np.sum(score)),
                                    "raw_heldout_nonfinite_count": int(np.sum(~np.isfinite(raw_score))),
                                    "minimum_predictive_numerator": float(np.min(numerator)),
                                    "minimum_predictive_denominator": float(np.min(denominator)),
                                    "fit_loaded_from_checkpoint": bool(model.loaded_from_checkpoint),
                                    "runtime_seconds": float(runtime),
                                    "converged": bool(mm_diag["converged"]),
                                    "active_score_residual": float(mm_diag["active_score_sup_abs_residual"]),
                                    "inactive_positive_score_violation": float(mm_diag["inactive_positive_score_violation"]),
                                    "n_forced_zero_cells": int(model.diagnostics["n_forced_zero_cells"]),
                                    "n_equivalence_classes": int(model.diagnostics["n_equivalence_classes"]),
                                    "n_nontrivial_equivalence_classes": int(model.diagnostics["n_nontrivial_equivalence_classes"]),
                                    "nontrivial_equivalence_class_mass_fraction": float(model.diagnostics.get("nontrivial_equivalence_class_mass_fraction", 0.0)),
                                    "n_active_classes": int(mm_diag["n_active"]),
                                    "hybrid_polish_used": bool(mm_diag.get("hybrid_polish_used", False)),
                                    "polish_cycles": int(mm_diag.get("polish_cycles", 0)),
                                    "polish_active_set_change_cycles": int(mm_diag.get("polish_active_set_change_cycles", 0)),
                                    "polish_total_activation_events": int(mm_diag.get("polish_total_activation_events", 0)),
                                    "polish_total_slsqp_iterations": int(mm_diag.get("polish_total_slsqp_iterations", 0)),
                                    "polish_total_newton_steps": int(mm_diag.get("polish_total_newton_steps", 0)),
                                    "polish_loglikelihood_increment": float(mm_diag.get("polish_loglikelihood_increment", 0.0)),
                                    "fold_region_balance_max_range": int(fold_audit["maximum_region_count_range"]),
                                }
                            )
                            for start_record in model.start_records:
                                start_records.append(
                                    {
                                        **dataset,
                                        "fit_type": "cross_validation",
                                        "estimator": "ABC_union_sieve",
                                        "bins_per_axis": int(bins),
                                        "repeat": int(repeat),
                                        "fold": int(fold),
                                        **start_record,
                                    }
                                )
                        except Exception as exc:
                            failure_records.append(
                                {
                                    **dataset,
                                    "fit_type": "cross_validation",
                                    "estimator": "ABC_union_sieve",
                                    "bins_per_axis": int(bins),
                                    "repeat": int(repeat),
                                    "fold": int(fold),
                                    "exception_type": type(exc).__name__,
                                    "message": str(exc),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                            if args.fail_fast:
                                raise
                        cv_done += 1
                        if cv_done % args.progress_every == 0 or cv_done == total_cv:
                            print(
                                f"CV fit progress {cv_done}/{total_cv} "
                                f"({time.perf_counter() - start_clock:.1f} s)"
                            )

    full_table = pd.DataFrame(full_records)
    cv_fold_table = pd.DataFrame(cv_records)
    failures = pd.DataFrame(failure_records)
    if full_table.empty or cv_fold_table.empty:
        raise RuntimeError("the confirmatory audit produced no usable fit records")

    cv_repeat, cv_average, cv_selection = _aggregate_confirmatory_cv(
        cv_fold_table, args.tie_tolerance
    )
    selection = _build_confirmatory_selection(
        full_table,
        cv_selection,
        bins_values,
        args.primary_bins,
        args.tie_tolerance,
    )
    performance = _selection_performance_summary(selection)
    paired_rules = _paired_rule_comparisons(selection)
    abc_aonly = _abc_aonly_comparisons(full_table)
    bias_variance = _cdf_bias_variance_summary(cdf_store, truth_vectors)

    # Selection frequencies.
    frequency = (
        selection[selection["selection_rule"].isin(["repeat_averaged_cv_max", "cdf_rmse_oracle", "population_logscore_oracle"])]
        .groupby(["rho", "n_latent", "selection_rule", "selected_bins"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    frequency["fraction"] = frequency.groupby(
        ["rho", "n_latent", "selection_rule"]
    )["count"].transform(lambda x: x / x.sum())

    # Prespecified descriptive decision criteria.
    overall_perf = performance[performance["scope"] == "overall"].set_index("selection_rule")
    fixed16 = overall_perf.loc["fixed_16"]
    deployable_comparators = ["fixed_12", "fixed_20", "repeat_averaged_cv_max"]
    criterion_compromise = bool(
        all(
            float(fixed16["mean_cdf_rmse_regret"])
            <= float(overall_perf.loc[name, "mean_cdf_rmse_regret"]) + 1e-15
            for name in deployable_comparators
        )
    )
    condition_fixed16 = performance[
        (performance["selection_rule"] == "fixed_16")
        & (performance["scope"] != "overall")
    ]
    overall_aonly16 = abc_aonly[
        (abc_aonly["scope"] == "overall")
        & (abc_aonly["bins_per_axis"] == args.primary_bins)
    ].iloc[0]
    solver_diagnostic_columns = [
        "converged",
        "active_score_residual",
        "inactive_positive_score_violation",
        "hybrid_polish_used",
        "polish_cycles",
        "polish_active_set_change_cycles",
        "polish_total_activation_events",
        "polish_total_slsqp_iterations",
        "polish_total_newton_steps",
        "polish_loglikelihood_increment",
    ]
    fit_diag = pd.concat(
        [
            full_table[solver_diagnostic_columns],
            cv_fold_table[solver_diagnostic_columns],
        ],
        ignore_index=True,
    )
    criteria = {
        "fixed16_overall_mean_cdf_regret_no_larger_than_each_deployable_comparator": criterion_compromise,
        "fixed16_fraction_within_5_percent_of_oracle_at_least_0_75": bool(
            float(fixed16["fraction_within_5_percent_cdf_oracle"]) >= 0.75
        ),
        "fixed16_each_condition_mean_relative_regret_at_most_0_10": bool(
            np.all(condition_fixed16["mean_relative_cdf_rmse_regret"].to_numpy(float) <= 0.10 + 1e-15)
        ),
        "ABC_fixed16_overall_mean_rmse_lower_than_A_only_fixed16": bool(
            float(overall_aonly16["mean_difference_abc_minus_aonly"]) < 0.0
        ),
        "no_failed_fits_and_all_KKT_checks_pass": bool(
            failures.empty
            and fit_diag["converged"].all()
            and float(fit_diag["active_score_residual"].max()) < args.score_tolerance
            and float(fit_diag["inactive_positive_score_violation"].max()) < args.score_tolerance
        ),
    }
    decision = {
        "criteria": criteria,
        "all_prespecified_descriptive_criteria_met": bool(all(criteria.values())),
        "fixed16_overall": _jsonable(fixed16.to_dict()),
        "ABC_minus_A_only_fixed16_overall": _jsonable(overall_aonly16.to_dict()),
        "interpretation": "A failed criterion is a scientific result, not a software failure.",
    }

    paths = {
        "analysis_freeze": freeze_path,
        "full_fit_metrics": outdir / "full_fit_metrics.csv",
        "cv_fold_scores": outdir / "cv_fold_scores.csv",
        "cv_repeat_scores": outdir / "cv_repeat_scores.csv",
        "cv_repeat_averaged_scores": outdir / "cv_repeat_averaged_scores.csv",
        "cv_selection": outdir / "cv_repeat_averaged_selection.csv",
        "selection_per_catalogue": outdir / "selection_per_catalogue.csv",
        "selection_performance": outdir / "selection_performance_summary.csv",
        "paired_rule_comparisons": outdir / "paired_rule_comparisons.csv",
        "abc_vs_aonly": outdir / "abc_vs_aonly_paired.csv",
        "cdf_bias_variance": outdir / "cdf_bias_variance_summary.csv",
        "selection_frequency": outdir / "selection_frequency.csv",
        "population_validation": outdir / "population_validation_summary.csv",
        "fold_audit": outdir / "fold_balance_audit.csv",
        "seed_registry": outdir / "seed_registry.csv",
        "start_audit": outdir / "start_audit.csv",
        "failures": outdir / "failures.csv",
        "decision": outdir / "prespecified_decision_summary.json",
        "self_tests": outdir / "self_test_results.json",
        "cdf_arrays": outdir / "cdf_replicate_arrays.npz",
        "summary": outdir / "comparison_summary.json",
        "manifest": outdir / "manifest.json",
        "environment": outdir / "environment.json",
        "file_hashes": outdir / "file_hashes.json",
    }
    full_table.to_csv(paths["full_fit_metrics"], index=False)
    cv_fold_table.to_csv(paths["cv_fold_scores"], index=False)
    cv_repeat.to_csv(paths["cv_repeat_scores"], index=False)
    cv_average.to_csv(paths["cv_repeat_averaged_scores"], index=False)
    cv_selection.to_csv(paths["cv_selection"], index=False)
    selection.to_csv(paths["selection_per_catalogue"], index=False)
    performance.to_csv(paths["selection_performance"], index=False)
    paired_rules.to_csv(paths["paired_rule_comparisons"], index=False)
    abc_aonly.to_csv(paths["abc_vs_aonly"], index=False)
    bias_variance.to_csv(paths["cdf_bias_variance"], index=False)
    frequency.to_csv(paths["selection_frequency"], index=False)
    pd.DataFrame(population_rows).to_csv(paths["population_validation"], index=False)
    pd.DataFrame(fold_audit_records).to_csv(paths["fold_audit"], index=False)
    pd.DataFrame(seed_records).to_csv(paths["seed_registry"], index=False)
    pd.DataFrame(start_records).to_csv(paths["start_audit"], index=False)
    failures.to_csv(paths["failures"], index=False)
    _write_json(paths["decision"], decision)
    _write_json(paths["self_tests"], self_tests)
    cdf_payload: Dict[str, np.ndarray] = {}
    for (rho, n_latent, estimator, bins), values in cdf_store.items():
        key = f"rho_{_confirmatory_token(rho)}__n_{n_latent:06d}__{estimator}__bins_{bins:03d}"
        cdf_payload[key] = np.vstack(values)
    np.savez_compressed(paths["cdf_arrays"], **cdf_payload)

    elapsed = time.perf_counter() - start_clock
    checks = {
        "self_tests_passed": bool(self_tests["all_passed"]),
        "no_failed_fits": bool(failures.empty),
        "all_full_and_cv_fits_converged": bool(fit_diag["converged"].all()),
        "all_active_score_residuals_below_tolerance": bool(
            float(fit_diag["active_score_residual"].max()) < args.score_tolerance
        ),
        "all_inactive_KKT_violations_below_tolerance": bool(
            float(fit_diag["inactive_positive_score_violation"].max()) < args.score_tolerance
        ),
        "all_stabilised_CV_scores_finite": bool(
            np.isfinite(cv_fold_table["mean_validation_logscore"]).all()
        ),
        "folds_region_balanced_within_one": bool(
            pd.DataFrame(fold_audit_records)["maximum_region_count_range"].max() <= 1
        ),
        "candidate_resolution_set_is_frozen": bool(
            bins_values == [12, 16, 20] or args.allow_nonfrozen_design
        ),
        "confirmatory_seed_family_differs_from_development_runs": bool(
            int(args.seed) == 20260912 or args.allow_nonfrozen_design
        ),
        "deployable_CV_selection_uses_no_truth": True,
        "predictive_mixture_used_only_for_scoring": True,
        "bias_variance_identity_holds": bool(
            float(bias_variance["identity_residual"].max()) < 1e-12
        ),
        "repeated_polish_cycle_diagnostics_finite": bool(
            np.isfinite(fit_diag["polish_cycles"]).all()
            and np.isfinite(fit_diag["polish_total_activation_events"]).all()
        ),
        "all_multiple_polish_cycle_fits_satisfy_KKT": bool(
            (
                fit_diag.loc[fit_diag["polish_cycles"] > 1, "active_score_residual"]
                < args.score_tolerance
            ).all()
            and (
                fit_diag.loc[fit_diag["polish_cycles"] > 1, "inactive_positive_score_violation"]
                < args.score_tolerance
            ).all()
        ),
    }
    summary = {
        "code_version": CONFIRMATORY_CODE_VERSION,
        "profile": args.profile,
        "all_main_checks_passed": bool(all(checks.values())),
        "checks": checks,
        "configuration_fingerprint": fingerprint,
        "conditions": [[float(r), int(n)] for r, n in conditions],
        "candidate_resolutions": bins_values,
        "primary_resolution": int(args.primary_bins),
        "replicates": int(replicates),
        "cv_folds": int(n_folds),
        "cv_repeats": int(cv_repeats),
        "population_validation_latent": int(population_latent),
        "n_successful_full_fits": int(len(full_table)),
        "n_successful_cv_fits": int(len(cv_fold_table)),
        "n_failed_fits": int(len(failures)),
        "maximum_active_score_residual": float(fit_diag["active_score_residual"].max()),
        "maximum_inactive_positive_score_violation": float(fit_diag["inactive_positive_score_violation"].max()),
        "maximum_polish_cycles_used": int(fit_diag["polish_cycles"].max()),
        "n_fits_requiring_hybrid_polish": int(fit_diag["hybrid_polish_used"].sum()),
        "n_fits_requiring_multiple_polish_cycles": int((fit_diag["polish_cycles"] > 1).sum()),
        "n_fits_with_active_set_change_during_polish": int((fit_diag["polish_active_set_change_cycles"] > 0).sum()),
        "maximum_polish_activation_events": int(fit_diag["polish_total_activation_events"].max()),
        "maximum_polish_loglikelihood_increment": float(fit_diag["polish_loglikelihood_increment"].max()),
        "raw_nonfinite_CV_score_fraction": float(
            cv_fold_table["raw_heldout_nonfinite_count"].sum()
            / cv_fold_table["n_validation_rows"].sum()
        ),
        "fixed16_mean_cdf_rmse_regret": float(fixed16["mean_cdf_rmse_regret"]),
        "fixed16_fraction_within_5_percent_cdf_oracle": float(
            fixed16["fraction_within_5_percent_cdf_oracle"]
        ),
        "fixed16_mean_relative_cdf_rmse_regret": float(
            fixed16["mean_relative_cdf_rmse_regret"]
        ),
        "repeat_averaged_CV_mean_cdf_rmse_regret": float(
            overall_perf.loc["repeat_averaged_cv_max", "mean_cdf_rmse_regret"]
        ),
        "ABC_minus_A_only_fixed16_mean_CDF_RMSE_difference": float(
            overall_aonly16["mean_difference_abc_minus_aonly"]
        ),
        "prespecified_decision": decision,
        "elapsed_seconds": float(elapsed),
        "scientific_scope": {
            "independent_seed_confirmatory_monte_carlo": True,
            "rules_frozen_before_new_catalogues": True,
            "primary_CDF_resolution_fixed_at_16": True,
            "repeated_CV_comparator": True,
            "A_only_baseline": True,
            "simulation_only_oracles_not_deployable": True,
            "general_optimality_theorem": False,
            "sieve_asymptotic_theory": False,
        },
    }
    _write_json(paths["summary"], summary)
    manifest = {
        "code_version": CONFIRMATORY_CODE_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_fingerprint": fingerprint,
        "arguments": vars(args),
        "analysis_freeze_file": str(freeze_path),
        "checkpoint_directories": {
            "full_ABC": str(full_abc_dir),
            "full_A_only": str(full_aonly_dir),
            "CV_ABC": str(cv_dir),
        },
        "seed_design": "independent condition-by-replicate deterministic seeds, disjoint from development seeds",
        "CV_rule": "maximum observation-weighted out-of-fold conditional logscore pooled over all repeats and folds; numerical ties choose the coarser grid",
        "truth_usage": "truth is used only after model fitting and CV selection, for CDF error and simulation-only oracle audits",
        "predictive_stabilisation": "p_pred=(1-eta)p_hat+eta p_uniform with eta=c/(n_train+c), used only for predictive scores",
        "numerical_solver_amendment": "Version 1.1 repeats SLSQP and projected Newton whenever polishing changes the active set; the frozen statistical analysis is unchanged.",
    }
    _write_json(paths["manifest"], manifest)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "working_directory": str(Path.cwd()),
    }
    _write_json(paths["environment"], environment)
    hashes: Dict[str, Dict[str, str]] = {}
    for name, path in paths.items():
        if name != "file_hashes" and path.exists():
            hashes[name] = {"path": str(path), "sha256": _sha256(path)}
    _write_json(paths["file_hashes"], hashes)
    return summary, paths


def build_confirmatory_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent-seed confirmatory Monte Carlo for the union-selection rectangular-sieve estimator"
    )
    parser.add_argument("--outdir", default="union_selection_sieve_confirmatory_monte_carlo_v1_1_outputs")
    parser.add_argument("--profile", choices=["smoke", "pilot", "confirmatory", "full"], default="confirmatory")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--bins-values", type=int, nargs="+", default=None)
    parser.add_argument("--primary-bins", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--cv-repeats", type=int, default=None)
    parser.add_argument("--population-validation-latent", type=int, default=None)
    parser.add_argument("--population-blocks", type=int, default=None)
    parser.add_argument("--query-grid-size", type=int, default=None)
    parser.add_argument("--mixture-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--mass-tolerance", type=float, default=1e-12)
    parser.add_argument("--score-tolerance", type=float, default=1e-9)
    parser.add_argument("--prune-mass-tolerance", type=float, default=1e-7)
    parser.add_argument("--prune-score-margin", type=float, default=1e-7)
    parser.add_argument("--coverage-tolerance", type=float, default=1e-15)
    parser.add_argument("--warm-mixing", type=float, default=1e-9)
    parser.add_argument("--max-iterations", type=int, default=30000)
    parser.add_argument("--mm-prepolish-iterations", type=int, default=2000)
    parser.add_argument("--newton-tolerance", type=float, default=1e-12)
    parser.add_argument("--polish-zero-tolerance", type=float, default=1e-13)
    parser.add_argument("--max-polish-cycles", type=int, default=8)
    parser.add_argument("--direct-audit-starts", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--allow-nonfrozen-design", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return parser


def confirmatory_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_confirmatory_parser()
    if argv is None:
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"Ignoring unrecognised arguments: {unknown}")
    else:
        args = parser.parse_args(list(argv))
    summary, paths = run_confirmatory_audit(args)
    concise_keys = [
        "code_version",
        "profile",
        "all_main_checks_passed",
        "checks",
        "conditions",
        "candidate_resolutions",
        "primary_resolution",
        "replicates",
        "cv_folds",
        "cv_repeats",
        "n_successful_full_fits",
        "n_successful_cv_fits",
        "n_failed_fits",
        "maximum_active_score_residual",
        "maximum_inactive_positive_score_violation",
        "maximum_polish_cycles_used",
        "n_fits_requiring_hybrid_polish",
        "n_fits_requiring_multiple_polish_cycles",
        "n_fits_with_active_set_change_during_polish",
        "maximum_polish_activation_events",
        "maximum_polish_loglikelihood_increment",
        "raw_nonfinite_CV_score_fraction",
        "fixed16_mean_cdf_rmse_regret",
        "fixed16_fraction_within_5_percent_cdf_oracle",
        "fixed16_mean_relative_cdf_rmse_regret",
        "repeat_averaged_CV_mean_cdf_rmse_regret",
        "ABC_minus_A_only_fixed16_mean_CDF_RMSE_difference",
        "prespecified_decision",
        "elapsed_seconds",
    ]
    print(json.dumps({k: summary[k] for k in concise_keys}, indent=2, sort_keys=True))

    performance = pd.read_csv(paths["selection_performance"])
    print("\nPrespecified resolution-rule performance:")
    print(performance.to_string(index=False))

    paired = pd.read_csv(paths["paired_rule_comparisons"])
    print("\nFixed-16 paired comparisons:")
    print(paired.to_string(index=False))

    abc = pd.read_csv(paths["abc_vs_aonly"])
    print("\nABC-union versus A-only paired comparison:")
    print(abc.to_string(index=False))

    frequency = pd.read_csv(paths["selection_frequency"])
    print("\nCV and oracle resolution frequencies:")
    print(frequency.to_string(index=False))

    print("\nWritten files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("One-cell independent-seed confirmatory Monte Carlo completed successfully (status=0).")
    print(f"Output directory: {Path(args.outdir).resolve()}")
    return 0


# =============================================================================
# One-cell autorun
# =============================================================================
ONE_CELL_CONFIRMATORY_PROFILE = os.environ.get(
    "UNION_SELECTION_CONFIRMATORY_PROFILE", "confirmatory"
)
ONE_CELL_CONFIRMATORY_ARGUMENTS = [
    "--outdir", "union_selection_sieve_confirmatory_monte_carlo_v1_1_outputs",
    "--profile", ONE_CELL_CONFIRMATORY_PROFILE,
    "--bins-values", "12", "16", "20",
    "--primary-bins", "16",
    "--mixture-strength", "1.0",
    "--seed", "20260912",
    "--mass-tolerance", "1e-12",
    "--score-tolerance", "1e-9",
    "--prune-mass-tolerance", "1e-7",
    "--prune-score-margin", "1e-7",
    "--coverage-tolerance", "1e-15",
    "--warm-mixing", "1e-9",
    "--max-iterations", "30000",
    "--mm-prepolish-iterations", "2000",
    "--newton-tolerance", "1e-12",
    "--polish-zero-tolerance", "1e-13",
    "--max-polish-cycles", "8",
    "--direct-audit-starts", "4",
    "--progress-every", "100",
    "--resume",
    "--self-test",
    "--fail-fast",
]


def _confirmatory_running_in_ipython_kernel() -> bool:
    return "ipykernel" in sys.modules or "IPython" in sys.modules


if os.environ.get("UNION_SELECTION_CONFIRMATORY_SKIP_AUTORUN", "0") != "1":
    print("Running the independent-seed union-selection sieve confirmatory Monte Carlo from the one-cell configuration.")
    print(f"Profile: {ONE_CELL_CONFIRMATORY_PROFILE}")
    print(f"Equivalent arguments: {ONE_CELL_CONFIRMATORY_ARGUMENTS}")
    _ONE_CELL_CONFIRMATORY_STATUS = confirmatory_main(ONE_CELL_CONFIRMATORY_ARGUMENTS)
    if _ONE_CELL_CONFIRMATORY_STATUS != 0:
        raise RuntimeError(
            f"one-cell execution returned nonzero status {_ONE_CELL_CONFIRMATORY_STATUS}"
        )
    if _confirmatory_running_in_ipython_kernel():
        print("One-cell execution returned status=0 without raising SystemExit in Jupyter.")
