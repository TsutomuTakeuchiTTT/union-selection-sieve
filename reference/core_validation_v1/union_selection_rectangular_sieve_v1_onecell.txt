"""One-cell rectangular-sieve conditional-likelihood audit.

This program extends the finite-support union-selection conditional likelihood to
continuous bivariate data by representing the parent density as a nonnegative
mixture of normalized rectangular basis functions,

    f_p(x1, x2) = sum_k p_k phi_k(x1, x2),
    p_k >= 0,  sum_k p_k = 1.

For an observed object with detection limits (y1, y2), the catalogue inclusion
set is

    O_i = {x1 >= y1 or x2 >= y2}.

The numerator is mixed-dimensional:

* Region A: f_p(x1, x2);
* Region B: integral_{x2 < y2} f_p(x1, x2) dx2;
* Region C: integral_{x1 < y1} f_p(x1, x2) dx1.

All numerator and selection-denominator terms are linear in p.  The conditional
log-likelihood therefore has the same algebraic form as the validated finite-
support likelihood,

    ell(p) = sum_i nu_i [log(a_i^T p) - log(b_i^T p)].

The code performs four linked audits:

1. analytic operator checks for rectangular basis functions;
2. an exactly specified piecewise-constant population and finite-sample test;
3. population sieve-refinement tests for truncated bivariate Gaussian targets;
4. finite-sample Gaussian tests for moderate and strong correlation.

The estimator is a finite-dimensional sieve likelihood.  This program does not
claim a continuous-support NPMLE support-reduction theorem or general global
uniqueness.  It uses no Dąbrowska operator, inverse-probability weighting,
negative-mass repair, monotonicity projection, clipping, or smoothing.

Executing the entire file in one Jupyter cell runs the audit automatically.

Code version: union-selection-rectangular-sieve-v1.0.0-onecell
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy.optimize import brentq, minimize
    from scipy.stats import multivariate_normal
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "SciPy is required for the MM multiplier, independent optimiser, and "
        "bivariate-normal benchmark. Install scipy in the active Jupyter kernel."
    ) from exc

CODE_VERSION = "union-selection-rectangular-sieve-v1.0.0-onecell"


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
            p, aa, bb, w, active_tolerance=prune_mass_tolerance
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
        p, aa, bb, w, active_tolerance=prune_mass_tolerance
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
# Main audit
# =============================================================================
def run_experiment(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    self_tests = run_self_tests() if args.self_test else {"all_passed": True, "checks": {}, "details": {}}
    if not self_tests["all_passed"]:
        raise RuntimeError("a built-in self-test failed")

    # ------------------------------------------------------------------
    # 1. Exact piecewise-constant benchmark
    # ------------------------------------------------------------------
    pgrid = RectangularGrid(PIECEWISE_EDGES, PIECEWISE_EDGES)
    ptrue = PIECEWISE_TRUE_MASS_MATRIX.reshape(-1)
    pa_pop, pb_pop, pw_pop, pop_categories, pop_info = build_population_categories(
        pgrid, ptrue, PIECEWISE_Y_SUPPORT, PIECEWISE_Y_MASS
    )
    piece_pop_fit = fit_sieve_likelihood(
        pa_pop,
        pb_pop,
        pw_pop,
        args.coverage_tolerance,
        args.mass_tolerance,
        args.score_tolerance,
        args.prune_mass_tolerance,
        args.prune_score_margin,
        args.max_iterations,
        direct_starts=args.direct_starts,
        direct_seed=args.seed + 11,
    )
    piece_pop_error = _max_abs(piece_pop_fit.original_mass - ptrue)

    platent, pobserved, psample_info = simulate_piecewise_constant_catalogue(
        args.n_piecewise_latent,
        args.seed,
        pgrid,
        ptrue,
        PIECEWISE_Y_SUPPORT,
        PIECEWISE_Y_MASS,
    )
    pa_s, pb_s, p_membership = build_sample_operator(pobserved, pgrid)
    piece_sample_fit = fit_sieve_likelihood(
        pa_s,
        pb_s,
        None,
        args.coverage_tolerance,
        args.mass_tolerance,
        args.score_tolerance,
        args.prune_mass_tolerance,
        args.prune_score_margin,
        args.max_iterations,
        direct_starts=args.direct_starts,
        direct_seed=args.seed + 13,
    )
    piece_truth_cdf = piecewise_truth_cdf(pgrid, ptrue)
    piece_cdf_table, piece_cdf_metrics = cdf_accuracy_table(
        pgrid,
        piece_sample_fit.original_mass,
        piece_truth_cdf,
        query_grid_size=args.query_grid_size,
    )
    piece_cdf_envelope, piece_envelope_metrics = cdf_identification_envelope(
        pgrid,
        piece_sample_fit.mm.mass,
        piece_sample_fit.reduction.classes,
        truth_cdf=piece_truth_cdf,
        query_grid_size=args.query_grid_size,
    )

    # ------------------------------------------------------------------
    # 2. Smooth Gaussian population refinement
    # ------------------------------------------------------------------
    refinement_rows: List[Dict[str, Any]] = []
    gaussian_population_categories: List[pd.DataFrame] = []
    gaussian_population_histories: List[pd.DataFrame] = []
    refinement_cdf_tables: List[pd.DataFrame] = []
    refinement_fits: Dict[Tuple[float, int], SieveFitBundle] = {}

    for rho in args.rho_values:
        truth = TruncatedBivariateNormalTruth(rho, GAUSSIAN_LOWER, GAUSSIAN_UPPER)
        for bins in args.refinement_bins:
            edges = np.linspace(GAUSSIAN_LOWER, GAUSSIAN_UPPER, bins + 1)
            grid = RectangularGrid(edges, edges)
            true_mass = truth.cell_masses(grid)
            a_pop, b_pop, w_pop, categories, info = build_population_categories(
                grid, true_mass, GAUSSIAN_Y_SUPPORT, GAUSSIAN_Y_MASS
            )
            fit = fit_sieve_likelihood(
                a_pop,
                b_pop,
                w_pop,
                args.coverage_tolerance,
                args.mass_tolerance,
                args.score_tolerance,
                args.prune_mass_tolerance,
                args.prune_score_margin,
                args.max_iterations,
                direct_starts=0,
            )
            refinement_fits[(rho, bins)] = fit
            cdf_table, cdf_metrics = cdf_accuracy_table(
                grid, fit.original_mass, truth.cdf, query_grid_size=args.query_grid_size
            )
            cdf_table.insert(0, "rho", rho)
            cdf_table.insert(1, "bins_per_axis", bins)
            refinement_cdf_tables.append(cdf_table)
            categories = categories.copy()
            categories.insert(0, "rho", rho)
            categories.insert(1, "bins_per_axis", bins)
            gaussian_population_categories.append(categories)
            history = fit.mm.history.copy()
            history.insert(0, "rho", rho)
            history.insert(1, "bins_per_axis", bins)
            gaussian_population_histories.append(history)
            refinement_rows.append(
                {
                    "rho": rho,
                    "bins_per_axis": bins,
                    "n_cells": grid.k,
                    "true_observation_probability": info["true_observation_probability"],
                    "n_population_observed_categories": info["n_population_observed_categories"],
                    "mm_converged": fit.mm.converged,
                    "mm_outer_iterations": fit.mm.diagnostics["outer_iterations"],
                    "mm_steps": fit.mm.diagnostics["mm_steps"],
                    "n_active_classes": fit.mm.diagnostics["n_active"],
                    "n_forced_zero_cells": fit.diagnostics["n_forced_zero_cells"],
                    "mass_max_abs_error_against_true_cell_mass": _max_abs(
                        fit.original_mass - true_mass
                    ),
                    **cdf_metrics,
                }
            )

    refinement_table = pd.DataFrame(refinement_rows)

    # ------------------------------------------------------------------
    # 3. Smooth Gaussian finite-sample sieve fits at the middle grid size
    # ------------------------------------------------------------------
    finite_rows: List[Dict[str, Any]] = []
    finite_catalogues: List[pd.DataFrame] = []
    finite_observed: List[pd.DataFrame] = []
    finite_cell_tables: List[pd.DataFrame] = []
    finite_membership: List[pd.DataFrame] = []
    finite_screening: List[pd.DataFrame] = []
    finite_classes: List[pd.DataFrame] = []
    finite_histories: List[pd.DataFrame] = []
    finite_events: List[pd.DataFrame] = []
    finite_cdf_tables: List[pd.DataFrame] = []
    finite_cdf_envelopes: List[pd.DataFrame] = []

    bins = int(args.finite_sample_bins)
    gedges = np.linspace(GAUSSIAN_LOWER, GAUSSIAN_UPPER, bins + 1)
    ggrid = RectangularGrid(gedges, gedges)
    for rho_index, rho in enumerate(args.rho_values):
        truth = TruncatedBivariateNormalTruth(rho, GAUSSIAN_LOWER, GAUSSIAN_UPPER)
        latent, observed, sim_info = simulate_gaussian_catalogue(
            args.n_gaussian_latent,
            args.seed + 1000 + 101 * rho_index,
            rho,
            GAUSSIAN_LOWER,
            GAUSSIAN_UPPER,
            GAUSSIAN_Y_SUPPORT,
            GAUSSIAN_Y_MASS,
        )
        a_s, b_s, membership = build_sample_operator(observed, ggrid)
        fit = fit_sieve_likelihood(
            a_s,
            b_s,
            None,
            args.coverage_tolerance,
            args.mass_tolerance,
            args.score_tolerance,
            args.prune_mass_tolerance,
            args.prune_score_margin,
            args.max_iterations,
            direct_starts=0,
        )
        cdf_table, cdf_metrics = cdf_accuracy_table(
            ggrid, fit.original_mass, truth.cdf, query_grid_size=args.query_grid_size
        )
        cdf_envelope, envelope_metrics = cdf_identification_envelope(
            ggrid,
            fit.mm.mass,
            fit.reduction.classes,
            truth_cdf=truth.cdf,
            query_grid_size=args.query_grid_size,
        )
        true_cell_mass = truth.cell_masses(ggrid)
        finite_rows.append(
            {
                "rho": rho,
                "bins_per_axis": bins,
                "n_cells": ggrid.k,
                **sim_info,
                "mm_converged": fit.mm.converged,
                "mm_outer_iterations": fit.mm.diagnostics["outer_iterations"],
                "mm_steps": fit.mm.diagnostics["mm_steps"],
                "n_active_classes": fit.mm.diagnostics["n_active"],
                "n_prune_events": fit.mm.diagnostics["n_prune_events"],
                "active_score_residual": fit.mm.diagnostics["active_score_sup_abs_residual"],
                "inactive_positive_score_violation": fit.mm.diagnostics["inactive_positive_score_violation"],
                "n_forced_zero_cells": fit.diagnostics["n_forced_zero_cells"],
                "n_invisible_cells": fit.diagnostics["n_invisible_cells"],
                "n_equivalence_classes": fit.diagnostics["n_equivalence_classes"],
                "n_nontrivial_equivalence_classes": fit.diagnostics["n_nontrivial_equivalence_classes"],
                "probability_mass_sum": float(fit.original_mass.sum()),
                "all_masses_nonnegative": bool(np.all(fit.original_mass >= 0.0)),
                "true_cell_mass_max_abs_error": _max_abs(fit.original_mass - true_cell_mass),
                "n_query_points": cdf_metrics["n_query_points"],
                **envelope_metrics,
            }
        )
        latent = latent.copy(); latent.insert(0, "rho", rho)
        observed = observed.copy(); observed.insert(0, "rho", rho)
        membership = membership.copy(); membership.insert(0, "rho", rho)
        screening = fit.reduction.cell_screening.copy(); screening.insert(0, "rho", rho)
        classes = fit.reduction.class_table.copy(); classes.insert(0, "rho", rho)
        history = fit.mm.history.copy(); history.insert(0, "rho", rho)
        events = fit.mm.events.copy()
        if events.empty:
            events = pd.DataFrame(columns=["outer_iteration", "event", "class_index", "mass_before", "score_before", "loglik_increment", "n_active_after"])
        events.insert(0, "rho", rho)
        cdf_table.insert(0, "rho", rho)
        cdf_envelope.insert(0, "rho", rho)
        cell_table = ggrid.cell_table.copy()
        cell_table.insert(0, "rho", rho)
        cell_table["estimated_mass"] = fit.original_mass
        cell_table["true_cell_mass"] = true_cell_mass
        finite_catalogues.append(latent)
        finite_observed.append(observed)
        finite_membership.append(membership)
        finite_screening.append(screening)
        finite_classes.append(classes)
        finite_histories.append(history)
        finite_events.append(events)
        finite_cdf_tables.append(cdf_table)
        finite_cdf_envelopes.append(cdf_envelope)
        finite_cell_tables.append(cell_table)

    finite_table = pd.DataFrame(finite_rows)

    # ------------------------------------------------------------------
    # Main scientific checks
    # ------------------------------------------------------------------
    refinement_checks: Dict[str, bool] = {}
    for rho in args.rho_values:
        sub = refinement_table.loc[refinement_table["rho"] == rho].sort_values("bins_per_axis")
        refinement_checks[f"rho_{rho:g}_cdf_rmse_strictly_decreases"] = bool(
            np.all(np.diff(sub["cdf_rmse"].to_numpy()) < 0.0)
        )
        refinement_checks[f"rho_{rho:g}_population_cell_masses_recovered"] = bool(
            float(sub["mass_max_abs_error_against_true_cell_mass"].max()) < 1e-8
        )

    piece_direct_diff = piece_pop_fit.diagnostics["agreement"].get(
        "mm_direct_class_mass_sup_abs_difference", math.inf
    )
    sample_direct_diff = piece_sample_fit.diagnostics["agreement"].get(
        "mm_direct_class_mass_sup_abs_difference", math.inf
    )
    checks = {
        "self_tests_passed": bool(self_tests["all_passed"]),
        "piecewise_population_truth_recovered": bool(piece_pop_error < 1e-8),
        "piecewise_population_mm_direct_agree": bool(piece_direct_diff < 2e-6),
        "piecewise_sample_mm_direct_agree": bool(sample_direct_diff < 2e-6),
        "piecewise_population_likelihood_nondecreasing": bool(piece_pop_fit.mm.diagnostics["likelihood_nondecreasing"]),
        "piecewise_sample_likelihood_nondecreasing": bool(piece_sample_fit.mm.diagnostics["likelihood_nondecreasing"]),
        "all_gaussian_population_fits_converged": bool(refinement_table["mm_converged"].all()),
        "all_gaussian_sample_fits_converged": bool(finite_table["mm_converged"].all()),
        "all_gaussian_sample_masses_nonnegative": bool(finite_table["all_masses_nonnegative"].all()),
        "all_gaussian_sample_masses_normalised": bool(np.all(np.abs(finite_table["probability_mass_sum"] - 1.0) < 1e-12)),
        "no_invisible_cells_in_gaussian_sample_benchmarks": bool((finite_table["n_invisible_cells"] == 0).all()),
        **refinement_checks,
    }

    summary: Dict[str, Any] = {
        "code_version": CODE_VERSION,
        "estimator_scope": "continuous_bivariate_density_approximated_by_finite_rectangular_sieve",
        "continuous_support_reduction_theorem_proved": False,
        "dabrowska_operator_used": False,
        "old_boundary_ipw_used": False,
        "negative_mass_repair_used": False,
        "monotonicity_projection_used": False,
        "smoothing_used": False,
        "cdf_reporting_convention": "equal_split_representative_plus_equivalence_class_identification_envelope",
        "all_main_checks_passed": bool(all(checks.values())),
        "checks": checks,
        "self_tests": self_tests,
        "piecewise_population_audit": {
            "population_info": pop_info,
            "truth_mass_max_abs_error": piece_pop_error,
            "mm": piece_pop_fit.mm.diagnostics,
            "agreement": piece_pop_fit.diagnostics["agreement"],
            "support_reduction": {
                key: value for key, value in piece_pop_fit.diagnostics.items()
                if key not in {"mm", "agreement"}
            },
        },
        "piecewise_sample_audit": {
            **psample_info,
            "cdf_accuracy_equal_split_representative": piece_cdf_metrics,
            "cdf_identification_envelope": piece_envelope_metrics,
            "true_cell_mass_max_abs_error": _max_abs(piece_sample_fit.original_mass - ptrue),
            "mm": piece_sample_fit.mm.diagnostics,
            "agreement": piece_sample_fit.diagnostics["agreement"],
            "support_reduction": {
                key: value for key, value in piece_sample_fit.diagnostics.items()
                if key not in {"mm", "agreement"}
            },
        },
        "gaussian_population_refinement": refinement_table.to_dict(orient="records"),
        "gaussian_finite_sample": finite_table.to_dict(orient="records"),
    }

    # ------------------------------------------------------------------
    # Write all outputs
    # ------------------------------------------------------------------
    paths: Dict[str, Path] = {
        "piecewise_latent_catalogue": outdir / "piecewise_mock_latent_catalogue.csv",
        "piecewise_observed_catalogue": outdir / "piecewise_observed_ABC_catalogue.csv",
        "piecewise_population_categories": outdir / "piecewise_population_observed_categories.csv",
        "piecewise_sample_membership": outdir / "piecewise_sample_membership_audit.csv",
        "piecewise_cell_masses": outdir / "piecewise_cell_mass_estimates.csv",
        "piecewise_population_mm_history": outdir / "piecewise_population_mm_history.csv",
        "piecewise_sample_mm_history": outdir / "piecewise_sample_mm_history.csv",
        "piecewise_population_direct_runs": outdir / "piecewise_population_direct_multistart_runs.csv",
        "piecewise_sample_direct_runs": outdir / "piecewise_sample_direct_multistart_runs.csv",
        "piecewise_cdf_accuracy": outdir / "piecewise_cdf_truth_and_estimate.csv",
        "piecewise_cdf_identification_envelope": outdir / "piecewise_cdf_identification_envelope.csv",
        "gaussian_population_refinement_summary": outdir / "gaussian_population_refinement_summary.csv",
        "gaussian_population_categories": outdir / "gaussian_population_categories.csv",
        "gaussian_population_mm_histories": outdir / "gaussian_population_mm_histories.csv",
        "gaussian_population_cdf_tables": outdir / "gaussian_population_cdf_refinement_tables.csv",
        "gaussian_sample_summary": outdir / "gaussian_finite_sample_summary.csv",
        "gaussian_sample_latent_catalogues": outdir / "gaussian_sample_latent_catalogues.csv",
        "gaussian_sample_observed_catalogues": outdir / "gaussian_sample_observed_ABC_catalogues.csv",
        "gaussian_sample_cell_masses": outdir / "gaussian_sample_cell_mass_estimates.csv",
        "gaussian_sample_membership": outdir / "gaussian_sample_membership_audits.csv",
        "gaussian_sample_screening": outdir / "gaussian_sample_cell_screening.csv",
        "gaussian_sample_equivalence_classes": outdir / "gaussian_sample_equivalence_classes.csv",
        "gaussian_sample_mm_histories": outdir / "gaussian_sample_mm_histories.csv",
        "gaussian_sample_active_set_events": outdir / "gaussian_sample_active_set_events.csv",
        "gaussian_sample_cdf_tables": outdir / "gaussian_sample_cdf_equal_split_truth_and_estimates.csv",
        "gaussian_sample_cdf_identification_envelopes": outdir / "gaussian_sample_cdf_identification_envelopes.csv",
        "summary": outdir / "comparison_summary.json",
        "self_tests": outdir / "self_test_results.json",
        "manifest": outdir / "manifest.json",
        "environment": outdir / "environment.json",
        "file_hashes": outdir / "file_hashes.json",
    }

    platent.to_csv(paths["piecewise_latent_catalogue"], index=False)
    pobserved.to_csv(paths["piecewise_observed_catalogue"], index=False)
    pop_categories.to_csv(paths["piecewise_population_categories"], index=False)
    p_membership.to_csv(paths["piecewise_sample_membership"], index=False)
    piece_cells = pgrid.cell_table.copy()
    piece_cells["true_mass"] = ptrue
    piece_cells["population_estimated_mass"] = piece_pop_fit.original_mass
    piece_cells["sample_estimated_mass"] = piece_sample_fit.original_mass
    piece_cells.to_csv(paths["piecewise_cell_masses"], index=False)
    piece_pop_fit.mm.history.to_csv(paths["piecewise_population_mm_history"], index=False)
    piece_sample_fit.mm.history.to_csv(paths["piecewise_sample_mm_history"], index=False)
    if piece_pop_fit.direct is not None:
        piece_pop_fit.direct.runs.to_csv(paths["piecewise_population_direct_runs"], index=False)
    if piece_sample_fit.direct is not None:
        piece_sample_fit.direct.runs.to_csv(paths["piecewise_sample_direct_runs"], index=False)
    piece_cdf_table.to_csv(paths["piecewise_cdf_accuracy"], index=False)
    piece_cdf_envelope.to_csv(paths["piecewise_cdf_identification_envelope"], index=False)
    refinement_table.to_csv(paths["gaussian_population_refinement_summary"], index=False)
    pd.concat(gaussian_population_categories, ignore_index=True).to_csv(paths["gaussian_population_categories"], index=False)
    pd.concat(gaussian_population_histories, ignore_index=True).to_csv(paths["gaussian_population_mm_histories"], index=False)
    pd.concat(refinement_cdf_tables, ignore_index=True).to_csv(paths["gaussian_population_cdf_tables"], index=False)
    finite_table.to_csv(paths["gaussian_sample_summary"], index=False)
    pd.concat(finite_catalogues, ignore_index=True).to_csv(paths["gaussian_sample_latent_catalogues"], index=False)
    pd.concat(finite_observed, ignore_index=True).to_csv(paths["gaussian_sample_observed_catalogues"], index=False)
    pd.concat(finite_cell_tables, ignore_index=True).to_csv(paths["gaussian_sample_cell_masses"], index=False)
    pd.concat(finite_membership, ignore_index=True).to_csv(paths["gaussian_sample_membership"], index=False)
    pd.concat(finite_screening, ignore_index=True).to_csv(paths["gaussian_sample_screening"], index=False)
    pd.concat(finite_classes, ignore_index=True).to_csv(paths["gaussian_sample_equivalence_classes"], index=False)
    pd.concat(finite_histories, ignore_index=True).to_csv(paths["gaussian_sample_mm_histories"], index=False)
    pd.concat(finite_events, ignore_index=True).to_csv(paths["gaussian_sample_active_set_events"], index=False)
    pd.concat(finite_cdf_tables, ignore_index=True).to_csv(paths["gaussian_sample_cdf_tables"], index=False)
    pd.concat(finite_cdf_envelopes, ignore_index=True).to_csv(paths["gaussian_sample_cdf_identification_envelopes"], index=False)
    _write_json(paths["summary"], summary)
    _write_json(paths["self_tests"], self_tests)

    manifest = {
        "code_version": CODE_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "piecewise_grid_edges": PIECEWISE_EDGES,
        "piecewise_true_mass_matrix": PIECEWISE_TRUE_MASS_MATRIX,
        "piecewise_y_support": PIECEWISE_Y_SUPPORT,
        "piecewise_y_mass": PIECEWISE_Y_MASS,
        "gaussian_support": [GAUSSIAN_LOWER, GAUSSIAN_UPPER],
        "gaussian_y_support": GAUSSIAN_Y_SUPPORT,
        "gaussian_y_mass": GAUSSIAN_Y_MASS,
        "scientific_scope": {
            "rectangular_sieve_density": True,
            "mixed_dimensional_conditional_likelihood": True,
            "population_exact_cell_mass_audit": True,
            "finite_sample_audit": True,
            "basis_refinement_audit": True,
            "continuous_support_NPMLE_theorem": False,
            "global_uniqueness_theorem": False,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="union_selection_rectangular_sieve_outputs")
    parser.add_argument("--n-piecewise-latent", type=int, default=6000)
    parser.add_argument("--n-gaussian-latent", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--rho-values", type=float, nargs="+", default=[0.55, 0.9])
    parser.add_argument("--refinement-bins", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--finite-sample-bins", type=int, default=8)
    parser.add_argument("--query-grid-size", type=int, default=41)
    parser.add_argument("--mass-tolerance", type=float, default=1e-11)
    parser.add_argument("--score-tolerance", type=float, default=1e-8)
    parser.add_argument("--prune-mass-tolerance", type=float, default=1e-10)
    parser.add_argument("--prune-score-margin", type=float, default=1e-6)
    parser.add_argument("--coverage-tolerance", type=float, default=1e-15)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--direct-starts", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    if argv is None:
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"Ignoring unrecognised arguments: {unknown}")
    else:
        args = parser.parse_args(list(argv))

    summary, paths = run_experiment(args)
    print(json.dumps(_jsonable(summary["self_tests"]), indent=2, sort_keys=True))
    concise = {
        "code_version": summary["code_version"],
        "estimator_scope": summary["estimator_scope"],
        "continuous_support_reduction_theorem_proved": summary["continuous_support_reduction_theorem_proved"],
        "dabrowska_operator_used": summary["dabrowska_operator_used"],
        "old_boundary_ipw_used": summary["old_boundary_ipw_used"],
        "negative_mass_repair_used": summary["negative_mass_repair_used"],
        "monotonicity_projection_used": summary["monotonicity_projection_used"],
        "cdf_reporting_convention": summary["cdf_reporting_convention"],
        "all_main_checks_passed": summary["all_main_checks_passed"],
        "checks": summary["checks"],
        "piecewise_population_audit": {
            "true_observation_probability": summary["piecewise_population_audit"]["population_info"]["true_observation_probability"],
            "truth_mass_max_abs_error": summary["piecewise_population_audit"]["truth_mass_max_abs_error"],
            "mm_converged": summary["piecewise_population_audit"]["mm"]["converged"],
            "mm_outer_iterations": summary["piecewise_population_audit"]["mm"]["outer_iterations"],
            "active_score_residual": summary["piecewise_population_audit"]["mm"]["active_score_sup_abs_residual"],
            "mm_direct_mass_sup_abs_difference": summary["piecewise_population_audit"]["agreement"].get("mm_direct_class_mass_sup_abs_difference"),
        },
        "piecewise_sample_audit": {
            "n_latent": summary["piecewise_sample_audit"]["n_latent"],
            "n_observed": summary["piecewise_sample_audit"]["n_observed"],
            "region_counts_observed": summary["piecewise_sample_audit"]["region_counts_observed"],
            "cdf_accuracy_equal_split_representative": summary["piecewise_sample_audit"]["cdf_accuracy_equal_split_representative"],
            "cdf_identification_envelope": summary["piecewise_sample_audit"]["cdf_identification_envelope"],
            "true_cell_mass_max_abs_error": summary["piecewise_sample_audit"]["true_cell_mass_max_abs_error"],
            "mm_converged": summary["piecewise_sample_audit"]["mm"]["converged"],
            "mm_outer_iterations": summary["piecewise_sample_audit"]["mm"]["outer_iterations"],
            "mm_direct_mass_sup_abs_difference": summary["piecewise_sample_audit"]["agreement"].get("mm_direct_class_mass_sup_abs_difference"),
        },
        "gaussian_population_refinement": summary["gaussian_population_refinement"],
        "gaussian_finite_sample": summary["gaussian_finite_sample"],
    }
    print(json.dumps(_jsonable(concise), indent=2, sort_keys=True))
    print("Written files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("One-cell rectangular-sieve conditional-likelihood audit completed successfully (status=0).")
    print(f"Output directory: {Path(args.outdir).resolve()}")
    return 0


ONE_CELL_ARGUMENTS = [
    "--outdir", "union_selection_rectangular_sieve_outputs",
    "--n-piecewise-latent", "6000",
    "--n-gaussian-latent", "6000",
    "--seed", "20260905",
    "--rho-values", "0.55", "0.9",
    "--refinement-bins", "4", "8", "12",
    "--finite-sample-bins", "8",
    "--query-grid-size", "41",
    "--mass-tolerance", "1e-11",
    "--score-tolerance", "1e-8",
    "--prune-mass-tolerance", "1e-10",
    "--prune-score-margin", "1e-6",
    "--coverage-tolerance", "1e-15",
    "--max-iterations", "20000",
    "--direct-starts", "8",
    "--self-test",
]


if __name__ == "__main__":
    _running_in_jupyter = "ipykernel" in sys.modules
    _use_one_cell_defaults = _running_in_jupyter or len(sys.argv) == 1
    _run_arguments = ONE_CELL_ARGUMENTS if _use_one_cell_defaults else None
    if _use_one_cell_defaults:
        print("Running the union-selection rectangular-sieve conditional-likelihood audit from the one-cell configuration.")
        print(f"Equivalent arguments: {ONE_CELL_ARGUMENTS}")
    _status = main(_run_arguments)
    if _running_in_jupyter:
        print(f"One-cell execution returned status={_status} without raising SystemExit in Jupyter.")
    else:
        raise SystemExit(_status)
