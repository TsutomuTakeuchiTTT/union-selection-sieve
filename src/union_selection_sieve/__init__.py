"""Union-selection conditional-likelihood rectangular-sieve estimation."""

from ._version import __version__
from .api import FitConfig, SieveEstimate, fit_catalogue, prepare_observed_catalogue, uniform_grid
from ._reference_impl import (
    GAUSSIAN_LOWER,
    GAUSSIAN_UPPER,
    GAUSSIAN_Y_MASS,
    GAUSSIAN_Y_SUPPORT,
    RectangularGrid,
    TruncatedBivariateNormalTruth,
    build_sample_operator,
    cdf_identification_envelope,
    confirmatory_main,
    run_confirmatory_audit,
    run_self_tests,
    simulate_gaussian_catalogue,
)

__all__ = [
    "__version__",
    "FitConfig",
    "SieveEstimate",
    "fit_catalogue",
    "prepare_observed_catalogue",
    "uniform_grid",
    "RectangularGrid",
    "TruncatedBivariateNormalTruth",
    "build_sample_operator",
    "cdf_identification_envelope",
    "simulate_gaussian_catalogue",
    "run_self_tests",
    "run_confirmatory_audit",
    "confirmatory_main",
    "GAUSSIAN_LOWER",
    "GAUSSIAN_UPPER",
    "GAUSSIAN_Y_SUPPORT",
    "GAUSSIAN_Y_MASS",
]
