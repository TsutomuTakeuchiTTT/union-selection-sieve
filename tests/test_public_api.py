from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from union_selection_sieve import FitConfig, fit_catalogue, uniform_grid
from union_selection_sieve.api import prepare_observed_catalogue


ROOT = Path(__file__).resolve().parents[1]


def test_example_catalogue_fit_is_proper_and_kkt_admissible() -> None:
    observed = pd.read_csv(ROOT / "examples/sample_catalog.csv")
    grid = uniform_grid(8, -3.0, 3.0)
    estimate = fit_catalogue(observed, grid, config=FitConfig())
    summary = estimate.summary()
    assert summary["converged"]
    assert np.isclose(summary["probability_mass_sum"], 1.0, atol=1e-12)
    assert summary["minimum_mass"] >= 0.0
    assert summary["active_score_residual"] < 1e-9
    assert summary["inactive_positive_score_violation"] < 1e-9


def test_cdf_envelope_contains_equal_split_representative() -> None:
    observed = pd.read_csv(ROOT / "examples/sample_catalog.csv")
    estimate = fit_catalogue(observed, uniform_grid(8, -3.0, 3.0))
    lower, equal, upper = estimate.cdf_envelope(0.1, -0.2)
    assert 0.0 <= lower <= equal <= upper <= 1.0
    assert np.isclose(equal, estimate.cdf(0.1, -0.2), atol=1e-13)


def test_region_d_rows_are_rejected() -> None:
    bad = pd.DataFrame(
        {
            "y1": [0.0],
            "y2": [0.0],
            "delta1": [0],
            "delta2": [0],
            "x1_obs": [np.nan],
            "x2_obs": [np.nan],
        }
    )
    with pytest.raises(ValueError, match="Region-D"):
        prepare_observed_catalogue(bad)
