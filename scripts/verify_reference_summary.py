#!/usr/bin/env python3
"""Compare a completed confirmatory summary with the frozen paper values."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _close(actual: float, expected: float, atol: float, rtol: float) -> bool:
    return math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path, help="comparison_summary.json from a completed run")
    parser.add_argument("--atol", type=float, default=5e-12)
    parser.add_argument("--rtol", type=float, default=5e-9)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    expected = json.loads((repo / "paper_reference/expected_confirmatory_summary.json").read_text())
    actual = json.loads(args.summary.read_text())

    checks: dict[str, bool] = {}
    checks["candidate_resolutions"] = list(actual["candidate_resolutions"]) == expected["candidate_resolutions"]
    checks["primary_resolution"] = int(actual["primary_resolution"]) == expected["primary_resolution"]
    checks["n_successful_full_fits"] = int(actual["n_successful_full_fits"]) == expected["n_successful_full_fits"]
    checks["n_successful_cv_fits"] = int(actual["n_successful_cv_fits"]) == expected["n_successful_cv_fits"]
    checks["n_failed_fits"] = int(actual["n_failed_fits"]) == expected["n_failed_fits"]
    checks["maximum_active_score_residual"] = _close(
        actual["maximum_active_score_residual"], expected["maximum_active_score_residual"], args.atol, args.rtol
    )
    checks["maximum_inactive_positive_score_violation"] = _close(
        actual["maximum_inactive_positive_score_violation"],
        expected["maximum_inactive_positive_score_violation"], args.atol, args.rtol
    )
    checks["abc_minus_a_only_fixed16"] = _close(
        actual["ABC_minus_A_only_fixed16_mean_CDF_RMSE_difference"],
        expected["abc_minus_a_only_fixed16"]["mean_difference"], args.atol, args.rtol
    )
    checks["fixed16_mean_cdf_rmse_regret"] = _close(
        actual["fixed16_mean_cdf_rmse_regret"],
        expected["fixed16"]["mean_cdf_rmse_regret"], args.atol, args.rtol
    )
    checks["fixed16_fraction_within_5_percent_oracle"] = _close(
        actual["fixed16_fraction_within_5_percent_cdf_oracle"],
        expected["fixed16"]["fraction_within_5_percent_oracle"], args.atol, args.rtol
    )

    print(json.dumps(checks, indent=2, sort_keys=True))
    passed = all(checks.values())
    print("Reference comparison:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
