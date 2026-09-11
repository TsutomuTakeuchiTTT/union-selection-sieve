"""Console entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import _reference_impl as impl
from .api import FitConfig, fit_catalogue, uniform_grid


def confirmatory(argv: Optional[Sequence[str]] = None) -> int:
    return impl.confirmatory_main(argv)


def _fit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the union-selection rectangular-sieve estimator to a CSV catalogue"
    )
    parser.add_argument("--input", required=True, help="CSV file with y1,y2,delta1,delta2,x1_obs,x2_obs")
    parser.add_argument("--outdir", default="fit_outputs")
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--x1-lower", type=float, default=-3.0)
    parser.add_argument("--x1-upper", type=float, default=3.0)
    parser.add_argument("--x2-lower", type=float, default=-3.0)
    parser.add_argument("--x2-upper", type=float, default=3.0)
    parser.add_argument("--query-grid-size", type=int, default=41)
    parser.add_argument("--score-tolerance", type=float, default=1e-9)
    parser.add_argument("--direct-audit-starts", type=int, default=0)
    return parser


def fit_csv(argv: Optional[Sequence[str]] = None) -> int:
    args = _fit_parser().parse_args(argv)
    data = pd.read_csv(args.input)
    grid = uniform_grid(
        args.bins,
        args.x1_lower,
        args.x1_upper,
        args.x2_lower,
        args.x2_upper,
    )
    config = FitConfig(
        score_tolerance=args.score_tolerance,
        direct_audit_starts=args.direct_audit_starts,
    )
    estimate = fit_catalogue(data, grid, config=config)
    paths = estimate.write(args.outdir, query_grid_size=args.query_grid_size)
    print(json.dumps(estimate.summary(), indent=2, sort_keys=True))
    print("Written files:")
    for key, path in paths.items():
        print(f"  {key}: {path.resolve()}")
    return 0


def _demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact worked example")
    parser.add_argument("--outdir", default="demo_outputs")
    parser.add_argument("--n-latent", type=int, default=1200)
    parser.add_argument("--rho", type=float, default=0.55)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--query-grid-size", type=int, default=41)
    return parser


def demo(argv: Optional[Sequence[str]] = None) -> int:
    args = _demo_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    latent, observed, simulation = impl.simulate_gaussian_catalogue(
        args.n_latent,
        args.seed,
        args.rho,
        impl.GAUSSIAN_LOWER,
        impl.GAUSSIAN_UPPER,
        impl.GAUSSIAN_Y_SUPPORT,
        impl.GAUSSIAN_Y_MASS,
    )
    grid = uniform_grid(args.bins, impl.GAUSSIAN_LOWER, impl.GAUSSIAN_UPPER)
    estimate = fit_catalogue(observed, grid)
    paths = estimate.write(outdir, query_grid_size=args.query_grid_size)
    latent.to_csv(outdir / "latent_catalogue.csv", index=False)
    observed.to_csv(outdir / "observed_ABC_catalogue.csv", index=False)

    truth = impl.TruncatedBivariateNormalTruth(args.rho)
    cdf = pd.read_csv(paths["cdf_grid"])
    cdf["cdf_truth"] = [truth.cdf(x1, x2) for x1, x2 in zip(cdf.x1, cdf.x2)]
    cdf["equal_split_error"] = cdf["cdf_equal_split"] - cdf["cdf_truth"]
    cdf.to_csv(paths["cdf_grid"], index=False)
    rmse = float(np.sqrt(np.mean(np.square(cdf["equal_split_error"]))))

    summary = {
        "simulation": simulation,
        "fit": estimate.summary(),
        "cdf_rmse": rmse,
        "outdir": str(outdir.resolve()),
    }
    (outdir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def self_test(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the validated analytic and operator self-tests")
    parser.parse_args(argv)
    result = impl.run_self_tests()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("all_passed", False) else 1


__all__ = ["confirmatory", "fit_csv", "demo", "self_test"]
