#!/usr/bin/env python3
"""Recreate the numerical result figures from exported summary tables.

By default the script reads the compact reference tables in ``paper_reference``.
Pass ``--input-dir`` to use the corresponding CSV files from a fresh
confirmatory run.  Schematic Figures 1--3 are stored separately under
``assets/schematics``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def population_refinement(table: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for rho, group in table.groupby("rho", sort=True):
        group = group.sort_values("bins_per_axis")
        ax.plot(group["bins_per_axis"], group["cdf_rmse"], marker="o", label=rf"$\rho={rho:.2f}$")
    ax.set_xlabel("Bins per coordinate")
    ax.set_ylabel("Population CDF RMSE")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _save(fig, outdir / "fig_gaussian_population_refinement.pdf")


def abc_forest(table: pd.DataFrame, outdir: Path) -> None:
    labels = [rf"$\rho={r.rho:.2f},\ n={int(r.n_latent)}$" for r in table.itertuples()]
    y = np.arange(len(table))[::-1]
    mean = table["mean_difference"].to_numpy()
    low = table["ci95_low"].to_numpy()
    high = table["ci95_high"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.errorbar(mean, y, xerr=np.vstack([mean - low, high - mean]), fmt="o", capsize=3)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"CDF-RMSE difference, $\widehat F_{ABC}-\widehat F_A$")
    _save(fig, outdir / "fig_abc_vs_aonly_fixed16.pdf")


def resolution_regret(table: pd.DataFrame, outdir: Path) -> None:
    labels = ["Fixed 12", "Fixed 16", "Fixed 20", "Repeated CV"]
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.errorbar(
        x,
        table["mean_cdf_rmse_regret"],
        yerr=1.96 * table["se_cdf_rmse_regret"],
        fmt="o",
        capsize=3,
    )
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Mean CDF-RMSE regret")
    _save(fig, outdir / "fig_resolution_cdf_regret.pdf")


def selection_frequencies(table: pd.DataFrame, outdir: Path) -> None:
    order = ["cdf_rmse_oracle", "population_logscore_oracle", "repeat_averaged_cv_max"]
    labels = ["CDF oracle", "Predictive oracle", "Repeated CV"]
    bins = [12, 16, 20]
    x = np.arange(len(order))
    width = 0.23
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for offset, b in enumerate(bins):
        values = []
        for rule in order:
            row = table[(table.selection_rule == rule) & (table.bins_per_axis == b)]
            values.append(float(row.fraction.iloc[0]))
        ax.bar(x + (offset - 1) * width, values, width, label=str(b))
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("Selection frequency")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Bins", frameon=False)
    _save(fig, outdir / "fig_resolution_selection_frequencies.pdf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("paper_figures"))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source = args.input_dir or (repo / "paper_reference")

    population_path = source / "population_refinement.csv"
    abc_path = source / "abc_vs_aonly_fixed16.csv"
    performance_path = source / "resolution_performance.csv"
    frequency_path = source / "selection_frequencies.csv"

    required = [population_path, abc_path, performance_path, frequency_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing input tables: " + ", ".join(missing))

    population_refinement(pd.read_csv(population_path), args.outdir)
    abc_forest(pd.read_csv(abc_path), args.outdir)
    resolution_regret(pd.read_csv(performance_path), args.outdir)
    selection_frequencies(pd.read_csv(frequency_path), args.outdir)
    print(f"Figures written to {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
