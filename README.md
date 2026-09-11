# Union-selection conditional-likelihood sieve estimation

Reference implementation for the manuscript

> **Bivariate Distribution Estimation under Truncation and One-Sided Censoring: A Conditional-Likelihood Sieve Method for Union-Selected Samples**

The code estimates a bivariate parent distribution when an object enters the observed sample if at least one coordinate exceeds its object-specific threshold. Region A records are fully observed, Regions B and C contain one exact and one left-censored coordinate, and Region D is absent from the catalogue. The likelihood is conditioned jointly on the observed threshold and the union-selection event.

This repository is intentionally separate from the earlier `Bivariate-survival-analysis-in-astronomy` project. It implements the **conditional-likelihood rectangular-sieve estimator** of the current manuscript and does **not** implement the retired inverse-probability-weighted Dąbrowska fixed-point construction.

## Main features

- exact mixed-dimensional numerator operators for Regions A, B, and C;
- the union-selection denominator for every included observation;
- normalized rectangular basis densities with simplex-constrained cell masses;
- forced-zero and invisible-cell screening;
- exact observational-equivalence classes;
- active-set minorize--maximize updates;
- repeated SLSQP and projected-Newton polishing after active-set changes;
- complete active and inactive boundary KKT diagnostics;
- an equal-split CDF representative and a likelihood-equivalence allocation envelope;
- a Region-A-only intersection-selection comparator;
- repeated region-stratified cross-validation with a vanishing predictive mixture;
- checkpointed, resumable confirmatory Monte Carlo calculations.

## Repository layout

```text
src/union_selection_sieve/   importable package and high-level API
scripts/                     command-line entry points and figure generation
notebooks/                   self-contained one-cell notebooks
reference/                   unmodified validated one-cell source snapshots
configs/                     frozen smoke and confirmatory configurations
examples/                    compact worked example and CSV schema
paper_reference/             finalized numerical summaries used in the paper
assets/schematics/           final schematic Figures 1--3
 tests/                      automated numerical checks
 docs/                       method, data-format, and reproducibility notes
```

## Installation

Python 3.11 or later is required.
The automated test suite is run on Python 3.11 and 3.12.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Then install the package:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

A Conda environment is also provided:

```bash
conda env create -f environment.yml
conda activate union-selection-sieve
python -m pip install -e ".[dev]"
```

## Fast verification

Run the analytic and operator self-tests:

```bash
union-selection-self-test
```

Run a compact end-to-end example:

```bash
union-selection-demo --outdir outputs/demo
```

This writes a simulated latent catalogue, the observed Region-A--C catalogue, fitted cell masses, equivalence classes, KKT diagnostics, and a CDF table.

Run the small confirmatory smoke profile:

```bash
python scripts/run_from_config.py configs/smoke.json
```

## Reproducing the confirmatory calculation

The frozen paper configuration is stored in `configs/paper_confirmatory.json`. Run it with

```bash
python scripts/run_from_config.py configs/paper_confirmatory.json
```

or directly with

```bash
union-selection-confirmatory \
  --profile confirmatory \
  --outdir outputs/confirmatory \
  --bins-values 12 16 20 \
  --primary-bins 16 \
  --mixture-strength 1.0 \
  --seed 20260912 \
  --mass-tolerance 1e-12 \
  --score-tolerance 1e-9 \
  --prune-mass-tolerance 1e-7 \
  --prune-score-margin 1e-7 \
  --coverage-tolerance 1e-15 \
  --warm-mixing 1e-9 \
  --max-iterations 30000 \
  --mm-prepolish-iterations 2000 \
  --newton-tolerance 1e-12 \
  --polish-zero-tolerance 1e-13 \
  --max-polish-cycles 8 \
  --direct-audit-starts 4 \
  --resume --self-test --fail-fast
```

The full design contains two correlations, three latent sample sizes, 30 independent catalogues per condition, three repeats of five-fold region-stratified cross-validation, and three candidate resolutions. It comprises 1080 full-sample fits and 8100 cross-validation fits. The recorded reference execution took approximately 7.6 hours on one Windows workstation; runtime depends strongly on hardware and existing checkpoints.

The expected headline values are stored in `paper_reference/expected_confirmatory_summary.json`. Use

```bash
python scripts/verify_reference_summary.py outputs/confirmatory/comparison_summary.json
```

to compare a completed run with the frozen reference values.

## Fitting a user catalogue

The CSV interface expects one row for each included object and the columns

```text
y1, y2, delta1, delta2, x1_obs, x2_obs
```

Region-D rows must not be supplied because such objects do not occur in the observed catalogue. A censored coordinate may be left blank; the loader fills it with its threshold as an internal placeholder. Every detected coordinate must have a finite observed value.

Example:

```bash
union-selection-fit \
  --input examples/sample_catalog.csv \
  --outdir outputs/example_fit \
  --bins 12 \
  --x1-lower -3 --x1-upper 3 \
  --x2-lower -3 --x2-upper 3
```

The analysis rectangle is part of the finite-sieve estimand. It should be fixed from scientific considerations and should not silently change across resolutions or validation folds.

See `docs/DATA_FORMAT.md` and `examples/README.md` for details.

## Python API

```python
import pandas as pd
from union_selection_sieve import FitConfig, fit_catalogue, uniform_grid

observed = pd.read_csv("examples/sample_catalog.csv")
grid = uniform_grid(12, -3.0, 3.0)
fit = fit_catalogue(observed, grid, config=FitConfig())

print(fit.summary())
print(fit.cdf(0.0, 0.0))
print(fit.cdf_envelope(0.0, 0.0))
fit.write("outputs/api_fit")
```

## Paper figures

The final schematic Figures 1--3 are stored under `assets/schematics`. The numerical figures can be regenerated from the compact finalized tables with

```bash
python scripts/make_paper_figures.py --outdir paper_figures
```

A fresh confirmatory run writes the complete machine-readable tables from which the paper summaries can be regenerated.

## Validation and provenance

The package module `src/union_selection_sieve/_reference_impl.py` is derived from the validated confirmatory one-cell source. The only packaging change is that the final automatic execution block is protected by `if __name__ == "__main__"`, allowing safe import. The unmodified source, notebook, text copy, Version 1.0-to-1.1 patch, and targeted regression record are retained under `reference/confirmatory_v1_1`.

The separate operator and population-level validation program is retained under `reference/core_validation_v1`. It checks exact Region-B and Region-C integration, the selection-denominator identity, finite-difference agreement of the score, complete-data reduction, MM likelihood monotonicity, recovery of exactly representable cell masses, and rectangular-grid population refinement.

No negative-mass repair, CDF monotonicity projection, smoothing, Dąbrowska plug-in operator, or old boundary inverse-probability weighting is used by this implementation.

## Citation

Citation metadata are provided in `CITATION.cff`. Until the manuscript receives a permanent identifier, cite the software release together with the manuscript title above. After publication, update `CITATION.cff`, `.zenodo.json`, and this section with the journal DOI.

## License

BSD 3-Clause License. See `LICENSE`.
