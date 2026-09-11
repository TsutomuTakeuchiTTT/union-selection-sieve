# Union-selection rectangular-sieve conditional-likelihood audit

This package implements the next controlled step in the reconstruction of the bivariate truncation/censoring paper. It replaces the earlier weighted Dąbrowska update by a conditional likelihood derived directly from the union-selection rule and extends the validated finite-support likelihood to continuous bivariate data through normalized rectangular basis functions.

## Recommended execution

Open

`union_selection_rectangular_sieve_v1_onecell.ipynb`

and run its only code cell. The notebook is fully self-contained. It does not import any project-specific module.

The same code can be pasted into one Jupyter cell from

`union_selection_rectangular_sieve_v1_onecell.txt`.

A command-line version is also provided:

```bash
python union_selection_rectangular_sieve_v1_onecell.py
```

## Dependencies

- Python 3.10 or later
- NumPy
- pandas
- SciPy

## What the code tests

1. Exact integration of Region A, B, and C likelihood contributions for rectangular basis functions.
2. The identity between the selection denominator and one minus the lower-left CDF.
3. Reduction to the empirical histogram under complete observation.
4. Recovery of an exactly representable piecewise-constant parent density.
5. Agreement between active-set MM and an independent multistart SLSQP optimizer.
6. Gaussian population sieve refinement at 4, 8, and 12 bins per axis for correlations 0.55 and 0.90.
7. Finite-sample Gaussian fits on an 8 by 8 grid.
8. KKT boundary conditions, zero-numerator screening, and exact operator-equivalence classes.
9. CDF identification envelopes when several geometric cells are likelihood-equivalent in a finite sample.

## Deliberately excluded

The code does not use:

- the Dąbrowska plug-in operator;
- boundary-conditioned inverse-probability weights;
- negative-mass clipping;
- CDF monotonicity projection;
- density smoothing;
- a claimed continuous-support NPMLE support theorem.

The output is a finite-dimensional rectangular-sieve estimator. Grid selection and asymptotic theory remain separate tasks.

## Default output directory

`union_selection_rectangular_sieve_outputs`

The output includes the latent and observed mock catalogues, cell-mass estimates, operator-membership audits, MM histories, active-set events, direct-optimizer runs, CDF accuracy tables, CDF identification envelopes, a machine-readable summary, environment metadata, and SHA-256 hashes.
