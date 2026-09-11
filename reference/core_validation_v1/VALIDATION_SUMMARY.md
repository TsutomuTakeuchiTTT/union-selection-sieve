# Validation summary

## Status

The reference run completed with status 0. All built-in checks and all six independent unit tests passed.

## Analytic operator checks

- Selection denominator identity error: `0.0`
- Region B analytic integral error: `0.0`
- Region C analytic integral error: `0.0`
- Complete-data reduction error: `0.0`
- Row-permutation difference: `0.0`
- Analytic-score versus finite-difference error: `1.75e-10`
- One MM step log-likelihood increment: `+1.518e-2`

## Exactly representable piecewise-constant model

Population observation probability: `0.8420`.

Population fit:

- maximum cell-mass error: `2.30e-11`
- MM iterations: `56`
- active-score residual: `8.20e-10`
- MM versus multistart SLSQP mass difference: `8.15e-9`
- MM versus SLSQP log-likelihood difference: `1.11e-15`

Finite sample:

- latent objects: `6000`
- observed objects: `5027`
- Regions A, B, C: `2898, 1083, 1046`
- maximum cell-mass error: `9.996e-3`
- CDF RMSE: `3.855e-3`
- CDF maximum absolute error: `1.080e-2`
- MM versus multistart SLSQP mass difference: `1.09e-8`

No observationally equivalent cells remained in this piecewise benchmark, so the CDF identification envelope had zero width.

## Smooth truncated Gaussian population refinement

The fitted sieve masses recovered the exact Gaussian cell probabilities to numerical precision. The remaining CDF error is therefore within-cell piecewise-constant approximation error.

| Correlation | Bins per axis | Cells | CDF RMSE | Maximum CDF error |
|---:|---:|---:|---:|---:|
| 0.55 | 4 | 16 | 0.03116 | 0.10460 |
| 0.55 | 8 | 64 | 0.007920 | 0.02897 |
| 0.55 | 12 | 144 | 0.003592 | 0.01360 |
| 0.90 | 4 | 16 | 0.03395 | 0.13539 |
| 0.90 | 8 | 64 | 0.009276 | 0.04312 |
| 0.90 | 12 | 144 | 0.004283 | 0.02169 |

For both correlations, CDF RMSE decreased strictly under refinement from 4 to 8 to 12 bins per axis.

## Smooth Gaussian finite-sample audit, 8 by 8 grid

### Correlation 0.55

- latent objects: `6000`
- observed objects: `4496`
- Regions A, B, C: `2322, 1088, 1086`
- active likelihood classes: `46`
- forced-zero cells: `2`
- nontrivial equivalence classes: `7`
- MM iterations: `151`
- active-score residual: `5.44e-9`
- equal-split representative CDF RMSE: `0.007782`
- maximum equivalence-only CDF envelope width: `0.004159`

### Correlation 0.90

- latent objects: `6000`
- observed objects: `4407`
- Regions A, B, C: `2524, 951, 932`
- active likelihood classes: `30`
- forced-zero cells: `12`
- nontrivial equivalence classes: `7`
- MM iterations: `150`
- active-score residual: `5.27e-11`
- equal-split representative CDF RMSE: `0.009996`
- maximum equivalence-only CDF envelope width: `0.001100`

All fitted masses were nonnegative and summed to one. No invisible cells were present in either default Gaussian sample.

The equal-split CDF is a reporting representative, not a uniquely identified estimate when a nontrivial equivalence class remains. The accompanying envelope records the range generated solely by likelihood-equivalent within-class allocations. It does not include sampling uncertainty.

## Interpretation

The rectangular sieve resolves the main structural failure of the earlier weighted Dąbrowska route: the fitted object is a proper probability distribution by construction, without negative-mass clipping or monotonicity projection. It also retains the correct conditional-likelihood role of the union-selection probability.

The test does not yet select an optimal grid, establish an unrestricted continuous NPMLE, prove global uniqueness, or provide asymptotic uncertainty quantification.
