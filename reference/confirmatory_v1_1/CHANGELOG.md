# Changelog

## Version 1.1.0

### Corrected

- Added an outer active-face polishing loop. After projected Newton changes the active set, SLSQP is rerun on the new face before another Newton and full KKT audit.
- Added `--max-polish-cycles`, with default value 8.
- Added a deterministic regression test for the exact Version 1.0 failure at `rho=0.55`, `n_latent=1500`, training seed `20290933`, CV repeat 2, fold 0, and a 20-by-20 sieve.
- Added durable `*__failure.json` sidecars written before fail-fast exception propagation.
- Enriched failure messages with fit identity, active and inactive KKT residuals, polish-cycle count, and the sidecar path.
- Added per-fit diagnostics for polish cycles, active-set changes, activation events, SLSQP iterations, Newton steps, and likelihood increment.
- Changed the default output directory to `union_selection_sieve_confirmatory_monte_carlo_v1_1_outputs` so that Version 1.0 checkpoints are not reused.
- Included the polish-cycle setting and numerical-amendment flag in the configuration fingerprint.

### Unchanged

- Conditional likelihood and rectangular-sieve estimator.
- Correlations and latent sample sizes.
- Thirty independent catalogues per condition.
- Candidate resolutions 12, 16, and 20 per axis.
- Primary fixed resolution 16 per axis.
- Repeat-averaged cross-validation comparator.
- Seed family beginning at 20260912.
- Population-validation design.
- A-only baseline and simulation-only oracles.
- Pre-specified descriptive decision criteria.
