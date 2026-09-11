# Reproducibility protocol

## Three execution levels

### 1. Unit and operator checks

```bash
union-selection-self-test
pytest
```

These check exact operator identities, complete-data reduction, score calculations, MM monotonicity, boundary behavior, and selected regression cases.

### 2. Smoke calculation

```bash
python scripts/run_from_config.py configs/smoke.json
```

The smoke profile uses two conditions, two catalogues per condition, one three-fold cross-validation repeat, a smaller population-validation sample, and the same candidate resolutions as the paper. It is intended to verify a new installation, not to reproduce the paper tables.

### 3. Frozen confirmatory calculation

```bash
python scripts/run_from_config.py configs/paper_confirmatory.json
```

The confirmatory profile fixes:

- correlations `0.55` and `0.90`;
- latent sample sizes `1500`, `3000`, and `6000`;
- 30 independent catalogues per condition;
- candidate resolutions `12`, `16`, and `20` bins per coordinate;
- primary resolution `16`;
- three repeats of five-fold region-stratified cross-validation;
- predictive mixture strength `1.0`;
- base seed `20260912`;
- active and inactive score tolerance `1e-9`.

## Checkpoints

Each full or fold-level fit is checkpointed. `--resume` reuses a checkpoint only when its configuration fingerprint matches the requested statistical and numerical design. Use a new output directory when changing the design.

## Failure behavior

The paper run uses `--fail-fast`. Before an unresolved fit terminates the run, the program writes a fit-specific diagnostic with its identity, active set, likelihood, KKT residuals, and solver state.

## Frozen reference values

`paper_reference/expected_confirmatory_summary.json` records the headline finalized values. After a complete run, compare them with

```bash
python scripts/verify_reference_summary.py outputs/confirmatory/comparison_summary.json
```

Small platform-dependent floating-point differences are acceptable; the statistical design, convergence status, resolution decisions, and reported summaries should agree.

## Output integrity

The confirmatory program writes `environment.json`, a manifest, a random-seed registry, fold-balance records, and SHA-256 hashes. Retain these files with any archived release of the results.
