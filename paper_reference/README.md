# Frozen paper reference summaries

These compact files record the finalized numerical values reported in the manuscript. They are provided for regression comparison and figure regeneration; they are not substitutes for the full confirmatory output directory.

- `expected_confirmatory_summary.json`: headline fit counts, KKT values, A--C versus A-only comparison, and fixed-16 resolution diagnostics.
- `population_refinement.csv`: population CDF approximation under 4, 8, and 12 bins per coordinate.
- `abc_vs_aonly_fixed16.csv`: condition-specific paired comparison at the primary resolution.
- `resolution_performance.csv`: overall deployable resolution-rule performance.
- `selection_frequencies.csv`: CDF-oracle, predictive-oracle, and repeated-CV resolution frequencies.

The full run additionally writes per-catalogue metrics, fold scores, CDF arrays, seed and fold registries, start audits, failures, environment metadata, and file hashes.
