# Repository preparation audit

Prepared: 2026-09-11

## Source basis

- Final confirmatory source: `union-selection-sieve-confirmatory-monte-carlo-v1.1.0-onecell`.
- Final core/operator validation source: `union-selection-rectangular-sieve-v1.0.0-onecell`.
- The package implementation is an import-safe copy of the validated confirmatory source. The only source-level modification is a header plus an `if __name__ == "__main__"` guard around the final autorun block.
- The exact unmodified source snapshots are retained under `reference/`.

## Automated checks performed in the preparation environment

- Python compilation of package, scripts, and tests: passed.
- Pytest suite: 6 passed.
- Validated embedded analytic/operator self-tests: passed.
- Public CSV fitting interface on `examples/sample_catalog.csv`: converged; masses nonnegative and normalized; active and inactive KKT tolerances satisfied.
- Compact end-to-end demo: completed and wrote all expected output files.
- Confirmatory smoke profile: 24 full fits and 36 cross-validation fits; all main checks passed; zero failed fits; maximum active score residual `2.4561597200545293e-10`; maximum inactive positive score violation `0.0`.
- Numerical paper-figure generation from the compact frozen tables: completed for all four result figures.
- Wheel build with local build dependencies and no network access: passed.

## Intentionally not rerun during repository assembly

The full 180-catalog, 9180-fit confirmatory experiment was not rerun during packaging. Its exact validated source, frozen configuration, expected headline values, Version 1.0-to-1.1 patch, and existing confirmatory records are included. The full calculation can be resumed or repeated with `configs/paper_confirmatory.json`.

## Repository URL

The metadata currently use the proposed URL:

`https://github.com/TsutomuTakeuchiTTT/union-selection-sieve`

Change this in `pyproject.toml`, `CITATION.cff`, `.zenodo.json`, and the manuscript if a different repository name is chosen.
