# Validated source snapshots

This directory preserves the exact self-contained programs used during method validation.

- `confirmatory_v1_1`: final independent-seed confirmatory Monte Carlo, including the numerical-control correction that repeats constrained polishing after an active-set change.
- `core_validation_v1`: operator identities, exactly representable population recovery, finite-sample fitting, and population grid-refinement checks.

The files in these directories are archival and intentionally remain one-cell/self-contained. The installable package under `src/` provides a cleaner public interface while retaining the same validated numerical core.
