# Implementation validation summary, Version 1.1

## Exact regression of the Version 1.0 failure

The original rare case was reproduced with the frozen confirmatory generator:

```text
rho                         = 0.55
n_latent                    = 1500
training seed               = 20290933
CV repeat, zero based       = 2
fold, zero based            = 0
bins per axis               = 20
training rows               = 874
```

Version 1.0 stopped after one SLSQP/Newton pass:

```text
converged                    = false
active-score residual       = 5.094069783984512e-02
inactive positive violation = 0
final log likelihood        = -1.9036791188466813
activation events           = 1
```

Version 1.1 repeated optimization on the changed active face:

```text
converged                    = true
polish cycles               = 2
activation events           = 1
total SLSQP iterations      = 131
total Newton steps          = 10
active-score residual       = 7.882583474838611e-14
inactive positive violation = 0
final log likelihood        = -1.9034107132655107
```

The second cycle increased the log likelihood by approximately `2.6841e-4` relative to the stalled Version 1.0 point and reduced the active KKT residual by more than eleven orders of magnitude.

## Independent unit and regression tests

```text
Ran 7 tests in 4.756s

OK
```

The tests verify:

- the unchanged frozen statistical design;
- the new Version 1.1 output identity;
- fingerprint sensitivity to `max_polish_cycles`;
- the complete internal self-test suite;
- exact convergence of the rare active-set regression;
- durable machine-readable failure sidecars;
- coarser-resolution tie breaking.

## Self-contained smoke run

A Version 1.1 smoke profile completed successfully:

```text
all_main_checks_passed                         = true
n_failed_fits                                 = 0
n_successful_full_fits                        = 24
n_successful_cv_fits                          = 36
maximum_active_score_residual                 = 2.4562e-10
maximum_inactive_positive_score_violation     = 0
bias_variance_identity_holds                  = true
status                                        = 0
```

The smoke run's scientific criteria are not interpreted because it contains only four catalogues. It validates the complete program path, output generation, checkpointing, aggregation, and the unchanged confirmatory-analysis machinery.
