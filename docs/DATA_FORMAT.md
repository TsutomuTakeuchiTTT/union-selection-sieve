# Observed catalogue format

The estimator takes one row for each object that appears in the union-selected catalogue. The core columns are:

| Column | Meaning |
|---|---|
| `y1`, `y2` | object-specific thresholds |
| `delta1`, `delta2` | detection indicators, each equal to 0 or 1 |
| `x1_obs`, `x2_obs` | observed coordinate values when the corresponding indicator is 1 |

The detection convention is inclusive:

```text
delta_b = 1  if X_b >= Y_b
delta_b = 0  if X_b <  Y_b
```

The observable patterns are:

| Region | `(delta1, delta2)` | Recorded information |
|---|---:|---|
| A | `(1, 1)` | exact `x1_obs`, exact `x2_obs` |
| B | `(1, 0)` | exact `x1_obs`, event `X2 < y2` |
| C | `(0, 1)` | event `X1 < y1`, exact `x2_obs` |

Region D, `(0,0)`, must not be included in the input file because such objects are absent from the observed sample. Their omission is represented by the union-selection denominator.

A censored coordinate can be blank in the CSV. The high-level loader replaces it internally by its threshold as a harmless placeholder; that coordinate is not point-evaluated in the relevant subdensity operator.

## Minimal example

```csv
y1,y2,delta1,delta2,x1_obs,x2_obs
0.0,0.0,1,1,0.7,0.4
0.0,0.0,1,0,0.9,
0.0,0.0,0,1,,1.1
```

## Analysis rectangle

The rectangular sieve has bounded support. Every exact detected coordinate must lie in the specified analysis rectangle. The rectangle is part of the finite-sieve estimand and must be held fixed across compared resolutions and cross-validation folds.

Thresholds may lie inside or outside individual cells. The code evaluates exact overlap fractions with every rectangular basis function.
