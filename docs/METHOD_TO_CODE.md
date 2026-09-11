# Method-to-code map

The public API wraps the validated one-cell implementation without changing its likelihood or optimizer.

| Manuscript object | Implementation |
|---|---|
| Rectangular cells and normalized basis densities | `RectangularGrid` |
| Within-cell lower fractions | `RectangularGrid.fraction_below1`, `fraction_below2` |
| Region-A point-density row | `RectangularGrid.density_row` |
| Region-B vertical subdensity row | `RectangularGrid.region_b_subdensity_row` |
| Region-C horizontal subdensity row | `RectangularGrid.region_c_subdensity_row` |
| Union-selection denominator row | `RectangularGrid.selection_row` |
| Observed numerator and denominator matrices | `build_sample_operator` |
| Conditional log-likelihood | `conditional_loglikelihood` |
| Score and Hessian | `conditional_score`, `conditional_hessian` |
| Forced-zero/invisible screening and equivalence collapse | `reduce_operator_columns` |
| Active-set MM update | `_mm_update_on_face`, `fit_active_set_mm` |
| SLSQP and projected-Newton polishing | `fit_hybrid_active_set` |
| Boundary KKT audit | `kkt_diagnostics` |
| Equal-split representative | `expand_class_mass(..., split="equal")` |
| CDF allocation envelope | `cdf_identification_envelope` |
| Region-A-only comparator | `build_region_a_intersection_operator` |
| Coarse-to-fine mass projection | `project_rectangular_mass` |
| Region-stratified folds | `make_stratified_fold_assignment` |
| Predictive scoring mixture | `predictive_mixture_mass` |
| Confirmatory calculation | `run_confirmatory_audit` |

The import-safe package implementation is in `src/union_selection_sieve/_reference_impl.py`. Its source is identical to the validated Version 1.1 one-cell program except for the final `if __name__ == "__main__"` import guard. The unchanged source is archived in `reference/confirmatory_v1_1`.
