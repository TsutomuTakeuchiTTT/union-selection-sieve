# Rectangular-sieve conditional likelihood

## 1. Observation model

For each object, let \(\bm X=(X_1,X_2)\) be the latent bivariate quantity and \(\bm Y=(Y_1,Y_2)\) the observed detection boundary. The catalogue retains the object when

\[
\mathcal O_i=\{X_1\geq Y_{1,i}\}\cup\{X_2\geq Y_{2,i}\}.
\]

Region A contains double detections, Region B a detection only in coordinate 1, Region C a detection only in coordinate 2, and Region D is absent.

Conditioning on \(\bm Y_i\) and catalogue inclusion gives the contribution

\[
\frac{F(\mathcal A_i)}{F(\mathcal O_i)},
\]

where \(\mathcal A_i\) denotes the information actually observed for object \(i\). For a continuous density, the numerator is mixed-dimensional: a two-dimensional density in Region A and a one-dimensional subdensity in Regions B and C.

## 2. Rectangular sieve

Partition a bounded rectangle into cells

\[
R_k=I_{1,k}\times I_{2,k},
\]

with widths \(h_{1,k}\), \(h_{2,k}\), and area \(|R_k|\). Define the normalized basis

\[
\phi_k(x_1,x_2)
=
\frac{\mathbf 1\{(x_1,x_2)\in R_k\}}{|R_k|}.
\]

The sieve density is

\[
f_{\bm p}(x_1,x_2)
=
\sum_{k=1}^{K}p_k\phi_k(x_1,x_2),
\qquad
p_k\geq0,
\qquad
\sum_{k=1}^{K}p_k=1.
\]

Thus \(p_k\) is the probability mass assigned to cell \(R_k\). Every admissible coefficient vector defines a proper probability distribution without any later CDF repair.

## 3. Linear likelihood operators

For a Region A observation \((x_{1,i},x_{2,i})\),

\[
a_{ik}^{A}=\phi_k(x_{1,i},x_{2,i}).
\]

For a Region B observation, \(X_{1,i}=x_{1,i}\) is exact and \(X_2<Y_{2,i}\):

\[
a_{ik}^{B}
=
\int_{-\infty}^{Y_{2,i}}
\phi_k(x_{1,i},x_2)\,\mathrm d x_2.
\]

For a Region C observation,

\[
a_{ik}^{C}
=
\int_{-\infty}^{Y_{1,i}}
\phi_k(x_1,x_{2,i})\,\mathrm d x_1.
\]

For a rectangular basis these integrals are analytic. For example, if \(x_{1,i}\in I_{1,k}\),

\[
a_{ik}^{B}
=
\frac{|I_{2,k}\cap(-\infty,Y_{2,i})|}{|R_k|};
\]

otherwise it is zero.

The selection denominator coefficient is

\[
b_{ik}
=
\int_{\mathcal O_i}\phi_k(\bm x)\,\mathrm d\bm x
=
1-
\frac{|I_{1,k}\cap(-\infty,Y_{1,i})|}{|I_{1,k}|}
\frac{|I_{2,k}\cap(-\infty,Y_{2,i})|}{|I_{2,k}|}.
\]

Therefore both the numerator and denominator are linear in \(\bm p\):

\[
F_{\bm p}(\mathcal A_i)=\bm a_i^{\mathsf T}\bm p,
\qquad
F_{\bm p}(\mathcal O_i)=\bm b_i^{\mathsf T}\bm p.
\]

## 4. Conditional likelihood

With row weights \(\nu_i\) normalized to sum to one,

\[
\ell(\bm p)
=
\sum_i\nu_i
\left[
\log(\bm a_i^{\mathsf T}\bm p)
-
\log(\bm b_i^{\mathsf T}\bm p)
\right].
\]

Its score is

\[
g_k(\bm p)
=
\sum_i\nu_i\frac{a_{ik}}{\bm a_i^{\mathsf T}\bm p}
-
\sum_i\nu_i\frac{b_{ik}}{\bm b_i^{\mathsf T}\bm p}.
\]

Because the likelihood is invariant under a common positive rescaling of \(\bm p\) before simplex normalization,

\[
\bm p^{\mathsf T}\bm g(\bm p)=0.
\]

At a simplex optimum,

\[
g_k(\widehat{\bm p})=0
\quad\text{if }\widehat p_k>0,
\qquad
g_k(\widehat{\bm p})\leq0
\quad\text{if }\widehat p_k=0.
\]

## 5. MM update

At iterate \(\bm p^{(m)}\), define

\[
R_k^{(m)}
=
p_k^{(m)}
\sum_i\nu_i
\frac{a_{ik}}{\bm a_i^{\mathsf T}\bm p^{(m)}},
\]

and

\[
D_k^{(m)}
=
\sum_i\nu_i
\frac{b_{ik}}{\bm b_i^{\mathsf T}\bm p^{(m)}}.
\]

The update is

\[
p_k^{(m+1)}
=
\frac{R_k^{(m)}}{D_k^{(m)}+\lambda_m},
\]

where \(\lambda_m\) is chosen so that \(\sum_kp_k^{(m+1)}=1\). The implementation solves this one-dimensional equation by a bracketed root finder and verifies likelihood nondecrease after every update.

## 6. Boundary KKT and operator equivalence

A cell with zero numerator coverage and positive denominator coverage is forced to have zero mass at an optimum. A cell with zero numerator and zero denominator coverage is completely invisible and must be reported as nonidentified rather than arbitrarily assigned mass.

Two cells are operator-equivalent in the realized experiment when their complete columns in both \(A\) and \(B\) are equal. Only their total mass is identified. The program collapses exact equivalence classes before optimization and reports a CDF identification envelope obtained by allocating each class mass over its member cells in every likelihood-equivalent way.

The envelope is not a confidence interval. It quantifies only finite-sample geometric nonidentifiability conditional on the fitted class masses.

## 7. Scope

The construction is a nonnegative finite-dimensional sieve estimator. It does not establish that an unrestricted continuous NPMLE has finite support, nor does it prove global uniqueness or asymptotic consistency as the grid is refined. Those require separate analysis. The present audit verifies the observation operators, optimization equations, properness, exact representability benchmark, and controlled approximation behavior.
