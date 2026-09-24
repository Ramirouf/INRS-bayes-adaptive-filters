# Section 3 of the new draft: work split

Leszek's email of 2026-09-19 asks Ignacio and me to implement Section 3 of the new
`main.tex` "on the same examples and the same tests for which we ran the marginalized
versions, so that we can compare directly". This is how we propose to divide it.

Section 4 (bank of filters) is Augusto's and is not covered here.

---

## Where we stand

Section 3 derives every filter from the **joint** posterior, reduced to the scalar
`s_t = x_t'(theta_t - w_{t-1})`. The noise enters only through two numbers,

    chi_t(e_t)   the correction applied to the output
    chi'_t(e_t)  the fraction of the predicted variance that the observation removes

and the four filters (KF, vKF, sKF, fKF) differ only in the shape of the predicted
covariance `Sigma_t`. Everything else is shared.

**Done.** Ignacio's notebook 07 (`9406077`) ports the **sKF** to this form and shows the
joint and the marginal filters are numerically the same filter: over 96000 steps,
`max |w_joint - w_marginal| = 2.7e-12`, floors within 1.6e-11 dB, step counts identical.
It is also 2.9x faster (11.7 us/step against 33.6).

That is consistent with the algebra: in the old exact filter `Gamma_t` is identically
zero and `Q_t = -k P_t`, which Leszek noted in the blue passage of
`main_minorization.tex:2109`. With those two substitutions the old update collapses onto
eq. (43) of the new draft. So **notebooks 03-07 and the gain analysis stay valid**; what
changes is the notation and the derivation we cite.

**Not done.** The sKF is the filter we already had. The genuinely new content of
Section 3 is untouched:

| | status | owner |
|---|---|---|
| sKF-L, joint form | done, nb 07 | Ignacio |
| vKF (diagonal family), Laplacian | not started | Ramiro |
| KF (full covariance), Laplacian | not started | Ramiro |
| Instances by quadrature (gen. Gaussian, Student-t) | not started | Ignacio |
| Gaussian baseline in the new notebooks | dropped from nb 07 | Ignacio |
| Section 5, convergence analysis | placeholder in the draft | Ramiro |

---

## Settle before starting: the value of `b_eta`

Our notebooks use `b_eta = sqrt(v_eta/2)`, matching the noise **variance**. Leszek's
Section 4 remark uses `b_eta = E|eta_t|`, matching the **mean absolute deviation**. For
generalized Gaussian noise with `beta* = 0.2` these differ by a factor of **0.355**, so
`k_t = sigma_t/b_eta` is **2.8x larger** in his runs than in ours.

```
beta* = 0.2 :  E|eta| / sqrt(v_eta) = 0.2509   ->  b_eta(E|eta|) / b_eta(sqrt(v/2)) = 0.355
beta* = 1.0 :                         0.7071   ->                                     1.000
beta* = 2.0 :                         0.7979   ->                                     1.128
```

`k_t` is the only parameter of the exact correction, so this moves every result along the
`k` family. We cannot reproduce his preliminary table, and our two sets of runs will not
be comparable to each other, until we pick one. **Proposal: run everything with both, and
report `k_t` explicitly in every table.** It costs one extra sweep and removes the
ambiguity for good.

---

## Ramiro

### 1. vKF, the diagonal family

One variance per coefficient instead of one for all of them. Eqs. (37)-(38):

    v~_{t,m} = v_{t-1,m} + eps
    sigma_t^2 = sum_m v~_{t,m} x_{t,m}^2
    w_{t,m}   = w_{t-1,m} + (v~_{t,m} x_{t,m} / sigma_t^2) * chi_t(e_t)
    v_{t,m}   = v~_{t,m} * (1 - chi'_t(e_t) * v~_{t,m} x_{t,m}^2 / sigma_t^2)

`O(M)` per step. Reuses Ignacio's `chi_laplacian` unchanged; only `sigma_t` differs.

### 2. KF, the full covariance

Eqs. (33)-(34):

    Sigma~_t  = Sigma_{t-1} + eps*I
    sigma_t^2 = x_t' Sigma~_t x_t
    w_t       = w_{t-1} + (Sigma~_t x_t / sigma_t^2) * chi_t(e_t)
    Sigma_t   = Sigma~_t - chi'_t(e_t) * (Sigma~_t x_t)(Sigma~_t x_t)' / sigma_t^2

`O(M^2)` per step, M = 128. This is the filter the marginal route structurally could not
produce, since it required a factorized prior, and it is the reason for the pivot.
Section 4 claims it is what recovers the loss under coloured input, without prewhitening,
because the update direction `Sigma~_t x_t` is decorrelated and `sigma_t^2` is then
correct in every direction. That claim is worth testing directly on the AR(-0.9) input of
notebook 03.

Watch: `Sigma_t` must stay symmetric and positive semidefinite. The draft guarantees
`chi'_t <= 1` for any noise, and `chi'_t >= 0` for log-concave noise such as the
Laplacian, so the Laplacian case is safe; the quadrature instances below are not, and
that is Ignacio's to flag.

### 3. The convergence gap

Simulation gives an exact/minorized step ratio of 1.56 at 0 dB (nb 07); the old Section 6
analysis predicts 1.13 for fixed-variance filters with white input and M = 16.
`normalized-gain-check.ipynb` already ruled out the variance recursion (1.54 with `v~_t`
frozen). What is left is the input, M, and how the floor is read. One change at a time:
white input at M = 128, then M = 16. If the last lands near 1.13, simulation and analysis
agree and the gap is explained. This feeds Section 5, which is currently a placeholder.

---

## Ignacio

### 1. Instances by quadrature

Only the Gaussian and the Laplacian have closed forms. Section 3.4 says that for every
other density the two numbers come from a one-dimensional quadrature on the scalar
posterior, eq. (24):

    p(s | y) ~ N(s; 0, sigma_t^2) * p_eta(e_t - s)        on a grid of s, then normalized
    chi_t(e_t)  = E[s]
    chi'_t(e_t) = 1 - Var[s] / sigma_t^2                  (eq. 32)

A few hundred nodes suffice, so the cost per step stays `O(M)` for the vKF, sKF and fKF.

The integrand has a cusp at `s = e_t` when `p_eta` is not differentiable at the origin, as
for the generalized Gaussian with `beta <= 1`. The draft's remedy is to place the nodes on
either side of `e_t` in the variable `(|e_t - s|/alpha)^beta`, which removes it.

Deliver `chi_quadrature(e, sigma, density, params)` with the same signature as
`chi_laplacian`, so it drops into any of the four filters.

### 2. A filter matched to the actual noise

The quadrature gives us, for the first time, a filter that assumes the noise we actually
simulate (`beta* = 0.2`) instead of pretending it is Laplacian. Section 4 claims such a
filter recovers from an abrupt change 1.2 to 2.2 times **more slowly** than the Laplacian
ones, because its score redescends and it reads the large errors of a change as outliers.
That is a counterintuitive result and the one worth reproducing first.

Note: the generalized Gaussian with `beta < 1` and Student-t are not log-concave, so
`chi'_t` can be negative and the variance update can **increase** the variance along
`Sigma~_t x_t`. Expected, per Section 3.2, but it needs a guard so runs do not diverge
silently.

### 3. Restore the Gaussian baseline

Notebook 07 dropped the Gaussian sKF that notebook 03 had. Put it back, so every figure
carries the same three references: Gaussian, Laplacian-minorized, Laplacian-exact.

---

## Shared, so the two halves merge

- **Same scenario and same seeds** as notebook 03: `M = 128`, room impulse response of
  section 7.1, AR(-0.9) input, `beta* = 0.2`, SNR 5 dB, `N = 96000`, `R = 20`.
- **Same protocol**: target floor first, `eps` chosen by grid search to land on it, floor =
  mean of the last quarter, converged = first step within 3 dB of the floor.
- **Same filter signature** as `sKF_L_joint(n, x, d, w0, parameters)`, returning
  `{"w_hist", "e"}`, so `run_filter` and `sweep_filter` in notebook 07 take any of them
  without changes.
- **`chi` and `chi'` live in one module**, imported by every notebook, rather than being
  redefined per notebook. Ignacio's `log_mills` / `chi_laplacian` is the starting point.
- **Report `k_t`** (median and range over the run) in every results table. It is the only
  parameter of the exact correction and the axis every comparison moves along.

### One implementation note worth sending upstream

The draft says to evaluate the Mills ratio as `R(z) = sqrt(pi/2)*erfcx(z/sqrt(2))`. That
overflows for `z` large and negative, which is exactly the regime of small `k` and large
error. Ignacio's version stays in the log throughout and has no bad region:

```python
def log_mills(z):  return log_ndtr(-z) + 0.5*z**2 + 0.5*np.log(2*np.pi)

Lambda    = np.tanh((log_R_minus - log_R_plus)/2)
chi_slope = 2*k*np.exp(-np.logaddexp(log_R_minus, log_R_plus)) - k**2*(1 - Lambda**2)
```

Worth proposing as the numerical note in Section 3.4.

---

## Questions for Leszek

1. `b_eta = E|eta_t|` or `sqrt(v_eta/2)`? It changes `k_t` by 2.8x at `beta* = 0.2` and
   decides whether our numbers can be compared with the Section 4 table.
2. Does "the same tests" include the convergence analysis, or does Section 5 wait until it
   is restated for white input in the `chi`, `tau`, `k` notation?
3. Is the `beta* = 0.2`-matched quadrature filter ours to run, or was it part of the
   exploratory work behind Section 4?
4. Section 3.4 suggests `erfcx` for the Mills ratio; it overflows where `k - |u|` is large
   and negative. Is the log form above acceptable for the draft?
