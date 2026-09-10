# Decisions

Append-only log. Each entry records what was decided, why, and what was rejected and why.

---

## 2026-09-10 — Section 6 (Laplacian prior, Laplacian likelihood) in `section6.ipynb`

Context: Section 6 of the draft, `LF (exact)` in Table 2, was the last block of the filter
hierarchy not yet implemented. Sections 3 to 5 already live in `filters.py`.

### Scope of the implementation

- **Decided**: implement the full Section 6.1-6.3 reference — the composite noise through
  `Psi_t` (80), the functional `I_t` (81), the moments (84), and the median and MAD by
  transform inversion (85)-(91).
- **Why**: the median and the MAD are what the Laplacian family projects on (11). They are
  the only route that keeps the Laplacian prior alive past a single step, and `LF (exact)` is
  the reference Section 7.5(v) needs in order to ask whether that prior earns its cost.
- **Rejected**: the moments-only Gaussian projection of Section 6.4 / Remark 10. It is much
  cheaper, but by construction it collapses to the Section 5 filter after one step, so it
  would not have implemented Section 6 at all. `marginal_moments` implements (84) anyway,
  since (91) needs the mean and since it makes Remark 10 reachable.

### Numerical inversion

- **Decided**: real-axis Simpson quadrature *plus* the saddlepoint tilt of Section 6.3 and
  Appendix B, with the contour `w = s + i*gamma` and `gamma` from `K'(gamma) = e_t`.
- **Why**: `Psi_t` and every multiplier are rational in `w`, so a complex argument costs
  nothing, and the `exp(-gamma*e_t)` prefactor cancels in every ratio (84), (88), (90) and is
  never formed. Measured: the plain axis and the tilt agree to 1e-10 up to `e_t = 8`, then the
  plain axis fails outright (97% error at `e_t = 30`) while the tilted result stays
  grid-converged to 1e-12.
- **Rejected**: plain real-axis quadrature alone. It degrades exactly on the outlier steps
  that a heavy-tailed likelihood exists to handle, which would put an error inside the object
  the other filters are measured against.

### Confluent limit (87)

- **Decided**: implement (87) literally, and apply the substitution **globally** — when
  `|b_{t-1} - b_xi| < 1e-2 * b_tilde`, both prior scales become `b_tilde/sqrt(2)` in `c_n`,
  `g_n`, `Psi_t`, `q_hat_m`, `R^(1)`, `R^(2)` and `K_t`, not only inside the truncated
  transform.
- **Why**: (88) and (90) divide by `q_hat_m`, and `Psi_t` carries the same prior factor. A
  truncated transform belonging to a different density than the one in `Psi_t` would leave
  `F_m(+inf) != 1`. Setting both scales equal preserves the variance, `b^2 + b^2 = b_tilde^2`,
  and every formula already takes two scales per coefficient, so only the truncated-transform
  path branches. `b_xi = sqrt(eps/2)` is fixed while `b_{t-1}` shrinks toward steady state, so
  the crossing does occur in a run and the guard is not optional.
- **Rejected**: clamping the scale separation (nudging `b_xi` away from `b_{t-1}`). Three
  lines instead of twenty-five and bounded precision loss just the same, but it perturbs the
  model rather than evaluating it, which is not acceptable inside a reference filter.

### Simulation setup

- **Decided**: the small setup of the Laplacian-likelihood cells of
  `bayes-adaptive-filters.ipynb` — `L = 3`, normalized sinc `ho`, `AR = [1.0, -0.6, 0.85]`,
  `N = 200`, `NR = 10`.
- **Why**: `LF (exact)` is O(M) with a large constant (about 43 ms per step at M = 3, some
  20-28 inversions per coefficient per step). `NR = 10` costs about two minutes and is what
  makes the learning curves readable; at `NR = 1` the EMSE plot is single-realization noise.
- **Rejected**: the Section 7 room, `M = 128` with an abrupt change. That is a separate task,
  and at `M = 128` a full Monte Carlo is impractical without Appendix B's pointwise-quadrature
  fallback.

### Noise environment

- **Decided**: copy `MC_Simulations_Modular_Variance` into the notebook with
  `laplacian_noise_behavior` in place of `std_behavior`, renaming the noise argument to `b_v`.
- **Why**: `MC_Simulations_Modular_Variance` hardcodes `std_behavior`, whose noise is
  Gaussian, while Section 6 assumes a Laplacian likelihood. `laplacian_noise_behavior` already
  exists in `filters.py`. Its argument is a Laplacian *scale*, not a variance, hence the rename.
- **Rejected**: importing the driver unchanged, as the existing notebook's Laplacian cells do.
  It leaves the generated noise mismatched to the likelihood the filter assumes. Also rejected
  running both drivers, which doubles the runtime for no new information.

### Baselines

- **Decided**: compare against `sKF_L_exact_algorithm` (Section 5) and `sKF_L_algorithm`
  (Section 4), all three given the same spread at `t = 0` through `v = 2 b^2`.
- **Why**: `sKF-L (exact)` differs from `LF (exact)` only in the prior, which is exactly the
  question Section 7.5(v) poses; the minorized filter completes the Table 2 progression.
- **Rejected**: adding `sKF_L_integral_algorithm`. It is slow, and its prior is Gaussian, so
  it is a numerical reference for Section 5 rather than for Section 6.

### Return convention

- **Decided**: `LF_exact_algorithm` returns the Laplacian scale `b_t` in the `'v'` slot, not a
  variance.
- **Why**: the Monte Carlo drivers written for the Gaussian filters then take it unchanged.
  Table 1's `v = 2 b^2` converts where a comparison is wanted; the scale plot does exactly that.
- **Rejected**: returning `2 b_t^2` so that the slot always holds a variance. It would hide
  the projected quantity the Laplacian family actually carries.

### Regressor regularization

- **Decided**: `x_reg = sign(x)*(|x| + 1e-3)`, as `sKF_integral_algorithm` already does.
- **Why**: the mean (84) divides by `x_{t,m}`, so an entry must not sit on zero.
- **Note**: this is a floor, not a discretization error — it does not shrink as the quadrature
  is refined. The caveat recorded in the `sKF_L_exact_algorithm` docstring of `filters.py`
  applies here too.

---

## 2026-09-10 — Deviations from the draft, made while implementing Section 6

These are places where the notebook does not follow `main.pdf` as printed. Each needs a
decision in the draft itself; the code currently takes the reading that is self-consistent.

### Eq. (93) omits the `g_n` factors

- **Decided**: use `K_t(gamma) = log Psi_t(i gamma)`, i.e. including the
  `g_n = b_xi*|x_{t,n}|` terms.
- **Why**: (93) as printed sums only over `b_eta` and the `c_n`, but `Psi_t` in (80) also
  carries a `g_n` factor per coefficient, so (93) is not the cumulant generating function of
  the predicted residual it is described as. The saddlepoint condition has to be solved for
  the function actually being inverted.
- **Rejected**: following (93) literally. It would place the contour at the wrong point and
  degrade the conditioning the tilt exists to fix.

### The quadrature spacing must account for the tilt

- **Decided**: node spacing from `1/max(scales) - |gamma|`, not `1/max(scales)`.
- **Why**: the poles of `Psi_t` sit at `w = +-i/s_i`, so moving the contour to `gamma` moves it
  toward the nearest singularity and narrows every Lorentzian in the same measure. This was a
  real bug during implementation: with the naive spacing the grid held about two nodes across
  the peak and the *tilted* route was the inaccurate one, disagreeing with the plain axis by
  1.8e-5 at `e_t = 2` where the two should agree to machine precision.
- **Note for the draft**: Section 6.3 does not mention this. Separately, the node counts the
  notebook's rule produces are far above the 1e3 quoted in Section 6.3 (6099 nodes at
  `e_t = 0.3`, M = 3). The rule here is deliberately conservative — extent to machine
  precision relative to the tilted peak, ten nodes across the narrowest feature, and an
  allowance for the `S^3` growth of the `T_m/q_hat_m` multiplier — and was not tuned down, so
  this is not evidence that the draft's figure is wrong.

### The (87) guard's scope

- **Decided**: see "Confluent limit (87)" above — applied globally.
- **Note for the draft**: Section 6.2 says only that "the prior factor already present in
  `Psi_t` must be replaced by its truncated counterpart", which reads as a statement about the
  truncated transform alone. It needs to say that the confluent substitution replaces the
  prior everywhere it appears.

---

## Open

- Section 7's room, `M = 128`, AR(1) with -0.9, abrupt change at `Tc`. Needs Appendix B's
  pointwise-quadrature fallback before `LF (exact)` is affordable there.
- Question 7.5(v) is not settled by `section6.ipynb`. With a 3-tap normalized sinc `ho`, no tap
  near zero, the sparsity-promoting prior has nothing to exploit and the three filters nearly
  coincide. A sparse `ho` and a larger `M` are needed to put the question properly.
