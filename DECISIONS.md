# Decisions

## 2026-09-16 - Posterior vs error notebook (`posterior-vs-error.ipynb`)

**Decided**
- The setup from the meeting with Leszek ("w = 0, Sm = 0, Xm = 1") is read as w_{t-1} = [0,0,0]
  and x_t = [1,1,1]. So S̄_m = 0, x_m = 1 and σ_m = sqrt(2ṽ) > 0, and eq. 56 is used in full.
  ṽ and b_η are the values from gain-functions.ipynb.
- The "saturation error" is the first e at which the posterior mean reaches 99 % of a_m = ṽ x_m / b_η.
  The slider runs from 0 to 1.25 times that value.
- The likelihood is eq. 56 written directly in logs (log_ndtr, logaddexp). The log prior and log
  likelihood are added, the max is subtracted, then exp, then division by the area.

**Why**
- σ_m = 0 (no interference) would turn eq. 56 into a plain Laplacian and make eq. 57 undefined,
  so there would be nothing to check.
- The mean only reaches a_m in the limit, so saturation needs a threshold.
- Logs computed directly cannot underflow, and eq. 56 overflows if evaluated term by term.

**Rejected**
- Copying the FFT likelihood of `sKF_L_integral_algorithm`: it does not use eq. 56, which is what
  had to be implemented.
- `exp(log(pdf) + log(pdf))` as in filters.py:860: it multiplies pdfs that were already evaluated,
  so it gives no protection against underflow.
- A closed-form saturation heuristic (κ₊ = 3) or a fixed multiple of b_η: neither is tied to how
  close the mean actually is to a_m.

## 2026-09-17 - SNR sweep target in `04_escenario_realista_large_k.ipynb`

**Decided**
- The sweep target is -(SNR + 15) dB, not a fixed -20 dB. It is still -20 dB at 5 dB, so the sweep
  keeps reusing the run of section 4.
- The epsilon grid moves with v_eta: `EPS_GRID_LAPLACIAN*10**(-(snr - SNR_DB)/10)`, one decade per 10 dB.
- The sweep scans and verifies at N_SWEEP = 48000 instead of N_SEARCH = 24000.
- The Figure 5 cell skips a filter whose search returned no curve, instead of crashing.

**Why**
- As epsilon grows, both Laplacian updates become NLMS with mu = 1, so no epsilon gives a floor
  shallower than about -SNR dB. Checked up to epsilon = 1: the floor stops changing, and both filters
  give the same floor and steps. At 20, 25 and 30 dB every floor is below -20 dB, so `pick_epsilon`
  returned "grid too narrow" (nan in Figure 4, None curve in Figure 5).
- Below 20 dB a fixed -20 dB target sits closer to that common NLMS floor the higher the SNR, where
  the two filters coincide. That pushes the exact/minorized step ratio toward 1 regardless of the
  filters, so the old "advantage shrinks with SNR" trend was at least partly caused by the target.
- At the deeper targets the filters take up to ~16000 steps (30 dB), more than half of 24000.
  At 48000 every chosen epsilon settles within half the run (scan at N = 96000, truncated).

**Rejected**
- Wider epsilon grid at -20 dB: the floor stops changing at large epsilon, so no grid reaches the target.
- Keeping -20 dB and dropping 20-30 dB: cheap, but leaves the target-depth confound in the 0-15 dB points.
- min(-20, -(SNR + 5)): fits the old grid, but mixes two target definitions in one figure.
- N_SWEEP = 96000: about twice the cost for no change in which epsilon settles.

**Result**
- All seven SNRs return "ok". Steps exact/minorized: 1.33, 1.34, 1.32, 1.25, 1.23, 1.13, 1.13 at
  0-30 dB (was 1.63 at 0 dB and 1.03 at 15 dB with the fixed target). Floor gap 0.05 dB or less.
- Floors at 48000 are within 0.06 dB of those at 96000, steps within 1%.
- At the same epsilon/v_eta the floors are exactly 10 dB apart per 10 dB of SNR: both updates are
  invariant under noise -> c noise, b_eta -> c b_eta, eps and v -> c^2. Only w0 = 0 and v0 = 2
  break that, i.e. the transient. So across this sweep the ratio changes through the transient only.
- 5 dB still comes from section 4 (search at N = 24000, run at N = 96000, R = 20), a different
  protocol from the other SNRs; its ratio lies between its neighbours.

## 2026-09-17 - Follow-up to Leszek's reply (`normalized-gain-check.ipynb`, sections 5 and 6)

**Decided**
- k is Leszek's: k = √ṽ‖x‖/b_η. Sections 1-4 keep their old k (= his k²); section 5 states the
  mapping and gives u* in his k. The earlier sections were not edited.
- The median-step gain ratio compares the effective step χ/e, each filter at its own median step
  (own τ and k), instead of both gains at the exact filter's operating point.
- Variance-recursion test: fixed-variance versions of both Laplacian filters with ṽ_t frozen at a
  constant, and that constant tuned on logspace(-6, -1, 11) in place of ε. Everything else is the SNR
  sweep of `03_escenario_realista.ipynb`: same signals and seeds (generated at N = 96000), M = 128,
  -20 dB target, R = 3 search, R = 10 check, floor from the last quarter, 3 dB criterion.
- The sKF baseline is rerun under the same protocol, including 5 dB, and everything runs at
  N = 24000 and N = 48000.
- The filters are vectorised over grid values and realisations (same update lines as section 4).

**Why**
- Leszek confirmed his definition; renaming inside the earlier sections was not requested.
- At equal floor the two filters settle at different τ (minorized 1.3-2.1 times the exact one).
- Freezing ṽ_t is the only change between the sKF and fKF rows, so any ratio change is due to the
  variance recursion.
- Notebook 03's 5 dB ratio came from a different N and R than its other SNRs.
- At 0 dB the exact fixed-variance filter converges at about 11000 of 24000 steps, so settling
  needed a check.
- Vectorised, the whole sweep takes about 2 min; the scalar loop would take several times longer.

**Rejected**
- Reusing notebook 03's ratios as the baseline: 5 dB was not on the same protocol.
- Only N = 24000: could not rule out unsettled floors at 0 dB.

**Result**
- Steps exact/minorized at 0 dB: 1.63 with the recursion, 1.54 frozen (1.56 / 1.55 at N = 48000).
  The variance recursion does not explain the gap to the analysis (1.13). Remaining differences:
  input (AR(1) vs white), M (128 vs 16), floor reading.

## 2026-09-21 - The convergence gap, 1.56 against 1.13 (`convergence-gap.ipynb`, issue #12)

**Decided**
- Re-implement Section 6 of `main_minorization.tex` (eqs. `steady.balance`, `variance.recursion`,
  `chi.four`) rather than take its 1.13 as given. Validated against its own published numbers before
  being used: 478.3 / 486.6 / 706.2 iterations per coefficient at J = -40 dB against its 479 / 487 /
  707, and tau_min/tau_exact 1.449 against 1.45.
- Everything in this notebook stays at the old `b_eta = sqrt(v_eta/2)`. Issue #9 switches the rest.
- Expectations over the noise by Gauss-Legendre on the **probability scale**,
  E[f(eta)] = int_0^1 f(F^{-1}(p)) dp, and Gauss-Hermite in the a priori error.
- The filters compared at the misadjustment J = sigma_a^2/v_eta each run actually reaches, recorded
  per run, not at the nominal -20 dB misalignment target.
- M = 128 uses notebook 03's `ho` exactly; only M = 16 moves the window onto the direct path.

**Why**
- The ratio is a function of the target, not a constant: 1.03 at J = -10 dB to 1.69 at -50 dB. No
  single number is "the prediction" without its target, so the comparison had to be made at matched J.
- Leszek's 1.13 was computed at the old convention, stated twice inside Section 6
  (`main_minorization.tex:2588` and `:2619`), and so were our simulations. Switching first would have
  destroyed the matched target.
- chi is bounded, so on the probability scale the integrand is flat where the quantile blows up. A
  grid in eta is not usable at beta* = 0.2.
- `rir.generate(nsample=16)` returns all zeros: the source-microphone delay is 33 samples at 8 kHz.

**Rejected**
- Taking 1.13 from the draft and only running simulations: it would have left the target-dependence
  invisible and the comparison unfalsifiable.
- Tabulating E[chi^2] and E[e^a chi] on a grid in sigma_a^2 to speed up the deep targets: brentq on
  log sigma with a warm-started fixed point brings the whole notebook to 97 s, so it was not needed.
- `room_taps` from the direct path at M = 128 as well: it silently returned 95 taps and moved the
  scenario off notebook 03's.

**Result**
- **1.13 is the analysis at J = -20 dB** (1.132 reproduced). The quoted row 1.13 / 1.05 / 1.00 / 0.99
  at 0 / 5 / 10 / 15 dB SNR is one curve read at J = -20, -15, -10 and -5 dB, a fixed misalignment
  target becoming a shallower misadjustment as the SNR rises.
- **M is not the explanation.** White input gives 1.105 at M = 128 and 1.124 at M = 16, both on the
  analysis (1.13). A random h at M = 16 gives 1.155, so h alone moves it by 0.03.
- **The floor reading matters only where it moves J.** AR at M = 16 settles at J = -27 dB, where the
  analysis gives 1.251 against 1.284 measured.
- **The gap is the input.** At M = 128, target held, J within 1.3 dB: ratio 1.105, 1.240, 1.305, 1.546
  as cond(Rxx) runs 1, 9, 32, 347.
- Simulation and analysis agree wherever the analysis is valid. Section 6 excludes correlated
  regressors itself: the recursion "no longer closes on sigma_a^2 alone".

## 2026-09-21 - vKF and KF from the joint posterior (`vkf-kf.ipynb`, issues #10 and #11)

**Decided**
- Two forms of each filter. The **scalar** form carries the shared signature
  `filter(n, x, d, w0, parameters) -> {"w_hist", "e"}`, so `run_filter` / `sweep_filter` of notebook 07
  take it unchanged; a **batched** form runs B filters side by side for the sweeps only. The two are
  checked against each other every run (agree to 1e-10 relative).
- `chi_laplacian` / `log_mills` copied read-only from Ignacio's notebook 07, commit `9406077`, with a
  source comment, as notebook 03 already does for the minorized filter. Not lifted into a shared
  module: that is Ignacio's file and his call.
- Guards: `Sigma = (Sigma + Sigma.T)/2` each step, and `v_m` clipped at zero in the vKF.
- Correctness established before any result: KF with the Gaussian chi against a textbook Kalman
  filter written independently of eqs. (33)-(34); all three families against each other on the first
  update from a scalar prior; chi' in [0, 1] over the operating range.
- `k_t` median and 10-90 % range reported in every table.
- The KF tested against the white-input sKF as the primary reference, and against an explicitly
  prewhitened sKF as a secondary one.

**Why**
- The shared signature is what lets this half merge with Ignacio's; the batched form is what makes the
  KF sweeps affordable (36 us/step/realisation against 50 us/step scalar at M = 128).
- chi' <= 1 holds for any noise and chi' >= 0 for log-concave noise, so the Laplacian case is safe in
  exact arithmetic; the guards are against roundoff only, and both were confirmed inactive.
- Prewhitening is not like-for-like: A(z) = 1 - a z^-1 colours the noise, raising its variance by
  1 + a^2 = 1.81 and costing 1.7 dB of effective SNR. It is reported with that caveat, not as the
  reference.

**Rejected**
- Lifting the chi functions into a shared module now, as the split plan calls for: Ignacio is still
  working in that file. To be cleared with him first.
- Batching only, with no scalar form: it would have broken the shared signature.
- Prewhitening as the primary reference for "recovers the loss": its SNR and noise are not the same.

**Result**
- **vKF buys nothing here.** 5659 steps against the sKF's 5787 under AR(-0.9), 1508 against 1511 under
  white input, same k_t to three decimals. The regressor is a sliding window of one stationary
  process, so every tap sees the same statistics and the v_{t,m} stay nearly equal.
- **The KF recovers the colour penalty in full, without prewhitening.** The sKF pays 3.83x for the
  AR(-0.9) input (5787 against 1511 on white); the KF takes 1521, within 1 % of the white-input sKF,
  giving back 99.8 %. Prewhitening gives back 74 %, at 1.7 dB of SNR and the i.i.d. noise.
- The KF is also 2.3x faster than the sKF on white input (651 against 1511), and reaches the same
  floor from a smaller and steadier k_t (median 0.508, 10-90 % 0.460-0.559, against 0.938 and
  0.750-1.180).
- Caveat: both KF rows and the white-input sKF and vKF report "optimum at grid edge". The selected
  epsilon is interior and the target bracketed, so the interpolation is sound, but a target deeper
  than -20 dB would need notebook 03's grid extended below 1e-8.

## 2026-09-21 - Adopting b_eta = E|eta_t| (issue #9)

**Decided**
- `b_eta` is a parameter of every filter and is never hardcoded; `b_eta_of(var_eta, convention)`
  gives either `"variance"` (sqrt(v_eta/2)) or `"mad"` (E|eta_t|).
- The vKF / KF sweeps are reported under **both** conventions for one run, so the size of the change
  is on the record.
- `convergence-gap.ipynb` stays at the old convention and says so.

**Why**
- E|eta_t| is the maximum-likelihood and KL-closest Laplacian fit to the noise actually simulated, and
  it is the same principle Section 2.3 already uses to project posteriors onto a family. Verified:
  closed form 0.25087 against 0.25108 sampled over 4e6 draws at beta* = 0.2.
- Issue #12's target was matched at the old convention, so that notebook had to stay there.

**Rejected**
- Switching before the filters existed: the code change is trivial, the sweeps are the expensive part.
- Running everything under both conventions everywhere: one run on the record is enough, since the
  change turns out to be uniform across families.

**Result**
- b_eta 0.35947 -> 0.12754 at the notebook 03 operating point, so every k_t grows by 2.82x
  (0.938 -> 1.496 for the sKF, 0.508 -> 0.848 for the KF).
- Every filter converges about 18 % faster to the same floor: 0.82, 0.83, 0.82 for the sKF, vKF, KF.
  The change is uniform, so nothing in the comparison between families moves.
- beta* = 1 needs no re-run: E|eta| = sqrt(v_eta/2) is an identity for the Laplacian (ratio 1.0000).
- **Notebook 03 re-run** (`03_escenario_realista_mad.ipynb`, R = 20, N = 96000, SNR 5 dB). The
  Gaussian row is byte-identical (eps 3.16e-08, floor -19.71, 53188 steps), as it must be since that
  filter never sees b_eta: a clean control that nothing else moved. The minorized filter is
  unchanged, 4329 -> 4332 steps, because b_eta enters it only through tau = v||x||^2/b_eta, so
  re-tuning epsilon undoes the change exactly, as Section 6 of `main_minorization.tex` states. The
  whole effect lands on the exact filter, 5787 -> 4729 steps, and the ratio exact/minorized at 5 dB
  falls from 1.337 to 1.092.
- Cross-check: the batched sKF of `vkf-kf.ipynb` reproduces notebook 03's scalar filter to the step
  under both conventions, 5787 and 4729. Two independent implementations, same numbers.
- **Notebook 04 re-run** (`04_escenario_realista_large_k_mad.ipynb`, SNR sweep). The result does not
  merely shrink, **it changes sign**:

  | SNR | 0 | 5 | 10 | 15 | 20 | 25 | 30 dB |
  |---|---|---|---|---|---|---|---|
  | ratio, b = sqrt(v/2) | 1.33 | 1.34 | 1.32 | 1.25 | 1.23 | 1.13 | 1.13 |
  | ratio, b = E\|eta\| | 1.11 | 1.09 | 1.06 | 1.01 | 0.99 | 0.96 | 0.93 |

  Above about 20 dB the exact filter is now the **faster** of the two. The minorized steps barely
  move across the sweep (within a few percent, the residue being the transient from w0 = 0 and
  v0 = 2, which breaks the tau-invariance); the exact filter gains 15 to 21 % at every SNR. Floors
  match to 0.00 dB against -0.04 dB before.
- So "the exact filter converges more slowly" was a statement about the variance-matched scale, not
  about the filter. Under the scale the new draft uses, it holds only below about 20 dB SNR. This is
  worth putting to Leszek with the table, since it changes what Section 6 concludes.
- Still to do under the new convention: Ignacio's notebooks 05 and 07.

## 2026-09-22 - Math notation in notebook markdown cells

**Decided**
- Notebook markdown writes vectors and matrices out in full, `\boldsymbol{x}`, `\boldsymbol{\Sigma}`,
  matching `gain-functions.ipynb`. The draft's preamble shorthands (`\bx`, `\bw`, `\bh`, `\bSigma`,
  `\bI`, `\bR`, `\btheta`, `\bzero`) are not used in notebooks. Applied to `vkf-kf.ipynb` (18 broken
  spans) and `convergence-gap.ipynb` (10), which had been written in the draft's notation.
- Accents take an explicit group: `\tilde{\boldsymbol{\Sigma}}`, not `\tilde\boldsymbol{\Sigma}`.
- Also unescaped the doubled backslashes in cell 0 of `03_escenario_realista_mad.ipynb` and
  `04_escenario_realista_large_k_mad.ipynb` (`$b_\\eta$` -> `$b_\eta$`), 8 per file.

**Why**
- Notebook markdown is rendered by KaTeX, which has no access to the LaTeX preamble, so every one of
  those macros rendered as red "Undefined control sequence" text in VSCode, GitHub and nbviewer.
- KaTeX gives `\tilde` only the next single token, so it swallows `\boldsymbol` and then fails on the
  missing argument. LaTeX expands the macro first, which is why `\tilde\bSigma` is fine in the .tex
  and its expansion is not.

**Rejected**
- Defining the macros in a `\newcommand` block per notebook: KaTeX's macro table is not reliably
  shared across notebook cells, so the definitions would have to be repeated in every cell.
- `.vscode/settings.json` with `markdown.math.macros`: editor-only, so the math would still break on
  GitHub and nbviewer, and that setting is documented for the markdown preview, not for notebooks.

## 2026-09-23 - fKF from the joint posterior, and the matched fKF across a change (`fkf.ipynb`)

**Decided**
- A new notebook, `fkf.ipynb`, rather than cells added to `vkf-kf.ipynb`: the comments that closed
  #10 and #11 cite that notebook's outputs, so it stays as it is.
- Three fKFs, differing only in the correction: minorized, exact (joint, eq. `fKF.mean`) and matched (the
  generalized Gaussian at beta* and its true scale, via `chi_quadrature`). The minorized one is included
  because Leszek's 2.14x / 2.22x compare the matched fKF against both Laplacian filters.
- Code copied read-only with its source commit, as Ignacio decided on #8 (no shared module):
  `chi_laplacian` from notebook 07 (`9406077`), the quadrature from notebook 08 (`0ed92a4`), the change
  test from notebook 09 (`9afa027`), the marginal fKF-L from notebook 04 (`db4b3a9`).
- At rest: the protocol of `vkf-kf.ipynb` (search R = 3, N = 24000; checked run R = 20, N = 96000), `v`
  swept over logspace(-6, -1, 11), both `b_eta` conventions. The sKF runs through the same code as a
  control and must land on 5787 / 4729 steps.
- Across the change: notebook 09's test (2N steps, h -> -h at N, recovered = within 3 dB of the floor at
  rest), at `b_eta = E|eta|` only, each filter at its `v` at rest.
- A table of the time to reach 0, -5, -10 and -15 dB after the change, beside the 3 dB criterion.
- Plot colours from a palette checked for colour-blind readers.

**Why**
- The joint fKF's result was predictable (Gamma = 0, so it must be notebook 04's filter); the new
  information is its `E|eta|` row and the matched run.
- The `vkf-kf.ipynb` protocol keeps the fKF comparable with the sKF / vKF / KF, and costs about 2.7e6
  quadrature calls against about 2.1e7 for notebook 07's full-grid protocol. The whole notebook runs in
  3.7 min.
- The 3 dB criterion reads recovery at one level, and the curves showed the matched fKF behind early and
  ahead late, so a single number would have hidden the result.
- Matplotlib's default orange and green are almost the same colour for red-blind readers (Delta E 0.7);
  the replacement passes (Delta E 9.2).

**Rejected**
- Notebook 07 / 09's protocol (every grid point at R = 20, N = 96000): about 8x more quadrature calls for
  the same kind of checked run.
- Rewriting `chi_quadrature` to run over batches: 28 us per call, and the notebook already runs in minutes.
- A Gaussian fKF: notebook 04 needs about 400000 steps for it, and it is not part of either question.
- The two setup tests suggested on #8 (white input; change after 6000 samples): not in the scope chosen.

**Result**
- Checks: joint fKF = notebook 04's marginal fKF-L to 7e-16; minorized = notebook 04's to 6e-16; quadrature
  at beta = 1 = closed form to 7e-16; batched = scalar to 1e-15. Control: 5787 and 4729 steps, exact.
- At rest, exact over minorized: **1.32x** at `sqrt(v/2)` (notebook 04: 1.32x), **1.05x** at `E|eta|`. The
  minorized fKF does not move with the convention (5168 -> 5162); the exact one gets 21 % faster
  (6842 -> 5422). The exact fKF is slower than the exact sKF (5422 against 4729).
- At rest, the matched fKF is fastest: **0.76x** the exact, **0.80x** the minorized.
- Across the change, at the 3 dB level, the matched fKF recovers slightly **faster**: **0.90x** the exact,
  **0.94x** the minorized, against Leszek's 2.14x / 2.22x. The draft's 1.2-2.2x does not hold here for the
  fKF either.
- **It depends on the level**: matched / exact is 1.44 at 0 dB, 1.17 at -5, 1.04 at -10, 0.95 at -15 and
  0.90 at floor + 3 dB. The matched fKF is slower while the error is large (the redescending score reads it
  as outliers, the mechanism Section 4 describes) and faster near the floor.
- Even at 0 dB, 1.44x is below Leszek's 2.14x, so the level does not explain his number; his setup must
  (white input, random response, 6000 samples before the change, a moving-average reading).

## 2026-09-24 - The convergence gap at both b_eta conventions (`convergence-gap.ipynb`)

**Decided**
- The notebook now runs at both conventions: the ratio against target J (section 4), the four
  configurations, the comparison at the J each run reaches, and the AR sweep (sections 5-6). The
  validation against the draft's 479 / 487 / 707 stays at sqrt(v/2) only, because the draft computed
  those numbers there. Every sqrt(v/2) result is kept, and each one reproduces to the last printed digit.
  This reverses the 2026-09-21 decision to keep this notebook at the old convention, as additive only.
- `b_eta_of` copied from `vkf-kf.ipynb` (cell 4), with a source comment. `B_ETA_A` is a dict keyed by
  convention; `run_configuration` and `analysis_at` take the convention as a parameter.
- Under E|eta| the simulation's v~ grid is multiplied by b_eta/sqrt(v_eta/2), so it is the same grid in
  tau = v||x||^2/b_eta under both conventions. The factor is exactly 1.0 under the old one.
- Figure: one colour per convention (#2a78d6 and #eb6834, the pair from `fkf.ipynb`), one marker shape
  per configuration, filled for white input and open for AR(-0.9). The left panel has two legends so
  that neither covers a point or a curve.
- `sigma_for_target` keeps its bracket (1e-3, 1e2).

**Why**
- Section 5 of the new draft (`main.tex:1576`, a placeholder) is this analysis, stated at E|eta|. The
  old draft's 1.13 and the #12 record are at sqrt(v/2). Both are needed, and neither replaces the other.
- b_eta enters the minorized filter only through tau. On a common tau grid the minorized search
  picks the same tau and runs the same filter, so its row becomes a control, and any change in the
  ratio comes from the exact filter. With a fixed v grid the minorized pick would have moved by
  interpolation error alone, and that error would have leaked into the ratio.
- Colours: the CIE76 Delta E between the two is at least 97 under simulated protan, deutan and
  tritan vision (Machado 2009 matrices). The old pair, COLORS[1] / COLORS[3] (orange / red), was 29-41.
- Bracket: at sigma = 1e-3 the steady state is at J = -76 dB and at 1e2 at -0.2 dB under E|eta| too,
  so every target here is bracketed. The acceptance check run with (1e-4, 1e2) gave the same step counts.

**Rejected**
- Switching the notebook to E|eta| only: it would have dropped the matched comparison with the
  draft's 1.13 and the record of #12.
- The same v grid under both conventions: see Why.
- Keeping colour for white vs AR and adding a line style for the convention: eight colour/fill/shape
  combinations on one panel, and the default orange/red pair is weak for colour-blind readers.

**Result**
- Acceptance check reproduced exactly, steps exact/minorized at sqrt(v/2) / E|eta|: 1.025 / 0.950,
  1.074 / 0.975, 1.132 / 1.012, 1.213 / 1.072, 1.305 / 1.138, 1.499 / 1.286 at J = -10, -15, -20, -25,
  -30, -40 dB. Minorized steps identical under both at every J.
- Analysis at E|eta|: 0.950 at -10 dB up to 1.439 at -50 dB. It crosses 1 between -15 and -20 dB,
  so at shallower targets the exact filter is the faster one. The exact filter takes 7 % (at -10 dB)
  to 15 % (at -50 dB) fewer steps than at sqrt(v/2). The quoted row 1.13 / 1.05 / 1.00 becomes
  1.01 / 0.98 / 0.95.
- Simulation at E|eta|: the minorized filter does not move (same steps in every configuration, floors
  within 3.6e-15 dB). White input gives 1.009 (M = 128) and 1.020 (M = 16), against 1.012 from the
  analysis. AR at M = 16 gives 1.106 at -26.9 dB against 1.097.
- **"The gap is the input" holds under E|eta| too, but the gap is much smaller.** AR(-0.9) at M = 128:
  1.141 against 1.000 from the analysis at its J, an excess of 0.14 where sqrt(v/2) gave 0.44
  (1.546 against 1.110). AR sweep: 1.009, 1.045, 1.048, 1.141 as cond(Rxx) runs 1, 9, 32, 347
  (was 1.105, 1.240, 1.305, 1.546). The step from 9 to 32 is within the protocol's noise. The exact
  filter gains more from the new scale under coloured input (10943 -> 8080 steps) than under white
  input (1568 -> 1432).
- Cross-checks at 5 dB, white, E|eta| (J = -15 dB, analysis 0.975): Ignacio's notebook 05 fKF-L
  1054 / 1057 = 1.00 (`ignacio/joint-vs-marginal-vs-minorized`, `4e3102a`); Leszek's
  `sims/sep18/ggb_summary_ar0.0.json` fLe / fLm 936 / 947 = 0.99. The 0 dB row of Ignacio's SNR sweep,
  1431 / 1419, matches this notebook's white M = 128 run (1432 / 1419) to within one step.
- Run time 175 s (was 102 s).
