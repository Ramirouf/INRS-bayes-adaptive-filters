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
