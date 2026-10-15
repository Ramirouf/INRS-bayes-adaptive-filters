# For Ignacio: our bank against the review's convex combination

Ignacio, this is where the comparison stands, so that you can pick it up. Written on 2026-10-15. Ramiro picked the test,
and Claude (Claude Code) set the details, wrote and ran the notebook while Ramiro was away. **Ramiro has not reviewed
the results yet.**

## Why this exists

At the last meeting Ramiro showed `kf-bank-q-sweep.ipynb` and `awu-gisv-vs-banks.ipynb`. Augusto said that banks of
filters already exist, and Leszek asked for references and a comparison. Augusto sent:

> J. Arenas-García, L. A. Azpicueta-Ruiz, M. T. M. Silva, V. H. Nascimento, A. H. Sayed, "Combinations of adaptive
> filters: performance and convergence properties", IEEE Signal Processing Magazine, Jan. 2016,
> DOI 10.1109/MSP.2015.2481746.

He suggested its convex and affine combinations with the "weight sharing" schemes, gradual transfer first. The PDF is not
in the repo (IEEE copyright); ask Ramiro or Augusto for it.

Our bank is the IMM estimator (Blom and Bar-Shalom, 1988). With the Gaussian chi, Algorithm 3 is a textbook IMM to
2e-15 (`kf-bank-q-sweep.ipynb`, check 4.5). The review is the other line of work: it learns one mixing weight by a
gradient on the squared output error, and shares weights by rules set by hand. The draft should cite both.

## What was run

`cvx-combination-vs-bank.ipynb`. The same two sKFs (exact Laplacian chi) are combined in two ways, so that only the
combination layer differs:

- **the review's convex combination:** cvx-PN-LMS (its Table 4), with each sharing scheme of its Table 5 (none, gradual
  transfer, simple copy, feedback);
- **our bank:** Algorithm 1 with K = 2.

The two sKFs:
- **slow:** the exact sKF at -20 dB, ε = 1.35e-6;
- **fast:** ε = 1e-4, the top of the bank grid.

Scenario: notebook 09's change test with AR(-0.9) input, with β* = 0.2 and with Gaussian noise (β* = 2). Ramiro asked
for the Gaussian run because there the review's squared error is the right cost.

Every combination setting was run (μa, λ0, and ℓ or N0), and so was every q for our bank. Each scheme is reported at its
best mean over the record, as we did for the banks' q. The tables in sections 6 and 7 print every setting.

## What came out

Mean over the record, misalignment, mean curves [dB]:

| | β* = 0.2 | β* = 2 |
|---|---|---|
| slow sKF alone | -14.08 | -6.77 |
| combination, no sharing | -14.68 | -9.14 |
| combination, gradual transfer or simple copy (best) | -15.00 | **-12.23** |
| our bank of the two, best q | **-15.28** | -10.49 |
| our bank of six, best q (other filters) | -16.52 | -10.53 |

1. **With heavy-tailed noise our bank wins, but only by 0.3 dB.** It also gets back to -15 dB 6 % sooner after the
   flip.
2. **With Gaussian noise the combination wins by 1.7 dB.** At rest our bank keeps 13 % of its weight on the fast sKF,
   so its floor is -12.2 dB, against -18.6 dB for the combination. The likely cause is the Laplacian density
   in our weights under Gaussian noise. That is not checked; it is the first thing to test.
3. **Our bank gains only through the mixing step.** At q = 0 it is exactly the slow sKF in both noises: the fast sKF's
   weight is 0.000 before and after the flip.
4. **The review's threshold barely lets its sharing act here.** With λ0 = 0.98 (close to its Fig. 8 value), gradual
   transfer and simple copy almost never fired after the flip. They only help with λ0 ≤ 0.9 and μa ≥ 1. Gradual
   transfer and simple copy behave as one scheme; feedback is the weakest of the three.

## Things to know before you continue

- **The grid grew twice during the work.**
  - First, λ0 = 0.98 alone. Then 0.9 and 0.7 were added, because the schemes were not firing.
  - Then μa = 3 and λ0 = 0.5, because the best settings sat at the edges.
  - At β* = 0.2 the best is now inside the grid. At β* = 2 it is still at an edge, but the top is flat: five other
    settings are within 0.15 dB of the best.
- **Table 5 was implemented from the review only.** Papers [47]-[49] were not read, so details they add may differ, for
  example the order of the update and the transfer. Only w is shared; each sKF keeps its own v.
- **λ_t is erratic in the first steps.** While the two filters are nearly equal, the cvx-PN-LMS step is huge, so a_t
  jumps between ±4. One unit in the last place can flip it. The output does not care, because w1 ≈ w2 at those moments
  (check 4.2).
- **Under Gaussian noise our filters keep the Laplacian likelihood,** as in `kf-bank-q-sweep.ipynb`. So the β* = 2 run
  favours the combination.

## Suggested next steps, in order

1. **Our bank with the Gaussian density in its weights under Gaussian noise.** This is the like-for-like test for
   result 2.
   - `bank_step` uses one `scale` for both chi and the density. Either run everything Gaussian (`chi_gaussian`,
     `log_density_gaussian`, v_eta), or add a second scale.
   - `awu-gisv-vs-banks.ipynb` has a -20 dB search for the Gaussian sKF in this scenario, saved under
     `results/awu-gisv-vs-banks/search_base_ar-0.9_beta2.pkl`.
2. **More than two filters.** Compare the review's softmax combination of the six sKFs (its eqs. 22-23) with our bank
   of six. That is the comparison closest to our actual bank.
3. Optional:
   - the affine combination (aff-PN-LMS, Table 4);
   - KF components, since the KF bank is our best filter with AR input;
   - a robust mixing rule for impulsive noise. There are papers on sign-error and correntropy rules, but none of them
     were read.

## How to run

- The notebook reads its 350 runs from `results/cvx-combination-vs-bank/`, and ours from `results/kf-bank-q-sweep/`.
  Both folders are git-ignored, so they are not in the push.
- With both folders in place a rerun takes about 10 s. From nothing, the 350 runs take about 2.5 h on Ramiro's
  machine.
- Without `results/kf-bank-q-sweep/`, the notebook also recomputes the runs of ours it reads. That was not timed; a KF
  search of this kind took about 13 min in `bank-kf-excess-error.ipynb`.
- New settings get new file names, so adding settings to the grids only runs the new ones.
- The decisions and the full numbers are in `DECISIONS.md` (entry of 2026-10-15) and in section 9 of the notebook.
