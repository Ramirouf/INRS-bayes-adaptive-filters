"""Exact scalar-variance filter for a Gaussian prior with a Laplacian likelihood.

Implements the sKF-L (exact) filter of Section 5 of the draft, paragraph
"Scalar-variance filter (sKF-L, exact)". This is the closed form of the exact
marginal, so it is the counterpart of `sKF_L_algorithm` in the notebook, which
implements the *minorized* filter of Section 4 (eqs. 50 and 51).

Where the minorization replaces the Laplacian log-density by a quadratic and
lands back on a Kalman-like gain, the exact marginal stays a two-component
mixture, one component per branch of |e_t - x_m d|. Its first two moments close
in terms of four scalars that are the same for every coefficient m:

    Lambda_t, Gamma_t, P_t, Q_t

so the per-step cost is four evaluations of the normal CDF, independent of L.

Per step, with e_t = d_t - x_t' h_{t-1}, sigma = +/-1 and ||x_t|| the window norm:

    v~_t      = v_{t-1} + epsilon
    kappa_s   = (s*e_t - v~_t*||x_t||^2/b_eta) / (sqrt(v~_t)*||x_t||)
    pi_s      = softmax(-s*e_t/b_eta + log Phi(kappa_s))
    mills(k)  = phi(k)/Phi(k)                        (inverse Mills ratio)

    Lambda = pi_+ - pi_-                 Gamma = pi_+ mills_+ - pi_- mills_-
    P      = pi_+ mills_+ + pi_- mills_- Q     = pi_+ k_+ mills_+ + pi_- k_- mills_-

    h_t = h_{t-1} + (v~_t/b_eta * Lambda - sqrt(v~_t)/||x_t|| * Gamma) * x_t
    D_t = g^2 (1 - Lambda^2) - 2*g*l*(P - Lambda*Gamma) - l^2*(Q + Gamma^2)
          with g = v~_t/b_eta and l = sqrt(v~_t)/||x_t||
    v_t = v~_t + D_t*||x_t||^2 / L

Lambda_t lies in [-1, 1] and acts as a saturating prediction error: a single
observation can move each coefficient by at most v~_t*|x_t,m|/b_eta. That
bounded influence is the point of the Laplacian likelihood.

Signature and return value follow the convention of the other filters in
`bayes-adaptive-filters.ipynb`, so this drops into the same `Algorithms` list
and the same `sKF_L_params` struct as `sKF_L_algorithm`:

    (N, x, d, h0, parameters) -> {'h': h_hist,   (N, L) weights, before each update
                                  'y': y,        (N,)   predicted output
                                  'e': e,        (N,)   prediction error
                                  'v': v_hist}   (N, L) scalar variance, repeated

Depends only on numpy and scipy.
"""

import numpy as np
from scipy.special import log_ndtr, logsumexp

# Same layout as sKF_L_params in bayes-adaptive-filters.ipynb, repeated here so
# this module runs on its own. Either struct works: the fields are identical.
sKF_L_params = np.dtype([("label", "U20"),   # Unicode string up to 20 characters
                         ("epsilon", "f8"),  # process noise variance
                         ("b_eta", "f8"),    # Laplacian observation noise scale
                         ("v_tilde_0", "f8") # initial prior variance per weight
                       ])

# Leading underscores so that `from skf_l_exact import *` cannot shadow the
# notebook's own shift(), which is @njit and is called from compiled filters.
_SIGN = np.array([1.0, -1.0])  # the two mixture branches, sigma = +1 and sigma = -1


def sKF_L_exact_algorithm(N, x, d, h0, parameters):
  """sKF-L (exact), Section 5 of the draft. See the module docstring."""
  h = h0
  epsilon = parameters["epsilon"]
  b_eta = parameters["b_eta"]
  v_tilde_0 = parameters["v_tilde_0"]

  # No regularization of x here, matching sKF_algorithm and sKF_L_algorithm.
  # sKF_L_integral_algorithm does regularize internally, with
  # x_reg = np.sign(x)*(np.abs(x) + 1e-3), so the two routes see slightly
  # different inputs. Measured, that is worth about 0.2% of the weights
  # (max 2.9e-3 at epsilon=0.01, b_eta=50*sqrt(var_v)), well under the
  # integral-vs-closed gap at dx_factor=1/100, which is why it does not change
  # the comparison today. It is a floor, though, not a discretization error: it
  # does not shrink as the grid is refined. If the grid is ever taken far enough
  # that the gap drops below ~3e-3, pass x_reg here as well.

  L = len(h)
  y = np.zeros((N,))
  e = np.zeros((N,))
  xtemp = np.zeros(L)
  v = v_tilde_0
  h_hist = np.zeros((N, L))
  v_hist = np.zeros((N, L))

  for k in range(0, N):
    xtemp = _shift(x[k], xtemp)
    y[k] = h @ xtemp
    e[k] = d[k] - y[k]
    h_hist[k] = h
    v_hist[k] = v

    if k >= L:
      # predict
      v_tilde = v + epsilon
      # update
      norm = xtemp @ xtemp
      norm_x = np.sqrt(norm)

      # The two mixture arguments. These are global: the per-coefficient terms
      # cancel, so kappa does not depend on m.
      kappa = (_SIGN*e[k] - v_tilde*norm/b_eta) / (np.sqrt(v_tilde)*norm_x)

      # --- log domain, and it has to stay that way ---------------------------
      # Phi(kappa) underflows to 0 for kappa below about -38, and the two
      # unnormalized weights differ by a factor exp(2*e[k]/b_eta), which
      # overflows for a residual of a few b_eta. With Laplacian noise both
      # happen on the same steps: the outliers. Computing exp(-s*e/b_eta) and
      # Phi(kappa) separately and multiplying loses every digit exactly where
      # this filter is supposed to earn its keep. log_ndtr is accurate deep
      # into the left tail, and logsumexp normalizes without ever forming the
      # individual factors. Same reason the Mills ratio goes through logs.
      log_Phi = log_ndtr(kappa)
      log_pi = -_SIGN*e[k]/b_eta + log_Phi
      pi = np.exp(log_pi - logsumexp(log_pi))          # mixture weights, sum to 1

      log_phi = -0.5*kappa**2 - 0.5*np.log(2*np.pi)
      mills = np.exp(log_phi - log_Phi)                # phi(kappa)/Phi(kappa)
      # ----------------------------------------------------------------------

      Lambda = np.sum(_SIGN*pi)                         # saturating error, in [-1, 1]
      Gamma = np.sum(_SIGN*pi*mills)
      P = np.sum(pi*mills)
      Q = np.sum(pi*kappa*mills)

      gain = v_tilde*Lambda/b_eta - np.sqrt(v_tilde)*Gamma/norm_x
      h = h + gain*xtemp  # not h+= because it would mutate h0

      g = v_tilde/b_eta
      l = np.sqrt(v_tilde)/norm_x
      D = (g**2*(1 - Lambda**2)
           - 2*g*l*(P - Lambda*Gamma)
           - l**2*(Q + Gamma**2))
      v = v_tilde + D*norm/L

  return {'h': h_hist, 'y': y, 'e': e, 'v': v_hist}


def _shift(new_x_sample, x_window):
  """Slide the input window by one sample. Same as the notebook's shift()."""
  L = len(x_window)
  new_x_window = np.zeros(L)
  new_x_window[0] = new_x_sample
  new_x_window[1:] = x_window[:-1]
  return new_x_window
