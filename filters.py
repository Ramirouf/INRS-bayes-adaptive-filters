"""Filter implementations for the Bayesian adaptive filtering study.

Everything defined here was extracted verbatim from `bayes-adaptive-filters.ipynb`,
which is the only consumer. The notebook keeps the simulations, configuration and
plots; this module keeps the algorithms, the parameter dtypes and the Monte Carlo
drivers, so that they get real diffs, blame and tests.

Contents, in order:
  - autocorrelation matrix estimation/calculation, AR settling time
  - shift() and filter() primitives
  - parameter dtypes and the closed-form filters: NLMS, sKF, sKF-L (minorized),
    sKF-L (exact)
  - signal environments (Gaussian and Laplacian noise) and MC_Simulations_Modular*
  - the numerical-integration reference filters and their FFT convolution machinery

Numba kernels are compiled with cache=True where numba allows it, so a kernel
restart does not pay for recompilation. Functions using objmode or taking a
function argument cannot be cached and are left as plain @njit.
"""

import numpy as np
import scipy.signal
import scipy
import numba
from numba import njit, prange, objmode, types
from typing import List, Optional, Any, AnyStr, TypedDict
from collections.abc import Callable
from matplotlib import pyplot as plt
from numba.types import Array, complex128, float64
from scipy.special import log_ndtr, logsumexp



@njit(cache=True)
def autocorr_matrix_estimate(signal, M = 4):
  # Inicializa as variáveis
  x_temp, R = np.zeros((M, 1)), np.zeros((M, M))

  for x in signal:
    # Desloca as amostras e insere um novo valor na primeira posição
    x_temp[1:], x_temp[0] = x_temp[:-1], x

    # Calcula a matriz de correlação e acumula
    R += x_temp @ x_temp.T

  # Retorna a matriz normalizada pelo tamanho do sinal
  return R/(signal.size)

@njit(cache=True)
def toeplitz(vector):
  L = len(vector)
  matrix = np.zeros((L,L))

  for k in range(L):
      matrix[k, :k] = vector[k:0:-1]
      matrix[k, k:] = vector[:L-k]

  return matrix

@njit(cache=True)
def autocorr_matrix_calc(AR, var_v, M = None):
  # Autoregressive order
  L = len(AR) - 1
  # Number of autocorrelation samples to be computed
  if M is None:
    M = L

  # Initialize the vector containing the autocorrelation values
  rxx = np.zeros((M,))

  # State Space matrix describing the autoregressive process
  A = np.zeros((L,L))
  A[1:,:] = np.eye(L-1, M = L)
  A[0,:] = -AR[1:]

  # Intermediate values associated with the calclulation
  AA = np.kron(A,A);
  BB_vec_IL = np.zeros((L**2,))  # Effect of the matrix operating over the input data on the state space
  BB_vec_IL[0] = 1;
  I_L2 = np.eye(L**2)

  # Calculation of the autocorrelation matrix with the first L values
  vec_Rxx = var_v * np.linalg.inv(I_L2 - AA) @ BB_vec_IL

  # Save the first L values
  rxx[:L] = vec_Rxx[:L]

  # Compute the remaining values using the autoregressive coeficients
  if M > L:
    for k in range(L, M):
      if k == L:
        rxx[k] = - AR[1:] @ rxx[(k-1)::-1]
      else:
        rxx[k] = - AR[1:] @ rxx[(k-1):(k - L - 1):-1]

  # Construct the autocorrelation matrix using the values in rxx
  R = toeplitz(rxx)

  return R

@njit(cache=True)
def AR_settling_time(AR, error = 0.01):
  L = len(AR) - 1
  A = np.zeros((L,L))
  A[1:,:] = np.eye(L-1, M = L)
  A[0,:] = -AR[1:]
  A = A.astype(np.complex64)

  eigen_values = np.linalg.eigvals(A)
  slowest_decay_ratio = np.max(np.abs(eigen_values))
  if slowest_decay_ratio >= 1:
    raise(Exception('The given AR process will not converge'))

  settling_time = np.log(error)/np.log(slowest_decay_ratio)
  return int(np.ceil(settling_time))


@njit(cache=True)
def shift(new_x_sample, x_window):
  L = len(x_window)
  new_x_window = np.zeros(L)
  new_x_window[0] = new_x_sample
  new_x_window[1:] = x_window[:-1]
  return new_x_window

@njit(cache=True)
def filter(a, b, x):
  K = len(x)
  L_a = len(a)
  L_b = len(b)
  y_vec = np.zeros(L_a - 1)
  x_vec = np.zeros(L_b)
  if a[0] != 0.0:
    b = b/a[0]
    a = a[1:]/a[0]
  else:
    raise(Exception('a[0] can NOT be zero!'))
  y = np.zeros(K)

  for k in range(K):
    x_vec = shift(x[k], x_vec)

    y[k] = b @ x_vec - a @ y_vec
    y_vec = shift(y[k], y_vec)

  return y


NLMS_params = np.dtype([("label", "U20"), # Unicode string up to 20 characters
                        ("mu", "f8"),     # 64-bit floating-point number
                        ("delta", "f8")   # 64-bit floating-point number
                      ])

sKF_params = np.dtype([("label", "U20"), # Unicode string up to 20 characters
                        ("epsilon", "f8"),     # 64-bit floating-point number
                        ("var_eta", "f8"),   # 64-bit floating-point number
                        ("v_tilde_0", "f8")   # 64-bit floating-point number
                      ])

sKF_L_params = np.dtype([("label", "U20"), # Unicode string up to 20 characters
                        ("epsilon", "f8"),     # 64-bit floating-point number
                        ("b_eta", "f8"),   # Escala de la distribución de Laplace
                        ("v_tilde_0", "f8")   # 64-bit floating-point number
                      ])

@njit(cache=True)
def NLMS_algorithm(N, x, d, h0, parameters):
  h = h0
  mu = parameters.mu
  delta = parameters.delta

  L = len(h)
  y = np.zeros((N,))
  e = np.zeros((N,))
  xtemp = np.zeros(L)

  for k in range(0,N):
    xtemp = shift(x[k], xtemp)
    y[k] = h @ xtemp
    e[k] = d[k] - y[k]

    if k >= L:
      x_power = delta + xtemp @ xtemp
      h = h + mu*xtemp*e[k]/x_power

  return {'h': h, 'y': y, 'e': e}

def sKF_algorithm(N, x, d, h0, parameters):
  h = h0
  epsilon = parameters["epsilon"]
  var_eta = parameters["var_eta"]
  v_tilde_0 = parameters["v_tilde_0"]

  # normalize x
  # regularization = 1e-3
  # x_reg = np.sign(x) * (np.abs(x) + regularization)

  L = len(h)
  y = np.zeros((N,))
  e = np.zeros((N,))
  xtemp = np.zeros(L)
  v=v_tilde_0
  h_hist = np.zeros((N, L))
  v_hist = np.zeros((N, L))
  # d: salida del sistema con ruido
  # y: salida estimada
  for k in range(0,N):
    xtemp = shift(x[k], xtemp)
    y[k] = h @ xtemp
    e[k] = d[k] - y[k]
    h_hist[k] = h
    v_hist[k] = v

    if k >= L:
      # predict
      v += epsilon
      # update
      norm = xtemp @ xtemp
      s = var_eta + v * norm
      h = h + xtemp * (v * e[k]/s) # not h+= because it would mutate h0
      v = v * (1 - (v * norm) / (L * s)) 

  return {'h': h_hist, 'y': y, 'e': e, 'v': v_hist}

def sKF_L_algorithm(N, x, d, h0, parameters):
    h = h0
    epsilon = parameters["epsilon"]
    b_eta = parameters["b_eta"]
    v_tilde_0 = parameters["v_tilde_0"]

    # normalize x
    # regularization = 1e-3
    # x_reg = np.sign(x) * (np.abs(x) + regularization)

    L = len(h)
    y = np.zeros((N,))
    e = np.zeros((N,))
    xtemp = np.zeros(L)
    v=v_tilde_0
    h_hist = np.zeros((N, L))
    v_hist = np.zeros((N, L))
    # d: salida del sistema con ruido
    # y: salida estimada
    for k in range(0,N):
        xtemp = shift(x[k], xtemp)
        y[k] = h @ xtemp
        e[k] = d[k] - y[k]
        h_hist[k] = h
        v_hist[k] = v

        if k >= L:
            # predict
            v += epsilon
            # update
            norm = xtemp @ xtemp
            s = b_eta * abs(e[k]) + v * norm
            h = h + xtemp * (v * e[k]/s) # not h+= because it would mutate h0
            v = v * (1 - (v * norm) / (L * s)) 

    return {'h': h_hist, 'y': y, 'e': e, 'v': v_hist}

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
        xtemp = shift(x[k], xtemp)
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
            kappa = (_SIGN * e[k] - v_tilde * norm / b_eta) / (
                np.sqrt(v_tilde) * norm_x
            )

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
            log_pi = -_SIGN * e[k] / b_eta + log_Phi
            pi = np.exp(log_pi - logsumexp(log_pi))  # mixture weights, sum to 1

            log_phi = -0.5 * kappa**2 - 0.5 * np.log(2 * np.pi)
            mills = np.exp(log_phi - log_Phi)  # phi(kappa)/Phi(kappa)
            # ----------------------------------------------------------------------

            Lambda = np.sum(_SIGN * pi)  # saturating error, in [-1, 1]
            Gamma = np.sum(_SIGN * pi * mills)
            P = np.sum(pi * mills)
            Q = np.sum(pi * kappa * mills)

            gain = v_tilde * Lambda / b_eta - np.sqrt(v_tilde) * Gamma / norm_x
            h = h + gain * xtemp  # not h+= because it would mutate h0

            g = v_tilde / b_eta
            l = np.sqrt(v_tilde) / norm_x
            D = (
                g**2 * (1 - Lambda**2)
                - 2 * g * l * (P - Lambda * Gamma)
                - l**2 * (Q + Gamma**2)
            )
            v = v_tilde + D * norm / L

    return {"h": h_hist, "y": y, "e": e, "v": v_hist}


@njit(cache=True)
def std_behavior(N, ho, var_x, var_v, AR, settling_time = 0):
  # AR process order and filter length
  P = len(AR)
  L = len(ho)

  # Determine the noise signal
  v = np.sqrt(var_v)*np.random.randn(N)

  # Determine the input signal x through a AR process
  x = np.random.randn(N + settling_time + L)

  # Generate the correlated signal
  AR = AR/AR[0]
  aux_Rxx = autocorr_matrix_calc(AR, 1, M = len(AR) - 1)
  b = np.sqrt(var_x/aux_Rxx[0,0])
  x = filter(AR, np.array([b]), x)[settling_time:]

  # Determine the desired signal
  d = v + np.convolve(ho, x, mode = 'full')[L:N+L]
  x = x[L:N+L]

  return {'x': x, 'v': v, 'd': d}

@njit(cache=True)
def laplacian_noise_behavior(N, ho, var_x, scale_v, AR, settling_time = 0):
  # AR process order and filter length
  P = len(AR)
  L = len(ho)

  # Determine the noise signal
  v = np.random.laplace(loc=0.0, scale=scale_v, size=(N,))

  # Determine the input signal x through a AR process
  x = np.random.randn(N + settling_time + L)

  # Generate the correlated signal
  AR = AR/AR[0]
  aux_Rxx = autocorr_matrix_calc(AR, 1, M = len(AR) - 1)
  b = np.sqrt(var_x/aux_Rxx[0,0])
  x = filter(AR, np.array([b]), x)[settling_time:]

  # Determine the desired signal
  d = v + np.convolve(ho, x, mode = 'full')[L:N+L]
  x = x[L:N+L]

  return {'x': x, 'v': v, 'd': d}

def MC_Simulations_Modular(N, NR, ho, var_x, var_v, h0, Algorithms, Parameters, AR, PBar = None):
  L = len(h0)
  N_Algorithms = len(Algorithms)

  tau = AR_settling_time(AR)
  measure_init = lambda taps, N_iter: {'h': np.zeros((N_iter, taps)),
                                       'J': np.zeros(N_iter),
                                       'Jex': np.zeros(N_iter)}

  measures = {Parameters[k]["label"]: measure_init(L, N) for k in range(N_Algorithms)}

  for k in range(NR):
    signals = std_behavior(N, ho, var_x, var_v, AR, tau)
    x = signals['x']
    d = signals['d']

    for c in range(N_Algorithms):
      label = Parameters[c]["label"]
      algorithm_signals = Algorithms[c](N, x, d, h0, Parameters[c])
      measures[label]['h'] += algorithm_signals['h']
      measures[label]['J'] += algorithm_signals['e']**2
      measures[label]['Jex'] += (algorithm_signals['e']- signals['v'])**2

    if not PBar is None:
      PBar.update(1)
    else:
      print(f'Realization {k} out of {NR}')

  for k in range(N_Algorithms):
    label = Parameters[k]["label"]
    measures[label]['h'] /= NR
    measures[label]['J'] /= NR
    measures[label]['Jex'] /= NR

  return measures


# COPY PASTE FOR RETURNING VARIANCE, FOR COMPARING
def MC_Simulations_Modular_Variance(N, NR, ho, var_x, var_v, h0, Algorithms, Parameters, AR, PBar = None):
  L = len(h0)
  N_Algorithms = len(Algorithms)
  tau = AR_settling_time(AR)
  measure_init = lambda taps, N_iter: {'h': np.zeros((N_iter, taps)),
                                       'J': np.zeros(N_iter),
                                       'Jex': np.zeros(N_iter),
                                       'var': np.zeros((N_iter, taps))}

  measures = {Parameters[k]["label"]: measure_init(L, N) for k in range(N_Algorithms)}

  for k in range(NR):
    signals = std_behavior(N, ho, var_x, var_v, AR, tau)
    x = signals['x']
    d = signals['d']

    for c in range(N_Algorithms):
      label = Parameters[c]["label"]
      algorithm_signals = Algorithms[c](N, x, d, h0, Parameters[c])
      measures[label]['h'] += algorithm_signals['h']
      measures[label]['J'] += algorithm_signals['e']**2
      measures[label]['Jex'] += (algorithm_signals['e']- signals['v'])**2
      measures[label]['var'] += algorithm_signals['v']

    if not PBar is None:
      PBar.update(1)
    else:
      print(f'Realization {k} out of {NR}')

  for k in range(N_Algorithms):
    label = Parameters[k]["label"]
    measures[label]['h'] /= NR
    measures[label]['J'] /= NR
    measures[label]['Jex'] /= NR
    measures[label]['var'] /= NR

  return measures


skf_int_params = np.dtype([("label", "U20"),  # Unicode string up to 20 characters
                           ("epsilon", "f8"), # 64-bit floating-point number
                           ("var_theta_0", "f8"),  # 64-bit floating-point number
                           ("var_eta", "f8"), # 64-bit floating-point number
                           ("dx_factor", "f8"),
                           ("min_std_deviations", "f8")])

skf_L_int_params = np.dtype([("label", "U20"),  # Unicode string up to 20 characters
                           ("epsilon", "f8"), # 64-bit floating-point number
                           ("var_theta_0", "f8"),  # 64-bit floating-point number
                           ("b_eta", "f8"), # 64-bit floating-point number
                           ("dx_factor", "f8"),
                           ("min_std_deviations", "f8")])

gaussian_params = np.dtype([("mean", "f8"),
                            ("variance", "f8")])
laplacian_params = np.dtype([("mean", "f8"),
                            ("b", "f8")])

_float_array_1d = types.float64[:]

@njit(cache=True)
def gaussian_pdf(x, params: gaussian_params):
  return (1/np.sqrt(2*np.pi*params.variance))*np.exp(-(x-params.mean)**2/(2*params.variance))

@njit(cache=True)
def laplacian_pdf(x, params: laplacian_params):
  return (1/(2*params.b))*np.exp(-np.abs(x-params.mean)/params.b)

@njit(cache=True)
def _compute_individual_interferences_pdfs(base_space, theta_pdf_function, pdf_parameters, input_data, regularization = 1e-15):
  f_inter_list = []
  for x in input_data:
    interference_pdf = theta_pdf_function(base_space/(regularization + np.abs(x)), pdf_parameters)
    interference_pdf /= regularization + np.abs(x)
    f_inter_list.append(interference_pdf)

  return f_inter_list

@njit(cache=True)
def fft_integral_convolve(f, g, dx):
    """
    Computes the numerical convolution integral using the FFT.
    Numba compatible.

    Parameters:
        f (np.ndarray): 1D array of first function samples.
        g (np.ndarray): 1D array of second function samples.
        dx (float): Grid spacing/step size.

    Returns:
        np.ndarray: 1D array containing the corrected trapezoidal convolution.
    """
    input_len = len(f)

    with objmode(riemann_sum=_float_array_1d):
        riemann_sum = scipy.signal.fftconvolve(f, g, mode='full')

    return dx * riemann_sum[input_len//2:3*input_len//2]

# Tipos de array estáticos exigidos pelo objmode
Complex1D = Array(complex128, 1, "C")
Complex2D = Array(complex128, 2, "C")
Float1D   = Array(float64, 1, "C")

@njit(cache=True)
def integral_convolve_from_base_pdf(full_signals,
                                    base_signal,
                                    freq_scalings,
                                    translations,
                                    center_sample,
                                    dx,
                                    log_sum = False):
    """
    Implements efficiently the convolution integral assuming that I signals f(t)
    are being convolved as well as J signals g(.) obtained from time scaling
    ___               ___
    | |Fourier{f_i(t)}| |Fourier{g[(t-T_j)/C_j]}
     i                 j
    Note that the Fourier transforms are approximations of the continuous time
    Fourier transform, therefore, assuming proper sampling, continuous time
    properties and linear interpolation are used.
    Allows robust treatment of time scalings like C_j --> 0.
    """
    I, N = full_signals.shape
    J = len(freq_scalings)
    K = I + J
    N_conv = K * N - (K - 1)

    # 1. FFT of all PDFs
    with objmode(espectres=Complex2D, base_spectre=Complex1D):
        espectres = scipy.fft.rfft(full_signals, n=N_conv, axis=1)
        base_spectre = scipy.fft.rfft(base_signal, n=N_conv)

    # 2. Correct the PDF center displacement
    freq_space = np.arange(0, (N_conv//2+1), 1)
    centering_vector = np.exp(2j*np.pi*center_sample*freq_space/N_conv)
    base_spectre *= centering_vector

    # 3. Reescale the PDF Fourier transform
    NF = len(base_spectre)
    input_space = np.arange(0, NF, 1)
    interpolated_spectres = np.zeros((J, NF), np.complex128)
    for j in range(J):
      interpolated_spectres[j] = np.interp(freq_scalings[j]*input_space,
                                           input_space,
                                           base_spectre)
    espectres = np.append(espectres, interpolated_spectres, axis=0)

    # 4. Product of all FFTs
    if log_sum:
      log_espectre_prod = np.log(espectres[0])
      for k in range(1, K):
          log_espectre_prod += np.log(espectres[k])
      total_translation = J*center_sample + (translations @ freq_scalings)/dx
      log_espectre_prod += -2j*np.pi*total_translation*freq_space/N_conv
      espectre_prod = np.exp(log_espectre_prod)
    else:
      espectre_prod = espectres[0].copy()
      for k in range(1, K):
          espectre_prod *= espectres[k]
      total_translation = J*center_sample + (translations @ freq_scalings)/dx
      espectre_prod *= np.exp(-2j*np.pi*total_translation*freq_space/N_conv)

    # 5. Inverse FFT tranform
    with objmode(y_full=Float1D):
        y_full = scipy.fft.irfft(espectre_prod, n=N_conv)

    # 6. Windowing of the convolved signal
    start_idx = (K - 1) * (N // 2)
    end_idx = start_idx + N

    return y_full[start_idx:end_idx]*(dx ** (K - 1))

@njit(cache=True)
def _compute_composite_noise_pdf(f_eta, f_inter_list, dx):
  f_zeta = f_eta
  for f_inter in f_inter_list:
    f_zeta = fft_integral_convolve(f_zeta, f_inter, dx)
  return f_zeta

def _get_integration_range(relative_range,
                           var_tilde,
                           var_zeta,
                           d_k,
                           x_k,
                           w_k,
                           mean_S,
                           regularization = 1e-12):
  worst_std = np.sqrt(np.max([var_tilde, var_zeta]))
  prior_range = relative_range*np.sqrt(var_tilde)
  prior_limits = (np.min(w_k) - prior_range, np.max(w_k) + prior_range)

  reg_xk = regularization + np.abs(x_k)
  corrected_mean = reg_xk*np.abs(w_k) + (np.abs(d_k) + np.abs(mean_S))
  likelihood_range = np.max((relative_range*np.sqrt(var_zeta) + corrected_mean)/reg_xk)
  likelihood_limits = (-likelihood_range, likelihood_range)
  final_limits = (np.min((prior_limits[0], likelihood_limits[0])),
                  np.max((prior_limits[1], likelihood_limits[1])))

  return final_limits

def sKF_integral_algorithm(N, x, d, w0, parameters):
  w_hist = np.zeros((N, len(w0))) # Nuevo
  var_theta_hist = np.zeros((N, len(w0))) # NUEVO !!!!
  w = w0.copy()
  L = len(w)
  regularization = 1e-3
  x_reg = np.sign(x)*(np.abs(x) + regularization)

  scalar_var_theta = parameters["var_theta_0"]
  var_theta = scalar_var_theta*np.ones((L,))
  epsilon = parameters["epsilon"]
  var_eta = parameters["var_eta"]
  eta_parameters = np.void((0, var_eta), dtype=gaussian_params)

  #dx = np.min([epsilon, var_eta])*parameters.dx_factor
  relative_range = parameters["min_std_deviations"]

  y = np.zeros((N,))
  e = np.zeros((N,))
  xtemp = np.zeros(L)

  #print(f"w0: {w}")
  for k in range(0, N):
    xtemp = shift(x_reg[k], xtemp)
    y[k] = w @ xtemp
    e[k] = d[k] - y[k]

    w_hist[k] = w.copy() # NUEVO
    var_theta_hist[k] = scalar_var_theta.copy() # NUEVO
    if k >= L:
      var_tilde = scalar_var_theta + epsilon
      worst_var_zeta = var_eta + (np.linalg.norm(xtemp)**2)*var_tilde
      mean_S = y[k] - w*xtemp

      #print(f"Var_tilde: {var_tilde}; Var_eta: {var_eta}; Worst Var_zeta: {worst_var_zeta}")
      dx = np.sqrt(np.min([var_tilde, var_eta]))*parameters["dx_factor"]
      int_range = _get_integration_range(relative_range,
                           var_tilde,
                           worst_var_zeta,
                           d[k],
                           xtemp,
                           w,
                           mean_S)

      theta_parameters = np.void((0, var_tilde), dtype=gaussian_params)
      #base_space = np.arange(-int_range, int_range, dx)
      base_space = np.arange(int_range[0], int_range[1], dx)
      f_eta = gaussian_pdf(base_space, eta_parameters)
      #prior_m = gaussian_pdf(base_space, theta_parameters)
      #f_inter_list = np.array(_compute_individual_interferences_pdfs(base_space, gaussian_pdf, theta_parameters, xtemp))
      base_pdf = gaussian_pdf(base_space, theta_parameters)

      #print(f"Worst calculated var_zeta: {worst_var_zeta}")
      #print(f"dx: {dx}\nbase space length: {len(base_space)}")

      #plt.clf()
      for m in range(L):
        zeta_space = base_space + mean_S[m]
        #theta_m_space = base_space - w[m]
        likelihood_space = d[k] - xtemp[m]*base_space
        freq_scalings = np.abs(xtemp[np.arange(0,L,1) != m])

        inter_mask = ~np.zeros((L,), dtype=bool); inter_mask[m] = False
        #f_zeta = _compute_composite_noise_pdf(f_eta, f_inter_list[~inter_mask], dx)
        f_zeta = integral_convolve_from_base_pdf(np.array([f_eta]),
                                                 base_pdf,
                                                 freq_scalings,
                                                 np.zeros((L-1,)),
                                                 (len(f_eta)+1)//2,
                                                 dx)

        likelihood = np.interp(likelihood_space, zeta_space, f_zeta, left=0, right=0)
        prior_m = gaussian_pdf(base_space, np.void((w[m], var_tilde), dtype=gaussian_params))
        post_m = np.nan_to_num(np.exp(np.log(prior_m) + np.log(likelihood))) # Search for numerical stable implementations
        post_m /= np.trapezoid(post_m, dx=dx)

        #print(f"Obtained var_zeta {m}: {np.trapezoid(base_space**2*f_zeta, dx=dx)}")
        #plt.plot(base_space, np.interp(base_space, zeta_space, f_zeta, left=0, right=0))
        #plt.scatter([mean_S[m]], [np.max(f_zeta)])
        #print(f"d: {d[k]}; w[{m}]: {w[m]}; x[{m}]: {xtemp[m]}; S[{m}]: {mean_S[m]}")
        #pred_mean = (d[k] - mean_S[m])/xtemp[m]
        #print(f"Predicted_mean: {pred_mean}; Value at pred. mean: {np.interp(pred_mean, theta_m_space, likelihood, left=0, right=0)}; Max value: {np.max(f_zeta)}")
        #plt.plot(theta_m_space, abs(xtemp[m])*likelihood)
        #plt.scatter([pred_mean], [abs(xtemp[m])*np.max(likelihood)])

        #pred_base = w[m] + (d[k] - mean_S[m])/xtemp[m]
        #print(f"pred_base: {pred_base}; actual mean = {np.trapezoid(base_space*abs(xtemp[m])*likelihood, dx=dx)}")
        w[m] = np.trapezoid(base_space*post_m, dx=dx)
        var_theta[m] = np.trapezoid(((base_space-w[m])**2)*post_m, dx=dx)
        #plt.plot(base_space, post_m)
        #plt.scatter([w[m]],[np.max(post_m)])
        #print(np.shape(post_m))
        #print(f"var_theta[{m}]: {var_theta[m]}]")

      #print(f"x[{k}]: {xtemp}")
      #print(f"w[{k}]: {w}")
      #print("----------------------------------------------------")
      scalar_var_theta = np.mean(var_theta)
      # var_theta_hist[k] = var_theta

  return {'h': w_hist, 'y': y, 'e': e, 'v': var_theta_hist}

def sKF_L_integral_algorithm(N, x, d, w0, parameters):
  w = w0.copy()
  L = len(w)

  w_hist = np.zeros((N, len(w0)))
  var_theta_hist = np.zeros((N, len(w0)))

  scalar_var_theta = parameters["var_theta_0"]
  var_theta = scalar_var_theta*np.ones((L,))
  epsilon = parameters["epsilon"]
  b_eta = parameters["b_eta"]
  var_eta = 2*b_eta**2
  eta_parameters = np.void((0, b_eta), dtype=laplacian_params)

  relative_range = parameters["min_std_deviations"]

  y = np.zeros((N,))
  e = np.zeros((N,))
  xtemp = np.zeros(L)

  regularization = 1e-3
  x_reg = np.sign(x)*(np.abs(x) + regularization)

  #print(f"w0: {w}")
  for k in range(0, N):
    xtemp = shift(x_reg[k], xtemp)
    y[k] = w @ xtemp
    e[k] = d[k] - y[k]

    w_hist[k] = w.copy()
    var_theta_hist[k] = scalar_var_theta.copy()
    if k >= L:
      var_tilde = scalar_var_theta + epsilon
      worst_var_zeta = var_eta + (np.linalg.norm(xtemp)**2)*var_tilde
      mean_S = y[k] - w*xtemp

      #print(f"Var_tilde: {var_tilde}; Var_eta: {var_eta}; Worst Var_zeta: {worst_var_zeta}")
      dx = np.sqrt(np.min([var_tilde, var_eta]))*parameters["dx_factor"]
      int_range = _get_integration_range(relative_range,
                           var_tilde,
                           worst_var_zeta,
                           d[k],
                           xtemp,
                           w,
                           mean_S)

      theta_parameters = np.void((0, var_tilde), dtype=gaussian_params)
      base_space = np.arange(int_range[0], int_range[1], dx)
      f_eta = laplacian_pdf(base_space, eta_parameters)
      base_pdf = gaussian_pdf(base_space, theta_parameters)

      #print(f"Worst calculated var_zeta: {worst_var_zeta}")
      #print(f"dx: {dx}\nbase space length: {len(base_space)}")

      #plt.clf()
      for m in range(L):
        zeta_space = base_space + mean_S[m]
        likelihood_space = d[k] - xtemp[m]*base_space
        freq_scalings = np.abs(xtemp[np.arange(0,L,1) != m])

        inter_mask = ~np.zeros((L,), dtype=bool); inter_mask[m] = False
        f_zeta = integral_convolve_from_base_pdf(np.array([f_eta]),
                                                 base_pdf,
                                                 freq_scalings,
                                                 np.zeros((L-1,)),
                                                 (len(f_eta)+1)//2,
                                                 dx)

        likelihood = np.interp(likelihood_space, zeta_space, f_zeta, left=0, right=0)
        prior_m = gaussian_pdf(base_space, np.void((w[m], var_tilde), dtype=gaussian_params))
        post_m  = np.nan_to_num(np.exp(np.log(prior_m) + np.log(likelihood))) # Search for numerical stable implementations
        post_m /= np.trapezoid(post_m, dx=dx)

        #print(f"Obtained var_zeta {m}: {np.trapezoid(base_space**2*f_zeta, dx=dx)}")
        #plt.plot(base_space, np.interp(base_space, zeta_space, f_zeta, left=0, right=0))
        #plt.scatter([mean_S[m]], [np.max(f_zeta)])
        #print(f"d: {d[k]}; w[{m}]: {w[m]}; x[{m}]: {xtemp[m]}; S[{m}]: {mean_S[m]}")
        #pred_mean = (d[k] - mean_S[m])/xtemp[m]
        #print(f"Predicted_mean: {pred_mean}; Value at pred. mean: {np.interp(pred_mean, theta_m_space, likelihood, left=0, right=0)}; Max value: {np.max(f_zeta)}")
        #plt.plot(theta_m_space, abs(xtemp[m])*likelihood)
        #plt.scatter([pred_mean], [abs(xtemp[m])*np.max(likelihood)])

        w[m] = np.trapezoid(base_space*post_m, dx=dx)
        var_theta[m] = np.trapezoid(((base_space-w[m])**2)*post_m, dx=dx)
        #plt.plot(base_space, post_m)
        #plt.scatter([w[m]],[np.max(post_m)])
        #print(np.shape(post_m))
        #print(f"var_theta[{m}]: {var_theta[m]}]")

      #print(f"x[{k}]: {xtemp}")
      #print(f"w[{k}]: {w}")
      #print("----------------------------------------------------")
      scalar_var_theta = np.mean(var_theta)

  return {'h': w_hist, 'y': y, 'e': e, 'v': var_theta_hist}
