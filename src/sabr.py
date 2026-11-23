"""
SABR model: Hagan's implied volatility formula, the calibration and an arbitrage check.

The main specification uses beta = 1 (Chapter 3 of the thesis). Other values of beta are used
only in a robustness check (beta = 0.5): there the formula takes the leading term of
Obloj (2008) and the correction term of Hagan et al. (2002).

The three parameters fitted to each smile:
  alpha  level of volatility (close to the at-the-money volatility)
  rho    correlation between the price and its volatility (negative = more expensive puts)
  nu     volatility of volatility (how curved the smile is)

The calibration is a least-squares fit of the implied volatilities, with errors in volatility points.
"""

import numpy as np
from scipy.optimize import least_squares
import black

# limits for the three parameters during the calibration
ALPHA_BOUNDS = (1e-4, 10.0)
RHO_BOUNDS = (-0.999, 0.999)
NU_BOUNDS = (1e-3, 100.0)
NU_STARTS = (1.0, 5.0, 20.0)       # three starting values for nu: the best fit is kept


def _z_over_x(z, rho):
    """Ratio z / x(z) of Hagan's formula, with x(z) = ln{[sqrt(1 - 2 rho z + z^2) + z - rho] / (1 - rho)}.

    When z goes to 0 the ratio goes to 1, so it is set to 1 there to avoid a division by zero."""
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    big = np.abs(z) > 1e-7
    zb = z[big]
    x = np.log((np.sqrt(1 - 2 * rho * zb + zb ** 2) + zb - rho) / (1 - rho))
    out[big] = zb / x
    return out


def hagan_iv(K, F, tau, alpha, rho, nu, beta=1.0):
    """SABR implied volatility at the strikes K.

    beta = 1: Hagan's formula of Chapter 3, sigma = I0 * (1 + I1 * tau).
    beta != 1 (robustness check only): the same form, with the I0 of Obloj (2008) and the I1 of Hagan."""
    K = np.asarray(K, dtype=float)
    logFK = np.log(F / K)
    if beta == 1.0:                                   # main case
        z = (nu / alpha) * logFK
        i0 = alpha * _z_over_x(z, rho)
        i1 = rho * nu * alpha / 4 + (2 - 3 * rho ** 2) * nu ** 2 / 24
        return i0 * (1 + i1 * tau)
    omb = 1.0 - beta                                  # general beta (robustness check only)
    z = (nu / alpha) * (F ** omb - K ** omb) / omb
    i0 = np.where(np.abs(logFK) < 1e-10, alpha * F ** (beta - 1),
                  nu * logFK / np.where(np.abs(z) < 1e-12, 1.0, _x_of_z(z, rho)))
    fk = (F * K) ** omb
    i1 = (omb ** 2 / 24) * alpha ** 2 / fk + rho * beta * nu * alpha / (4 * np.sqrt(fk)) \
        + (2 - 3 * rho ** 2) * nu ** 2 / 24
    return i0 * (1 + i1 * tau)


def _x_of_z(z, rho):
    """x(z) of Hagan's formula (used in the general-beta case)."""
    return np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))


def calibrate(K, iv, F, tau, beta=1.0, sigma_atm=None, weights=None):
    """Find the alpha, rho and nu that bring the SABR smile closest to the market smile.

    K, iv: strikes and market implied volatilities used for the fit.
    The fit minimises the sum of squared errors (model IV - market IV) x 100, that is, errors in
    volatility points. Optional weights (for example vega) give some strikes more importance.
    Three starting points are tried (alpha = ATM volatility, rho = -0.5, nu = 1, 5 or 20) and the
    best result is kept.
    Returns a dictionary with the parameters, the fit error (RMSE in volatility points) and two
    flags: whether the optimiser converged and whether a parameter stopped at one of its limits."""
    K, iv = np.asarray(K, float), np.asarray(iv, float)
    w = np.ones_like(iv) if weights is None else np.asarray(weights, float) / np.mean(weights)
    s0 = float(np.median(iv)) if sigma_atm is None else float(sigma_atm)
    a0 = s0 * F ** (1 - beta)
    lo = [ALPHA_BOUNDS[0], RHO_BOUNDS[0], NU_BOUNDS[0]]
    hi = [ALPHA_BOUNDS[1], RHO_BOUNDS[1], NU_BOUNDS[1]]

    def resid(p):                                     # one error per strike, in volatility points
        return w * (hagan_iv(K, F, tau, p[0], p[1], p[2], beta) - iv) * 100

    best = None
    for nu0 in NU_STARTS:
        p0 = [min(max(a0, lo[0] * 1.01), hi[0] * 0.99), -0.5, nu0]
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12,
                              max_nfev=2000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return dict(alpha=np.nan, rho=np.nan, nu=np.nan, beta=beta, rmse=np.nan, converged=False, at_bound=True)
    a, rho, nu = best.x
    # a parameter stuck at its limit is the sign of a poor fit
    at_bound = (nu > 0.99 * NU_BOUNDS[1]) or (abs(rho) > 0.99) or (a < 2 * ALPHA_BOUNDS[0]) or (a > 0.99 * ALPHA_BOUNDS[1])
    rmse = float(np.sqrt(np.mean(((hagan_iv(K, F, tau, a, rho, nu, beta) - iv) * 100) ** 2)))
    return dict(alpha=float(a), rho=float(rho), nu=float(nu), beta=beta, rmse=rmse,
                converged=bool(best.success) and not at_bound, at_bound=bool(at_bound), nfev=int(best.nfev))


def check_arbitrage(F, tau, D, params, K_min, K_max, n=201):
    """Check that the SABR smile gives arbitrage-free call prices between K_min and K_max.

    On a grid of n strikes, call prices must fall (or stay equal) when the strike rises and must be
    convex in the strike; otherwise the probability density implied by the prices would be negative.
    Returns the number of violations and the worst one (the most negative butterfly, in dollars)."""
    Kg = np.linspace(K_min, K_max, n)
    iv = hagan_iv(Kg, F, tau, params["alpha"], params["rho"], params["nu"], params.get("beta", 1.0))
    if not np.all(np.isfinite(iv)) or np.any(iv <= 0):
        return dict(n_viol=n, worst=np.nan)
    C = black.price(F, Kg, tau, iv, D, True)
    mono = np.diff(C)                                   # must be <= 0
    bfly = C[:-2] - 2 * C[1:-1] + C[2:]                 # must be >= 0
    n_viol = int((mono > 1e-10).sum() + (bfly < -1e-10).sum())
    worst = float(min(bfly.min(), -mono.max())) if n_viol else 0.0
    return dict(n_viol=n_viol, worst=worst)
