"""
Black's formula on the forward price (Black-76): option prices, Greeks and implied volatility.

This is the base module of the project (Chapter 3 of the thesis). The other scripts use it
to price options, to turn prices into implied volatilities and back, and to compute delta
and vega.

Notation used in all functions:
  F        forward price of SPY for the expiry of the option
  K        strike price
  tau      time to expiry, in years
  sigma    volatility (0.20 means 20%)
  D        discount factor: the value today of 1 dollar paid at expiry
  is_call  True for a call, False for a put

All functions accept single numbers or numpy arrays.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

IV_LO, IV_HI = 0.001, 5.0        # the implied volatility is searched between 0.1% and 500%


def d1d2(F, K, sigma, tau):
    """The two terms d1 and d2 of Black's formula."""
    st = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * tau) / st
    return d1, d1 - st


def price(F, K, tau, sigma, D, is_call):
    """Black-76 price of a call or a put on the forward."""
    d1, d2 = d1d2(F, K, sigma, tau)
    call = D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)


def vega(F, K, tau, sigma, D):
    """Vega: how much the option price changes when the volatility changes (derivative with respect to sigma)."""
    d1, _ = d1d2(F, K, sigma, tau)
    return D * F * norm.pdf(d1) * np.sqrt(tau)


def delta_fwd(F, K, tau, sigma, is_call):
    """Forward delta: N(d1) for a call, N(d1) - 1 for a put (Chapter 3)."""
    d1, _ = d1d2(F, K, sigma, tau)
    return np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)


def implied_vol(p, F, K, tau, D, is_call):
    """Implied volatility of one option price p: the sigma at which Black's formula gives p.

    It is found with Brent's method (scipy brentq) between IV_LO and IV_HI.
    Returns nan when no such sigma exists, for example when p breaks the no-arbitrage bounds."""
    upper = D * F if is_call else D * K          # an option can never be worth more than this
    if not (0.0 < p < upper):
        return np.nan
    f = lambda s: float(price(F, K, tau, s, D, is_call)) - p
    try:
        lo, hi = f(IV_LO), f(IV_HI)
    except (FloatingPointError, ValueError):
        return np.nan
    if lo > 0 or hi < 0:                          # p is outside the range of prices that Black can give
        return np.nan
    try:
        return brentq(f, IV_LO, IV_HI, xtol=1e-10, rtol=1e-10, maxiter=200)
    except (RuntimeError, ValueError):
        return np.nan


def strike_from_delta(F, tau, sigma, delta_call):
    """Strike that has a given forward call delta (used to move the Vanna-Volga pillars, Chapter 3).

    K = F exp(-q sigma sqrt(tau) + sigma^2 tau / 2), with q = N^-1(delta_call).
    For a put, pass delta_call = 1 + delta_put (for example 0.75 for the 25-delta put)."""
    q = norm.ppf(delta_call)
    return F * np.exp(-q * sigma * np.sqrt(tau) + 0.5 * sigma ** 2 * tau)
