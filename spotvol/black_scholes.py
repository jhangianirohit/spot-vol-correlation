"""
Black-Scholes functions for FX options. Zero rates by default.
All functions accept numpy arrays for S where noted.
"""

import numpy as np
from scipy.stats import norm


def d1(S, K, T, sigma, r_d=0.0, r_f=0.0):
    return (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, sigma, r_d=0.0, r_f=0.0):
    return d1(S, K, T, sigma, r_d, r_f) - sigma * np.sqrt(T)


def price(S, K, T, sigma, r_d=0.0, r_f=0.0, is_call=True):
    _d1 = d1(S, K, T, sigma, r_d, r_f)
    _d2 = _d1 - sigma * np.sqrt(T)
    if is_call:
        return S * np.exp(-r_f * T) * norm.cdf(_d1) - K * np.exp(-r_d * T) * norm.cdf(_d2)
    else:
        return K * np.exp(-r_d * T) * norm.cdf(-_d2) - S * np.exp(-r_f * T) * norm.cdf(-_d1)


def delta(S, K, T, sigma, r_d=0.0, r_f=0.0, is_call=True):
    _d1 = d1(S, K, T, sigma, r_d, r_f)
    if is_call:
        return np.exp(-r_f * T) * norm.cdf(_d1)
    else:
        return np.exp(-r_f * T) * (norm.cdf(_d1) - 1)


def vega(S, K, T, sigma, r_d=0.0, r_f=0.0):
    """dPrice/dSigma (per unit sigma)."""
    _d1 = d1(S, K, T, sigma, r_d, r_f)
    return S * np.exp(-r_f * T) * norm.pdf(_d1) * np.sqrt(T)


def vanna(S, K, T, sigma, r_d=0.0, r_f=0.0):
    """Analytical vanna: d(delta)/d(sigma) = d(vega)/d(S)."""
    _d1 = d1(S, K, T, sigma, r_d, r_f)
    _d2 = _d1 - sigma * np.sqrt(T)
    return -np.exp(-r_f * T) * norm.pdf(_d1) * _d2 / sigma


def vanna_numerical(S, K, T, sigma, bump_pct=0.01, r_d=0.0, r_f=0.0):
    """Numerical vanna: change in vega for a bump_pct (default 1%) spot move.
    Central difference for accuracy."""
    S_up = S * (1 + bump_pct)
    S_dn = S * (1 - bump_pct)
    return (vega(S_up, K, T, sigma, r_d, r_f) - vega(S_dn, K, T, sigma, r_d, r_f)) / 2


def volga(S, K, T, sigma, r_d=0.0, r_f=0.0):
    """d(vega)/d(sigma) = vega × d1 × d2 / sigma."""
    _d1 = d1(S, K, T, sigma, r_d, r_f)
    _d2 = _d1 - sigma * np.sqrt(T)
    _vega = vega(S, K, T, sigma, r_d, r_f)
    return _vega * _d1 * _d2 / sigma


def find_strike_for_delta(S, T, sigma, target_delta, r_d=0.0, r_f=0.0, is_call=True, tol=1e-10):
    """Find strike K such that BS delta = target_delta. Newton's method."""
    F = S * np.exp((r_d - r_f) * T)
    if is_call:
        d1_guess = norm.ppf(target_delta * np.exp(r_f * T))
    else:
        d1_guess = norm.ppf((target_delta + np.exp(-r_f * T)) * np.exp(r_f * T))
    K = F * np.exp(-d1_guess * sigma * np.sqrt(T) + 0.5 * sigma**2 * T)

    for _ in range(100):
        _d1 = d1(S, K, T, sigma, r_d, r_f)
        _delta = delta(S, K, T, sigma, r_d, r_f, is_call)
        ddelta_dK = -np.exp(-r_f * T) * norm.pdf(_d1) / (K * sigma * np.sqrt(T))
        err = _delta - target_delta
        if abs(err) < tol:
            break
        K = K - err / ddelta_dK

    return K
