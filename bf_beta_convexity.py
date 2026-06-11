"""
Does β² feed the butterfly, on top of idiosyncratic vol-of-vol?

Desk intuition: a long 25d fly is long "RR gamma" — vanna-flat at
inception, but ∂vanna/∂S ≠ 0, so the position gets longer RR as spot
rallies. If vol actually responds to spot (β ≠ 0), that developing
vanna monetizes. Equivalently: volga pays on (dσ)² regardless of what
drove the vol move; the RR only compensates the first-order dS·dσ
cross term, not the vol variance the spot-vol link generates.

Units make this clean. β = vol points per 1% spot move, so
    dσ/σ per 1% spot = β/σ_pct,  spot's annual 1σ in % = σ_pct
    ⇒ annualized stdev of dσ/σ driven by spot = β  (exactly)
The spot-driven component of lognormal vol-of-vol IS β. With
independent idiosyncratic vol shocks:

    dσ/σ = β·dW_spot + ν_idio·dW_idio,   ν_tot² = β² + ν_idio²

Hypothesis:  fair BF = 0.075 × ATM × T × (β² + ν_idio²)

Counter-hypothesis (SABR log-strike expansion): correlated vol-of-vol
contributes to curvature with coefficient (2−3ρ²)/24, i.e. the
spot-correlated part could even SUBTRACT. But the 25d delta-quoted fly
adds back skew² via asymmetric strike placement, so the net β²
coefficient must be measured, not asserted.

Experiment (extends bf_mc_verification.py, which validated the ρ=0
case at ~0.90 recovery):
  - Structure: vega-neutral fly = 25d strangle − k × DN straddle,
    k = strangle vega / straddle vega at inception. Static, delta-
    hedged daily at prevailing vol. (Memory note: must test the full
    fly, not the naked strangle.)
  - Mean hedged P&L is hedge-invariant (E[dS]=0), so BS delta is fine.
  - Grid over (β, ν_idio). For each cell, convert mean P&L into the
    implied fair BF (exact strangle repricing, brentq), then into an
    effective ν²: eff_ν² = BF_implied / (0.075 × ATM × T).
  - Test: eff_ν² ≈ r × (β² + ν_idio²) with the same recovery factor
    r ≈ 0.90 everywhere ⇒ β² enters one-for-one with ν_idio².
    The β=0 column is the validated baseline.
  - RR cancellation: wing premium is computed with both wings at
    ATM+BF; the ±RR/2 legs cancel to first order across the vega-
    matched 25d wings, so the fly premium isolates the BF.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# ─── BS helpers (vectorized, zero rates) ───────────────────────────

def bs_price_vec(S, K, T, sigma, is_call=True):
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 0.001)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * norm.cdf(d2)
    return K * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta_vec(S, K, T, sigma, is_call=True):
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 0.001)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1


def bs_vega(S, K, T, sigma):
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def find_strike(S, T, sigma, target_delta, is_call=True):
    d1g = norm.ppf(target_delta) if is_call else norm.ppf(target_delta + 1)
    K = S * np.exp(-d1g * sigma * np.sqrt(T) + 0.5 * sigma**2 * T)
    for _ in range(100):
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1) if is_call else norm.cdf(d1) - 1
        ddK = -norm.pdf(d1) / (K * sigma * np.sqrt(T))
        K -= (delta - target_delta) / ddK
        if abs(delta - target_delta) < 1e-10:
            break
    return K


# ─── Setup ─────────────────────────────────────────────────────────

S0 = 1.1415
sigma_atm = 0.077
T = 30 / 365
N_days = 30
dt = T / N_days
N_paths = 100_000
BF_CONSTANT = 0.075

K_call = find_strike(S0, T, sigma_atm, 0.25, True)
K_put = find_strike(S0, T, sigma_atm, -0.25, False)
K_dns = S0 * np.exp(0.5 * sigma_atm**2 * T)   # delta-neutral straddle

vega_strangle = bs_vega(S0, K_call, T, sigma_atm) + bs_vega(S0, K_put, T, sigma_atm)
vega_straddle = 2 * bs_vega(S0, K_dns, T, sigma_atm)
k_ratio = vega_strangle / vega_straddle


def fly_value(S, tau, vol):
    """Value of the vega-neutral fly: strangle − k × straddle."""
    strangle = (bs_price_vec(S, K_call, tau, vol, True)
                + bs_price_vec(S, K_put, tau, vol, False))
    straddle = (bs_price_vec(S, K_dns, tau, vol, True)
                + bs_price_vec(S, K_dns, tau, vol, False))
    return strangle - k_ratio * straddle


def fly_delta(S, tau, vol):
    dc = bs_delta_vec(S, K_call, tau, vol, True)
    dp = bs_delta_vec(S, K_put, tau, vol, False)
    dsc = bs_delta_vec(S, K_dns, tau, vol, True)
    dsp = bs_delta_vec(S, K_dns, tau, vol, False)
    return dc + dp - k_ratio * (dsc + dsp)


def strangle_premium(bf_decimal):
    """Extra premium of the fly when wings are priced at ATM+BF.
    The straddle leg is at ATM either way, so only the strangle moves."""
    sw = sigma_atm + bf_decimal
    return (bs_price_vec(S0, K_call, T, sw, True)
            + bs_price_vec(S0, K_put, T, sw, False)
            - bs_price_vec(S0, K_call, T, sigma_atm, True)
            - bs_price_vec(S0, K_put, T, sigma_atm, False))


def run_cell(beta, nu_idio, seed=42):
    """Delta-hedged fly P&L under dσ/σ = β·dW_spot + ν_idio·dW_idio."""
    rng = np.random.default_rng(seed)   # common random numbers per cell
    S = np.full(N_paths, S0)
    vol = np.full(N_paths, sigma_atm)
    pnl = np.zeros(N_paths)
    nu_tot2 = beta**2 + nu_idio**2

    for day in range(N_days):
        tau = T - day * dt
        if tau < 1e-8:
            break
        tau_next = max(tau - dt, 1e-8)

        delta = fly_delta(S, tau, vol)

        Z1 = rng.standard_normal(N_paths)
        Z2 = rng.standard_normal(N_paths)

        dS = S * (np.exp(-0.5 * vol**2 * dt + vol * np.sqrt(dt) * Z1) - 1)
        # vol shock loads β on the SPOT brownian, ν_idio on the orthogonal one
        vol_new = vol * np.exp(-0.5 * nu_tot2 * dt
                               + np.sqrt(dt) * (beta * Z1 + nu_idio * Z2))
        vol_new = np.maximum(vol_new, 0.005)
        S_new = S + dS

        pnl += (fly_value(S_new, tau_next, vol_new) - fly_value(S, tau, vol)
                - delta * dS)

        S, vol = S_new, vol_new

    return np.mean(pnl), np.std(pnl) / np.sqrt(N_paths)


def implied_bf_from_pnl(pnl):
    """Exact inversion: BF (decimal) whose wing premium equals the P&L."""
    if abs(pnl) < 1e-12:
        return 0.0
    lo, hi = (0.0, 0.10) if pnl > 0 else (-0.05, 0.0)
    return brentq(lambda x: strangle_premium(x) - pnl, lo, hi)


# ─── Grid ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('β² → BUTTERFLY MC EXPERIMENT')
    print('=' * 78)
    print(f'  S0={S0}, ATM={sigma_atm*100:.2f}%, T={N_days}d, '
          f'{N_paths:,} paths, daily hedge')
    print(f'  K_25c={K_call:.5f}, K_25p={K_put:.5f}, K_dns={K_dns:.5f}, '
          f'vega ratio k={k_ratio:.4f}')
    print()
    print(f'  Hypothesis: eff_ν² = BF_mc/(0.075·ATM·T) ≈ r·(β² + ν_idio²), '
          f'same r ≈ 0.90 everywhere')
    print()

    BETAS = [0.0, 0.5, 1.0, 1.5]
    NUS = [0.0, 1.0, 1.95]

    header = (f'  {"β":>5s}  {"ν_idio":>6s}  {"ν_tot²":>7s}  '
              f'{"PnL(1e-5)":>10s}  {"SE":>8s}  {"BF_mc%":>8s}  '
              f'{"eff_ν²":>7s}  {"r=eff/tot":>9s}')
    print(header)
    print('  ' + '─' * (len(header) - 2))

    cells = []
    for nu in NUS:
        for beta in BETAS:
            if nu == 0.0 and beta == 0.0:
                continue
            mean_pnl, se = run_cell(beta, nu)
            bf_mc = implied_bf_from_pnl(mean_pnl)          # decimal
            bf_mc_pct = bf_mc * 100
            eff_nu2 = bf_mc_pct / (BF_CONSTANT * sigma_atm * 100 * T)
            nu_tot2 = beta**2 + nu**2
            r = eff_nu2 / nu_tot2
            cells.append({'beta': beta, 'nu': nu, 'eff_nu2': eff_nu2,
                          'nu_tot2': nu_tot2, 'r': r})
            print(f'  {beta:5.2f}  {nu:6.2f}  {nu_tot2:7.3f}  '
                  f'{mean_pnl*1e5:10.3f}  {se*1e5:8.3f}  {bf_mc_pct:8.4f}  '
                  f'{eff_nu2:7.3f}  {r:9.3f}')
        print()

    # Isolate the β² coefficient: at each ν_idio, slope of eff_ν² vs β²
    print('  β² COEFFICIENT (slope of eff_ν² vs β² at fixed ν_idio,')
    print('  relative to the β=0 baseline recovery at that ν_idio):')
    for nu in NUS:
        sub = [c for c in cells if c['nu'] == nu]
        if len(sub) < 2:
            continue
        b2 = np.array([c['beta']**2 for c in sub])
        e = np.array([c['eff_nu2'] for c in sub])
        slope = np.polyfit(b2, e, 1)[0]
        base = next((c['r'] for c in sub if c['beta'] == 0.0), None)
        rel = f'  → vs baseline r={base:.3f}: ratio {slope/base:.3f}' if base else ''
        print(f'    ν_idio={nu:4.2f}:  d(eff_ν²)/d(β²) = {slope:+.3f}{rel}')

    print()
    print('  Reading: ratio ≈ 1 ⇒ β² enters the fly one-for-one with ν_idio²')
    print('           ⇒ fair BF = 0.075·ATM·T·(β² + ν_idio²)')
    print('           ratio < 1 ⇒ partial offset (SABR −3ρ² effect wins part way)')
