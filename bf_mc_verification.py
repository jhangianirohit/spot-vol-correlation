"""
MC verification: is the BF price fair given ν_log?

Setup: buy 25d strangle at wing vols (ATM + BF), sell at flat vol.
The extra premium = cost of the smile on the wings.
Under lognormal σ dynamics with ρ(spot,vol)=0, simulate, delta hedge,
and verify that volga P&L earned ≈ extra premium paid.

Expect ~92% recovery due to lognormal-dispersion bias.
"""

import numpy as np
from scipy.stats import norm
from bf_analysis import implied_nu_log, fair_bf, tenor_to_T


def bs_price_vec(S, K, T, sigma, is_call=True):
    """Vectorized BS price."""
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 0.001)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        return K * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta_vec(S, K, T, sigma, is_call=True):
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 0.001)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1


def bs_vega_scalar(S, K, T, sigma):
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def find_strike(S, T, sigma, target_delta, is_call=True):
    d1g = norm.ppf(target_delta) if is_call else norm.ppf(target_delta + 1)
    K = S * np.exp(-d1g * sigma * np.sqrt(T) + 0.5 * sigma**2 * T)
    for _ in range(100):
        d1 = (np.log(S/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
        delta = norm.cdf(d1) if is_call else norm.cdf(d1) - 1
        ddK = -norm.pdf(d1) / (K * sigma * np.sqrt(T))
        K -= (delta - target_delta) / ddK
        if abs(delta - target_delta) < 1e-10:
            break
    return K


S0 = 1.1415
sigma_atm = 0.077
T = 30 / 365
N_days = 30
dt = T / N_days

nu_log = 1.95
bf_pct = fair_bf(nu_log, sigma_atm * 100, T)
bf_decimal = bf_pct / 100  # in sigma units

K_call = find_strike(S0, T, sigma_atm, 0.25, True)
K_put = find_strike(S0, T, sigma_atm, -0.25, False)

# BF premium = extra cost of pricing wings at (ATM + BF) vs flat ATM
# This is what the market maker receives for selling vol-of-vol
sigma_wing = sigma_atm + bf_decimal
price_call_wing = bs_price_vec(S0, K_call, T, sigma_wing, True)
price_put_wing = bs_price_vec(S0, K_put, T, sigma_wing, False)
price_call_flat = bs_price_vec(S0, K_call, T, sigma_atm, True)
price_put_flat = bs_price_vec(S0, K_put, T, sigma_atm, False)

bf_premium = (price_call_wing + price_put_wing) - (price_call_flat + price_put_flat)

vega_25d = bs_vega_scalar(S0, K_call, T, sigma_atm)
approx_premium = 2 * vega_25d * bf_decimal

print('BF MC VERIFICATION')
print('=' * 65)
print(f'  S0 = {S0}, ATM = {sigma_atm*100:.2f}%, T = {N_days}d')
print(f'  ν_log = {nu_log:.2f}, fair BF = {bf_pct:.4f}%')
print(f'  K_call = {K_call:.6f}, K_put = {K_put:.6f}')
print(f'  σ_wing = {sigma_wing*100:.4f}%')
print(f'  BF premium (exact): {bf_premium:.8f}')
print(f'  BF premium (≈ 2×vega×BF): {approx_premium:.8f}')
print()

# MC: simulate spot + vol paths, delta hedge the strangle at current vol
N_paths = 200000
np.random.seed(42)

S_arr = np.full(N_paths, S0)
vol_arr = np.full(N_paths, sigma_atm)
net_pnl = np.zeros(N_paths)

for day in range(N_days):
    tau = T - day * dt
    if tau < 1e-8:
        break
    tau_next = max(tau - dt, 1e-8)

    # Delta of the 25d strangle at current vol (not wing vol — we hedge at market vol)
    delta_call = bs_delta_vec(S_arr, K_call, tau, vol_arr, True)
    delta_put = bs_delta_vec(S_arr, K_put, tau, vol_arr, False)
    delta_strangle = delta_call + delta_put

    # Independent BMs
    Z1 = np.random.standard_normal(N_paths)
    Z2 = np.random.standard_normal(N_paths)

    # Spot dynamics
    dS = S_arr * (np.exp(-0.5 * vol_arr**2 * dt + vol_arr * np.sqrt(dt) * Z1) - 1)

    # Vol dynamics (lognormal, independent of spot)
    vol_new = vol_arr * np.exp(-0.5 * nu_log**2 * dt + nu_log * np.sqrt(dt) * Z2)
    vol_new = np.maximum(vol_new, 0.005)

    S_new = S_arr + dS

    # MTM: price strangle at new (S, vol, tau)
    old_val = (bs_price_vec(S_arr, K_call, tau, vol_arr, True)
               + bs_price_vec(S_arr, K_put, tau, vol_arr, False))
    new_val = (bs_price_vec(S_new, K_call, tau_next, vol_new, True)
               + bs_price_vec(S_new, K_put, tau_next, vol_new, False))

    d_val = new_val - old_val
    hedge_pnl = -delta_strangle * dS

    net_pnl += d_val + hedge_pnl

    S_arr = S_new
    vol_arr = vol_new

mean_pnl = np.mean(net_pnl)
se_pnl = np.std(net_pnl) / np.sqrt(N_paths)
recovery = mean_pnl / bf_premium if bf_premium != 0 else 0

print(f'  Simulating {N_paths:,} paths (ρ_spot_vol = 0)...')
print()
print(f'  RESULTS')
print(f'  {"─"*50}')
print(f'  BF premium (extra wing cost):  {bf_premium:+.8f}')
print(f'  Mean hedged P&L (strangle):    {mean_pnl:+.8f}  (SE: {se_pnl:.8f})')
print(f'  Recovery ratio:                {recovery:.4f}')
print()
print(f'  Expected ~0.92 recovery (lognormal-dispersion bias).')
print(f'  {"PASS ✓" if 0.70 < recovery < 1.10 else "CHECK — outside expected range"}')

# Sweep across ν_log
print()
print(f'  SWEEP: recovery across different ν_log')
print(f'  {"ν_log":>8s}  {"BF%":>8s}  {"Premium":>12s}  {"Hedged PnL":>12s}  {"Recovery":>10s}')
print(f'  {"─"*8}  {"─"*8}  {"─"*12}  {"─"*12}  {"─"*10}')

N_sweep = 100000
for nu in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    bf_p = fair_bf(nu, sigma_atm*100, T)
    bf_d = bf_p / 100
    sw = sigma_atm + bf_d
    prem = (bs_price_vec(S0, K_call, T, sw, True) + bs_price_vec(S0, K_put, T, sw, False)
            - bs_price_vec(S0, K_call, T, sigma_atm, True) - bs_price_vec(S0, K_put, T, sigma_atm, False))

    np.random.seed(42)
    S_a = np.full(N_sweep, S0)
    v_a = np.full(N_sweep, sigma_atm)
    npnl = np.zeros(N_sweep)
    for day in range(N_days):
        tau = T - day * dt
        if tau < 1e-8: break
        tau_n = max(tau - dt, 1e-8)
        dc = bs_delta_vec(S_a, K_call, tau, v_a, True)
        dp = bs_delta_vec(S_a, K_put, tau, v_a, False)
        Z1 = np.random.standard_normal(N_sweep)
        Z2 = np.random.standard_normal(N_sweep)
        dS = S_a * (np.exp(-0.5*v_a**2*dt + v_a*np.sqrt(dt)*Z1) - 1)
        v_n = v_a * np.exp(-0.5*nu**2*dt + nu*np.sqrt(dt)*Z2)
        v_n = np.maximum(v_n, 0.005)
        S_n = S_a + dS
        ov = bs_price_vec(S_a, K_call, tau, v_a, True) + bs_price_vec(S_a, K_put, tau, v_a, False)
        nv = bs_price_vec(S_n, K_call, tau_n, v_n, True) + bs_price_vec(S_n, K_put, tau_n, v_n, False)
        npnl += (nv - ov) + (-(dc+dp) * dS)
        S_a, v_a = S_n, v_n

    rec = np.mean(npnl) / prem if prem != 0 else 0
    print(f'  {nu:8.2f}  {bf_p:8.4f}  {prem:+12.8f}  {np.mean(npnl):+12.8f}  {rec:10.4f}')
