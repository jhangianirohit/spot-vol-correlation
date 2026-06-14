"""
MC verification: is the RR price fair given β (spot-vol beta)?

Parallel to bf_mc_verification.py (which validates the BF/volga side).
Here we validate the RR/vanna side that, until now, only had algebraic
round-trip tests — never a simulation.

Setup: a 25d risk reversal (long 25d call, short 25d put) carries a skew
markup of ≈ vega × RR in premium. Under spot-vol dynamics where vol moves
WITH spot (β ≠ 0), a delta-hedged RR earns vanna P&L from the realized
dS·dσ covariance. If β and RR are consistent, that realized vanna P&L
should recover the skew markup — the same way realized volga P&L recovers
the BF markup.

Vol dynamics (purely spot-driven, the cleanest test of β):
    dσ = β · σ · dW_spot            (β in vol points per 1% spot)
equivalently  dσ/σ = β · dW_spot, so β IS the spot-driven lognormal
vol-of-vol. This is exactly the link bf_beta_convexity.py relies on.

We report the RECOVERY RATIO = mean hedged P&L / skew markup. The BF side
lands ~0.90–0.92 (lognormal-dispersion bias). If the RR side lands in the
same ballpark, the 0.5 vanna scaling factor in implied_beta.py is fair.
A recovery far from ~0.9 means the implied β is mis-scaled — i.e. the
β we read out of market RR is biased by the formula, not just by risk
premium.
"""

import numpy as np
from scipy.stats import norm
from spotvol.implied_beta import compute_fair_rr
from spotvol.tenors import tenor_to_T


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


def bs_vega_scalar(S, K, T, sigma):
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


# ─── Setup ──────────────────────────────────────────────────────────
S0 = 1.1415
sigma_atm = 0.077
LABEL = '1M'
T = tenor_to_T(LABEL)
N_days = 30
dt = T / N_days

K_call = find_strike(S0, T, sigma_atm, 0.25, True)
K_put = find_strike(S0, T, sigma_atm, -0.25, False)


def skew_markup(rr_decimal):
    """Extra premium of a long-call/short-put RR when the smile prices
    the call at ATM+RR/2 and the put at ATM-RR/2, vs both at flat ATM.
    Position: +1 call, -1 put.  markup = (C(σc)-C(σ)) - (P(σp)-P(σ))."""
    sc = sigma_atm + rr_decimal / 2
    sp = sigma_atm - rr_decimal / 2
    c_skew = bs_price_vec(S0, K_call, T, sc, True)
    c_flat = bs_price_vec(S0, K_call, T, sigma_atm, True)
    p_skew = bs_price_vec(S0, K_put, T, sp, False)
    p_flat = bs_price_vec(S0, K_put, T, sigma_atm, False)
    return (c_skew - c_flat) - (p_skew - p_flat)


def skew_markup_T(rr_decimal, T_, Kc, Kp):
    """skew_markup for an arbitrary tenor/strikes (used in the tenor sweep)."""
    sc, sp = sigma_atm + rr_decimal / 2, sigma_atm - rr_decimal / 2
    return ((bs_price_vec(S0, Kc, T_, sc, True) - bs_price_vec(S0, Kc, T_, sigma_atm, True))
            - (bs_price_vec(S0, Kp, T_, sp, False) - bs_price_vec(S0, Kp, T_, sigma_atm, False)))


def run_cell(beta, n_paths, seed=42, T_=None, Kc=None, Kp=None,
             n_grid=None, hedge_every=1):
    """Delta-hedged RR (+call, -put) P&L under dσ/σ = β·dW_spot.

    Both legs repriced at a single evolving vol (no static smile carried);
    the skew markup is the inception edge, and realized vanna P&L is what
    we measure against it.

    Two knobs are deliberately separated:
      n_grid      — number of time steps the SPOT+VOL paths are simulated
                    on (numerical resolution of the continuous dynamics).
      hedge_every — rebalance the delta hedge every k grid steps; the hedge
                    delta is held constant in between.

    The mean P&L is hedge-invariant: the hedge term is −delta·dS with delta
    predictable and E[dS]=0, so E[hedge P&L]=0 at ANY hedge_every. Changing
    hedge_every moves only the variance (SE). Changing n_grid moves the mean,
    via Euler discretization bias, and the mean converges as n_grid grows."""
    T_ = T if T_ is None else T_
    Kc = K_call if Kc is None else Kc
    Kp = K_put if Kp is None else Kp
    n_grid = N_days if n_grid is None else n_grid
    h = T_ / n_grid

    rng = np.random.default_rng(seed)
    S = np.full(n_paths, S0)
    vol = np.full(n_paths, sigma_atm)
    pnl = np.zeros(n_paths)
    delta_rr = None

    for i in range(n_grid):
        tau = T_ - i * h
        if tau < 1e-8:
            break
        tau_next = max(tau - h, 1e-8)

        if i % hedge_every == 0:  # rebalance, else carry prior delta
            delta_rr = (bs_delta_vec(S, Kc, tau, vol, True)
                        - bs_delta_vec(S, Kp, tau, vol, False))

        Z1 = rng.standard_normal(n_paths)
        dS = S * (np.exp(-0.5 * vol**2 * h + vol * np.sqrt(h) * Z1) - 1)

        # vol moves WITH spot: dσ/σ = β·dW_spot  (β = vol pts per 1% spot)
        vol_new = vol * np.exp(-0.5 * beta**2 * h + beta * np.sqrt(h) * Z1)
        vol_new = np.maximum(vol_new, 0.005)
        S_new = S + dS

        old_val = (bs_price_vec(S, Kc, tau, vol, True)
                   - bs_price_vec(S, Kp, tau, vol, False))
        new_val = (bs_price_vec(S_new, Kc, tau_next, vol_new, True)
                   - bs_price_vec(S_new, Kp, tau_next, vol_new, False))

        pnl += (new_val - old_val) - delta_rr * dS
        S, vol = S_new, vol_new

    return np.mean(pnl), np.std(pnl) / np.sqrt(n_paths)


if __name__ == '__main__':
    print('RR / β  MC VERIFICATION  (vanna mechanism)')
    print('=' * 72)
    print(f'  S0={S0}, ATM={sigma_atm*100:.2f}%, T={N_days}d, '
          f'K_25c={K_call:.5f}, K_25p={K_put:.5f}')
    print(f'  vol dynamics: dσ/σ = β·dW_spot  (purely spot-driven)')
    print()
    print(f'  {"β":>6s}  {"fair RR%":>9s}  {"markup":>11s}  '
          f'{"hedged PnL":>11s}  {"SE":>9s}  {"recovery":>9s}')
    print('  ' + '─' * 64)

    N = 400_000
    for beta in [-3.0, -2.6, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
        rr = compute_fair_rr(S0, LABEL, sigma_atm, beta)  # decimal vol
        markup = skew_markup(rr)
        mean_pnl, se = run_cell(beta, N)
        rec = mean_pnl / markup if markup != 0 else float('nan')
        print(f'  {beta:6.2f}  {rr*100:9.4f}  {markup:+11.8f}  '
              f'{mean_pnl:+11.8f}  {se:9.8f}  {rec:9.4f}')

    print()
    print('  BF side recovers ~0.90–0.92. If the RR column matches that,')
    print('  the 0.5 vanna scaling is fair and implied β is unbiased by the')
    print('  formula (any excess in market β is then genuine risk premium).')
    print('  Systematic departure ⇒ the formula itself biases implied β.')

    # ─── Stress A: hedge frequency only ─────────────────────────────
    # Fix a fine path grid (4x/day) and change ONLY how often the delta is
    # rebalanced. The MEAN recovery is hedge-invariant; only the SE (your
    # real-world tracking error) should move.
    beta_s = -2.0
    rr_s = compute_fair_rr(S0, LABEL, sigma_atm, beta_s)
    mk_s = skew_markup(rr_s)
    FINE = 120  # ~4x/day over a 30d month
    print()
    print('  STRESS A — vary delta-hedge frequency on a FIXED fine grid (4x/day):')
    print(f'    {"rebalance":<24s}{"recovery":>9s}{"SE":>9s}')
    for he, lab in [(1, 'every step (4x/day)'), (4, 'daily'),
                    (20, 'weekly'), (FINE, 'never')]:
        mp, se = run_cell(beta_s, 400_000, n_grid=FINE, hedge_every=he)
        print(f'    {lab:<24s}{mp/mk_s:9.4f}{se/abs(mk_s):9.4f}')
    print('    → mean flat ⇒ hedge frequency sets VARIANCE, not the edge.')

    # ─── Stress B: path-grid granularity ────────────────────────────
    # Hedge every step; vary the simulation resolution. This is the Euler
    # discretization bias — the mean converges as the grid refines. (The
    # earlier "weekly → 0.72" was THIS, a 4-step grid, mislabeled.)
    print()
    print('  STRESS B — vary path-simulation granularity (hedge every step):')
    print(f'    {"grid steps":<24s}{"recovery":>9s}{"SE":>9s}')
    for ng in [4, 8, 30, 120, 480]:
        mp, se = run_cell(beta_s, 400_000, n_grid=ng, hedge_every=1)
        print(f'    {str(ng) + " steps":<24s}{mp/mk_s:9.4f}{se/abs(mk_s):9.4f}')
    print('    → converges to ~0.98 from below; the ~2% gap is the genuine')
    print('      lognormal-dispersion (Jensen) bias, NOT risk premium.')

    # ─── Stress C: across tenors (fine grid) ────────────────────────
    print()
    print('  STRESS C — recovery across tenors (β=-2.0, ~4x/day grid):')
    print(f'    {"tenor":<24s}{"recovery":>9s}{"SE":>9s}')
    for lbl, ng in [('1W', 28), ('1M', 120), ('3M', 360)]:
        T_l = tenor_to_T(lbl)
        Kc_l = find_strike(S0, T_l, sigma_atm, 0.25, True)
        Kp_l = find_strike(S0, T_l, sigma_atm, -0.25, False)
        rr_l = compute_fair_rr(S0, lbl, sigma_atm, beta_s)
        mk_l = skew_markup_T(rr_l, T_l, Kc_l, Kp_l)
        mp, se = run_cell(beta_s, 400_000, T_=T_l, Kc=Kc_l, Kp=Kp_l,
                          n_grid=ng, hedge_every=1)
        print(f'    {lbl:<24s}{mp/mk_l:9.4f}{se/abs(mk_l):9.4f}')

    print()
    print('  Risk premium proper lives in the GAP between implied β (this MC, fed')
    print('  by market RR) and realized β (regress dσ on dS/S in the data).')
    print('  Same logic for ν_log vs realized vol-of-vol. Recovery > 1 when fed')
    print('  the MARKET RR ⇒ the RR was cheap vs realized (a buy); < ~0.98 ⇒ rich.')
