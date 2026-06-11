"""
Butterfly / Vol-of-Vol (ν_log) framework.

Parallel to the RR / β architecture. The 25d BF prices vol-of-vol
through the volga P&L mechanism, just as the RR prices spot-vol
correlation through vanna.

Core formula (lognormal vol dynamics: dσ/σ = ν_log × dW):

    ν_log = √( BF / (0.075 × σ_ATM × T) )

where:
    0.075 = 0.5 × 0.337 × 0.9098 × 0.5

    0.5    = Taylor coefficient (½ × Volga × (Δσ)²)
    0.337  = path-averaged volga scaling factor from MC
             (analogous to the 0.5 vanna scaling for RR)
    0.9098 = inception volga ratio for 25d BF (= 2 × 0.4549)
    0.5    = BF premium normalization (premium = 2 × Vega × BF)

BF and σ_ATM in same units (both % or both decimal — ratio cancels).
T in years.
"""

import numpy as np

BF_CONSTANT = 0.075  # 0.5 × 0.337 × 0.9098 × 0.5

# β² contribution to the fly, MC-measured (bf_beta_convexity.py).
# Spot-driven vol-of-vol (= β in our units) does NOT add to the fly the
# way idiosyncratic ν does — it contributes slightly NEGATIVELY:
#   fair BF = 0.075 × ATM × T × (ν_idio² + BETA_BF_COEFF × β²)
# Measured: eff_ν² slope of −0.107·β² vs +0.90·ν_idio² ⇒ ratio ≈ −0.12.
# Economic reading: the fly prices vol variance ORTHOGONAL to spot;
# spot-spanned vol risk is hedgeable (and charged) via the RR, and at
# |ρ|=1 the vol-of-vol budget is spent on skew with interest (SABR
# 2−3ρ² < 0). For realistic β (≤1) the correction is sub-basis-point.
BETA_BF_COEFF = -0.12

TENOR_DAYS = {
    'ON':1,'1D':1,'1W':7,'2W':14,'3W':21,'1M':30,'2M':60,
    '3M':91,'6M':182,'9M':274,'1Y':365,'2Y':730
}


def tenor_to_T(tenor):
    d = TENOR_DAYS.get(tenor.upper())
    return d / 365 if d else None


def implied_nu_log(bf, atm, T_years, beta=0.0):
    """
    Extract implied lognormal vol-of-vol from 25d butterfly.

    Args:
        bf: 25d butterfly in same units as atm (e.g., both in %)
        atm: ATM vol in same units as bf
        T_years: time to expiry in years
        beta: pair β (vol points per 1% spot move). If given, strips the
              small negative β² convexity contribution, so the result is
              the spot-ORTHOGONAL (idiosyncratic) vol-of-vol:
              ν_idio² = BF/(0.075·ATM·T) − BETA_BF_COEFF·β²

    Returns:
        ν_log: implied lognormal vol-of-vol (annualized, dimensionless)
              dσ/σ = ν_log × dW
    """
    if atm <= 0 or T_years <= 0 or bf <= 0:
        return None
    nu2 = bf / (BF_CONSTANT * atm * T_years) - BETA_BF_COEFF * beta**2
    if nu2 < 0:
        return None
    return np.sqrt(nu2)


def fair_bf(nu_log, atm, T_years, beta=0.0):
    """Fair BF from idiosyncratic ν_log and (optionally) pair β.

    fair BF = 0.075 × ATM × T × (ν_idio² + BETA_BF_COEFF × β²), floored at 0.
    """
    return max(0.0, BF_CONSTANT * atm * T_years * (nu_log**2 + BETA_BF_COEFF * beta**2))


def daily_vol_stdev(atm_pct, nu_log):
    """
    Daily 1-sigma stdev of ATM vol change, in vol points.

    Under dσ/σ = ν_log × dW:
        daily Δσ ~ σ × ν_log × √(1/252)

    Args:
        atm_pct: ATM vol in % (e.g., 7.5)
        nu_log: lognormal vol-of-vol

    Returns:
        Daily 1σ vol move in vol points
    """
    return atm_pct * nu_log / np.sqrt(252)


# ═══════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('BF → ν_log examples')
    print('=' * 65)
    print()

    # Example data: EURUSD
    cases = [
        ('EURUSD', 'ON',  11.88, 0.55),
        ('EURUSD', '1W',   9.03, 0.30),
        ('EURUSD', '1M',   7.70, 0.18),
        ('EURUSD', '3M',   7.25, 0.15),
        ('EURUSD', '1Y',   7.12, 0.12),
        ('USDJPY', '1M',  10.20, 0.25),
        ('GBPUSD', '1M',   6.92, 0.14),
    ]

    print(f'{"Pair":>8s}  {"Tenor":>5s}  {"ATM%":>6s}  {"BF%":>6s}  {"ν_log":>8s}  {"Daily σ move":>12s}')
    print(f'{"─"*8}  {"─"*5}  {"─"*6}  {"─"*6}  {"─"*8}  {"─"*12}')

    for pair, tenor, atm, bf in cases:
        T = tenor_to_T(tenor)
        nu = implied_nu_log(bf, atm, T)
        daily_move = daily_vol_stdev(atm, nu) if nu else None

        print(f'{pair:>8s}  {tenor:>5s}  {atm:6.2f}  {bf:6.2f}  {nu:8.2f}  '
              f'{daily_move:.2f}v/day' if nu else f'{pair:>8s}  {tenor:>5s}  —')

    print()
    print('Interpretation:')
    print('  ν_log = annualized lognormal vol-of-vol')
    print('  Daily σ move = ATM × ν_log / 16 ≈ expected daily 1σ vol change')
    print()

    # Verify round-trip
    print('Round-trip verification:')
    nu = implied_nu_log(0.18, 7.70, 30/365)
    bf_back = fair_bf(nu, 7.70, 30/365)
    print(f'  BF=0.18%, ATM=7.70%, 1M → ν_log={nu:.4f} → fair BF={bf_back:.4f}% ✓')
