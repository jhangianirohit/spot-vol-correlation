"""
Core computation: extract implied spot-vol beta from market RR,
or compute fair RR from a given beta.

Uses Rohit's spreadsheet approach:
  vol_premium_per_leg = (vanna/2) × (σ√T/1%)² × beta / vega
  RR ≈ call_premium - put_premium
  Invert to get beta from market RR.

The /2 is the scaling factor (~0.5) that accounts for vanna decaying
as spot moves away from inception over the option's life.

Beta convention:
  beta = vol points change per 1% spot move
  e.g., beta = -2.0 means: spot up 1% → vol falls by 2.0 vol points
  Internally stored in "vol points" (i.e., beta=-2.0 means Δσ = -0.02 per 1% spot).
  The _rr_factor works with decimal sigma, so we convert at the boundary.
"""

from spotvol import black_scholes as bs
from spotvol.models import TenorInput, BetaResult
from spotvol.tenors import tenor_to_T, tenor_to_ndays, DAYS_PER_YEAR


DEFAULT_SCALING_FACTOR = 0.5


def _compute_greeks(S, T, sigma, r_d=0.0, r_f=0.0):
    """Compute 25d strikes and Greeks for an RR.

    Vanna and vega are both analytical.
    Vanna is expressed "per 1% spot move" = analytical_vanna × 0.01 × S
    so the formula RR = (vanna/2) × (σ√T/1%)² × beta / vega works directly.
    """
    K_call = bs.find_strike_for_delta(S, T, sigma, 0.25, r_d, r_f, is_call=True)
    K_put  = bs.find_strike_for_delta(S, T, sigma, -0.25, r_d, r_f, is_call=False)

    # Vanna: analytical, converted to "per 1% spot move"
    # analytical vanna = dVega/dS (per unit S, per unit sigma)
    # per 1% spot = analytical × 0.01 × S
    vanna_call = bs.vanna(S, K_call, T, sigma, r_d, r_f) * 0.01 * S
    vanna_put  = bs.vanna(S, K_put,  T, sigma, r_d, r_f) * 0.01 * S

    # Vega: analytical (per unit sigma)
    vega_call = bs.vega(S, K_call, T, sigma, r_d, r_f)
    vega_put  = bs.vega(S, K_put,  T, sigma, r_d, r_f)

    return K_call, K_put, vanna_call, vanna_put, vega_call, vega_put


def _rr_factor(S, T, sigma, sf=DEFAULT_SCALING_FACTOR, r_d=0.0, r_f=0.0):
    """Compute the factor such that RR(decimal) = factor × beta(decimal per 1% spot).

    factor = (σ√T / 1%)² × [(vanna_call × sf) / vega_call - (vanna_put × sf) / vega_put]
    """
    K_call, K_put, vanna_call, vanna_put, vega_call, vega_put = \
        _compute_greeks(S, T, sigma, r_d, r_f)

    std_to_term = sigma * (T ** 0.5)  # σ√T in decimal
    std_ratio_sq = (std_to_term / 0.01) ** 2  # (σ√T / 1%)²

    factor_call = (vanna_call * sf) / vega_call
    factor_put  = (vanna_put * sf) / vega_put

    rr_factor = std_ratio_sq * (factor_call - factor_put)

    return rr_factor, K_call, K_put, vanna_call, vanna_put, vega_call, vega_put


def compute_implied_beta(
    spot: float,
    tenors: list[TenorInput],
    r_d: float = 0.0,
    r_f: float = 0.0,
    scaling_factor: float = DEFAULT_SCALING_FACTOR,
) -> list[BetaResult]:
    """Extract implied spot-vol beta from market RR at each tenor.

    Returns:
        list of BetaResult with beta in vol points per 1% spot move.
        e.g., beta = -2.6 means spot up 1% → vol down 2.6 vol points.
    """
    results = []

    for t in tenors:
        T = tenor_to_T(t.label)
        n_days = tenor_to_ndays(t.label)
        sigma = t.atm_vol

        factor, K_call, K_put, vanna_call, vanna_put, vega_call, vega_put = \
            _rr_factor(spot, T, sigma, scaling_factor, r_d, r_f)

        # Invert: beta_decimal = market_rr / factor
        # Then convert to vol points: beta_volpts = beta_decimal × 100
        beta_decimal = t.rr_25d / factor
        beta = beta_decimal * 100  # vol points per 1% spot

        # beta_sd: vol points change per 1 daily std dev of spot
        # daily std dev as % of spot = sigma * sqrt(1/365) * 100
        # beta is "vol points per 1% spot move", daily_std_pct is in %
        # so: beta_sd = beta × (daily_std_pct / 1) since beta is already per 1%
        daily_std_pct = sigma * (1 / DAYS_PER_YEAR) ** 0.5 * 100
        beta_sd = beta * daily_std_pct

        daily_std = spot * sigma * (1 / DAYS_PER_YEAR) ** 0.5

        results.append(BetaResult(
            label=t.label,
            T=T,
            n_days=n_days,
            atm_vol=sigma,
            rr_25d=t.rr_25d,
            K_call=K_call,
            K_put=K_put,
            vanna_call=vanna_call,
            vanna_put=vanna_put,
            vanna_rr=vanna_call - vanna_put,
            vega_call=vega_call,
            vega_put=vega_put,
            avg_vega=(vega_call + vega_put) / 2,
            beta=beta,
            beta_sd=beta_sd,
            daily_std=daily_std,
            daily_std_pct=daily_std_pct,
        ))

    return results


def compute_fair_rr(
    spot: float,
    label: str,
    atm_vol: float,
    beta: float,
    r_d: float = 0.0,
    r_f: float = 0.0,
    scaling_factor: float = DEFAULT_SCALING_FACTOR,
) -> float:
    """Compute fair RR (in decimal vol) given a spot-vol beta.

    Args:
        beta: vol points per 1% spot move (e.g., -2.6)

    Returns:
        Fair 25d RR in decimal vol (e.g., -0.02 for -2.0%)
    """
    T = tenor_to_T(label)
    factor, *_ = _rr_factor(spot, T, atm_vol, scaling_factor, r_d, r_f)
    # Convert beta from vol points to decimal: /100
    return factor * (beta / 100)
