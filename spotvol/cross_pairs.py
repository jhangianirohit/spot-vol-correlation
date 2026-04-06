"""
Cross-pair RR decomposition.

Convention: RR is always from LHS currency perspective.
  Positive RR = LHS calls over LHS puts (vol up when LHS strengthens)
  Negative RR = LHS puts over LHS calls (vol up when LHS weakens)

Cross = leg1 × leg2 when second leg is USDXXX
Cross = leg1 / leg2 when second leg is XXXUSD

Example: EURCNH = EURUSD × USDCNH  (sign = +1)
         EURGBP = EURUSD / GBPUSD  (sign = -1)
"""

import numpy as np
from dataclasses import dataclass
from spotvol.models import TenorInput
from spotvol.implied_beta import compute_implied_beta, compute_fair_rr
from spotvol.tenors import tenor_to_T


# Cross → (leg1, leg2, sign)
# sign = +1: cross = leg1 × leg2,  d(cross) = d(leg1) + d(leg2)
# sign = -1: cross = leg1 / leg2,  d(cross) = d(leg1) - d(leg2)
CROSS_LEGS = {
    # EUR crosses
    "EURJPY":  ("EURUSD", "USDJPY",  +1),
    "EURCHF":  ("EURUSD", "USDCHF",  +1),
    "EURCNH":  ("EURUSD", "USDCNH",  +1),
    "EURGBP":  ("EURUSD", "GBPUSD",  -1),
    "EURAUD":  ("EURUSD", "AUDUSD",  -1),
    "EURNZD":  ("EURUSD", "NZDUSD",  -1),
    "EURNOK":  ("EURUSD", "USDNOK",  +1),
    "EURSEK":  ("EURUSD", "USDSEK",  +1),
    "EURCAD":  ("EURUSD", "USDCAD",  +1),
    "EURSGD":  ("EURUSD", "USDSGD",  +1),
    # GBP crosses
    "GBPJPY":  ("GBPUSD", "USDJPY",  +1),
    "GBPCHF":  ("GBPUSD", "USDCHF",  +1),
    "GBPCNH":  ("GBPUSD", "USDCNH",  +1),
    "GBPAUD":  ("GBPUSD", "AUDUSD",  -1),
    "GBPNZD":  ("GBPUSD", "NZDUSD",  -1),
    "GBPCAD":  ("GBPUSD", "USDCAD",  +1),
    # AUD crosses
    "AUDJPY":  ("AUDUSD", "USDJPY",  +1),
    "AUDCHF":  ("AUDUSD", "USDCHF",  +1),
    "AUDCNH":  ("AUDUSD", "USDCNH",  +1),
    "AUDNZD":  ("AUDUSD", "NZDUSD",  -1),
    "AUDSGD":  ("AUDUSD", "USDSGD",  +1),
    "AUDCAD":  ("AUDUSD", "USDCAD",  +1),
    # NZD crosses
    "NZDJPY":  ("NZDUSD", "USDJPY",  +1),
    "NZDCHF":  ("NZDUSD", "USDCHF",  +1),
    "NZDCAD":  ("NZDUSD", "USDCAD",  +1),
    # CAD crosses
    "CADJPY":  ("USDJPY", "USDCAD",  -1),  # CADJPY = USDJPY / USDCAD
    "CADCHF":  ("USDCHF", "USDCAD",  -1),  # CADCHF = USDCHF / USDCAD
    # NOK/SEK
    "NOKSEK":  ("USDSEK", "USDNOK",  -1),  # NOKSEK = USDSEK / USDNOK
    # CHF
    "CHFJPY":  ("USDJPY", "USDCHF",  -1),  # CHFJPY = USDJPY / USDCHF
    # SGD
    "SGDJPY":  ("USDJPY", "USDSGD",  -1),  # SGDJPY = USDJPY / USDSGD
    # CNH
    "CNHJPY":  ("USDJPY", "USDCNH",  -1),  # CNHJPY = USDJPY / USDCNH
}


@dataclass
class CrossResult:
    cross: str
    tenor: str
    leg1: str
    leg2: str
    sign: int          # +1 or -1
    rho: float         # implied correlation between leg1 and leg2 returns
    beta_leg1: float
    beta_leg2: float
    vol_w_leg1: float
    vol_w_leg2: float
    spot_w_leg1: float
    spot_w_leg2: float
    beta_cross: float
    predicted_rr: float
    market_rr: float | None
    residual: float | None


def decompose_cross(
    cross: str,
    tenor: str,
    vol_data: dict,
    spot_approx: float = 1.0,
) -> CrossResult | None:
    """Decompose a cross pair RR into USD pair components.

    vol_data: {pair: {tenor: (vol_decimal, rr_decimal)}}
    Pairs are in market quoting convention (EURUSD, USDCNH, etc.).
    RR follows LHS convention (positive = LHS calls over puts).
    """
    cross = cross.upper()
    tenor = tenor.upper()

    if cross not in CROSS_LEGS:
        return None

    leg1, leg2, sign = CROSS_LEGS[cross]

    if leg1 not in vol_data or tenor not in vol_data[leg1]:
        return None
    if leg2 not in vol_data or tenor not in vol_data[leg2]:
        return None
    if cross not in vol_data or tenor not in vol_data[cross]:
        return None

    sigma1, rr1 = vol_data[leg1][tenor]
    sigma2, rr2 = vol_data[leg2][tenor]
    sigma_cross, rr_cross_market = vol_data[cross][tenor]

    T = tenor_to_T(tenor)
    var_cross = sigma_cross ** 2

    # Implied correlation between leg1 and leg2 returns
    # σ²(cross) = σ²₁ + σ²₂ + 2s×ρ×σ₁×σ₂
    # ρ = (σ²_cross - σ²₁ - σ²₂) / (2s×σ₁×σ₂)
    rho_num = sigma_cross**2 - sigma1**2 - sigma2**2
    rho_den = 2 * sign * sigma1 * sigma2
    if rho_den == 0:
        return None
    rho = rho_num / rho_den
    rho = max(-1.0, min(1.0, rho))

    # Implied betas of each leg (in their own quoting convention)
    r1 = compute_implied_beta(spot_approx, [TenorInput(tenor, sigma1, rr1)])[0]
    r2 = compute_implied_beta(spot_approx, [TenorInput(tenor, sigma2, rr2)])[0]

    # Vol sensitivities: ∂σ_cross/∂σ_leg
    # ∂σ/∂σ₁ = (σ₁ + s×ρ×σ₂) / σ_cross
    # ∂σ/∂σ₂ = (σ₂ + s×ρ×σ₁) / σ_cross
    vol_w1 = (sigma1 + sign * rho * sigma2) / sigma_cross
    vol_w2 = (sigma2 + sign * rho * sigma1) / sigma_cross

    # Spot decomposition: when cross moves 1%, expected leg moves
    # Cov(leg1, cross) = σ²₁ + s×ρ×σ₁×σ₂
    # Cov(leg2, cross) = ρ×σ₁×σ₂ + s×σ²₂
    spot_w1 = (sigma1**2 + sign * rho * sigma1 * sigma2) / var_cross
    spot_w2 = (rho * sigma1 * sigma2 + sign * sigma2**2) / var_cross

    # Cross beta
    beta_cross = vol_w1 * r1.beta * spot_w1 + vol_w2 * r2.beta * spot_w2

    # Predicted RR
    predicted_rr = compute_fair_rr(spot_approx, tenor, sigma_cross, beta_cross)

    residual = None
    if rr_cross_market is not None:
        residual = rr_cross_market - predicted_rr

    return CrossResult(
        cross=cross, tenor=tenor, leg1=leg1, leg2=leg2, sign=sign,
        rho=rho,
        beta_leg1=r1.beta, beta_leg2=r2.beta,
        vol_w_leg1=vol_w1, vol_w_leg2=vol_w2,
        spot_w_leg1=spot_w1, spot_w_leg2=spot_w2,
        beta_cross=beta_cross,
        predicted_rr=predicted_rr,
        market_rr=rr_cross_market,
        residual=residual,
    )


def parse_vol_file(text: str) -> dict:
    """Parse a vol/RR data file.

    Format (tab/space/comma separated):
      pair  tenor  vol(%)  rr(%)

    e.g.:
      EURUSD  1M  7.53  -1.03
      USDCNH  1M  3.55  -0.25
      EURCNH  1M  6.38  -0.65

    No sign flipping. Everything in natural market convention.
    RR follows LHS convention: positive = LHS calls over puts.

    Returns: {pair: {tenor: (vol_decimal, rr_decimal)}}
    """
    vol_data = {}

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").replace("\t", " ").split()
        if len(parts) < 3:
            continue

        pair = parts[0].upper()
        tenor = parts[1].upper()
        try:
            vol = float(parts[2]) / 100
            rr = float(parts[3]) / 100 if len(parts) >= 4 else None
        except (ValueError, IndexError):
            continue

        if pair not in vol_data:
            vol_data[pair] = {}
        vol_data[pair][tenor] = (vol, rr)

    return vol_data
