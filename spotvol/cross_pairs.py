"""
Cross-pair RR decomposition.

Given USD pair vols and RRs, predict cross-pair RRs and compare to market.
"""

import numpy as np
from dataclasses import dataclass
from spotvol.models import TenorInput
from spotvol.implied_beta import compute_implied_beta, compute_fair_rr
from spotvol.tenors import tenor_to_T


# Standard cross → USD leg mapping
# Cross = leg1 / leg2 (both vs USD)
# e.g., EURCNH: EURUSD / CNHUSD, so leg1=EURUSD, leg2=CNHUSD
CROSS_LEGS = {
    "EURJPY":  ("EURUSD", "JPYUSD"),
    "EURGBP":  ("EURUSD", "GBPUSD"),
    "EURCHF":  ("EURUSD", "CHFUSD"),
    "EURCNH":  ("EURUSD", "CNHUSD"),
    "EURAUD":  ("EURUSD", "AUDUSD"),
    "EURNOK":  ("EURUSD", "NOKUSD"),
    "EURSEK":  ("EURUSD", "SEKUSD"),
    "GBPJPY":  ("GBPUSD", "JPYUSD"),
    "GBPCHF":  ("GBPUSD", "CHFUSD"),
    "AUDJPY":  ("AUDUSD", "JPYUSD"),
    "AUDNZD":  ("AUDUSD", "NZDUSD"),
    "AUDSGD":  ("AUDUSD", "SGDUSD"),
    "AUDCNH":  ("AUDUSD", "CNHUSD"),
    "NZDJPY":  ("NZDUSD", "JPYUSD"),
    "CADJPY":  ("CADUSD", "JPYUSD"),
    "CADCHF":  ("CADUSD", "CHFUSD"),
    "NOKJPY":  ("NOKUSD", "JPYUSD"),
    "NOKSEK":  ("NOKUSD", "SEKUSD"),
    "CHFJPY":  ("CHFUSD", "JPYUSD"),
    "SGDJPY":  ("SGDUSD", "JPYUSD"),
    "CNHJPY":  ("CNHUSD", "JPYUSD"),
}

# For pairs quoted as USDXXX, the XXX vs USD pair is inverted
# e.g., USDJPY: we need JPYUSD vol. JPYUSD vol = USDJPY vol (symmetric).
# The RR sign flips: JPYUSD RR = -USDJPY RR
# We handle this by allowing input as USDXXX and converting internally.
USD_PAIR_ALIASES = {
    "USDJPY": ("JPYUSD", -1),   # JPYUSD vol = USDJPY vol, RR sign flips
    "USDCHF": ("CHFUSD", -1),
    "USDCNH": ("CNHUSD", -1),
    "USDSGD": ("SGDUSD", -1),
    "USDNOK": ("NOKUSD", -1),
    "USDSEK": ("SEKUSD", -1),
    "USDCAD": ("CADUSD", -1),
    # These are already quoted as XXXUSD
    "EURUSD": ("EURUSD", 1),
    "GBPUSD": ("GBPUSD", 1),
    "AUDUSD": ("AUDUSD", 1),
    "NZDUSD": ("NZDUSD", 1),
}


@dataclass
class CrossResult:
    cross: str
    tenor: str
    leg1: str
    leg2: str
    rho: float
    beta_leg1: float
    beta_leg2: float
    vol_w_leg1: float
    vol_w_leg2: float
    spot_w_leg1: float
    spot_w_leg2: float
    beta_cross: float
    predicted_rr: float  # in decimal
    market_rr: float | None  # in decimal, None if not provided
    residual: float | None  # market - predicted, in decimal


def normalize_pair(pair: str) -> tuple[str, float, float]:
    """Convert any USD pair to XXXUSD convention.
    Returns (normalized_name, vol_multiplier, rr_multiplier).
    vol is always positive so multiplier is always 1.
    RR sign flips for USDXXX pairs.
    """
    pair = pair.upper()
    if pair in USD_PAIR_ALIASES:
        name, rr_sign = USD_PAIR_ALIASES[pair]
        return name, 1.0, float(rr_sign)
    # Already in XXXUSD form or unknown
    return pair, 1.0, 1.0


def decompose_cross(
    cross: str,
    tenor: str,
    vol_data: dict,  # {pair: {tenor: (vol, rr)}}
    spot_approx: float = 1.0,  # approximate spot for strike finding (doesn't affect beta much)
) -> CrossResult | None:
    """Decompose a single cross pair at a single tenor.

    vol_data should have entries for both USD legs and the cross itself.
    Vols and RRs in decimal (e.g., 0.075 for 7.5%, -0.01 for -1.0%).
    """
    cross = cross.upper()
    tenor = tenor.upper()

    if cross not in CROSS_LEGS:
        return None

    leg1_name, leg2_name = CROSS_LEGS[cross]

    # Get data for each leg and the cross
    if leg1_name not in vol_data or tenor not in vol_data[leg1_name]:
        return None
    if leg2_name not in vol_data or tenor not in vol_data[leg2_name]:
        return None
    if cross not in vol_data or tenor not in vol_data[cross]:
        return None

    sigma1, rr1 = vol_data[leg1_name][tenor]
    sigma2, rr2 = vol_data[leg2_name][tenor]
    sigma_cross, rr_cross_market = vol_data[cross][tenor]

    T = tenor_to_T(tenor)

    # Implied correlation
    rho_num = sigma1**2 + sigma2**2 - sigma_cross**2
    rho_den = 2 * sigma1 * sigma2
    if rho_den == 0:
        return None
    rho = rho_num / rho_den
    rho = max(-1.0, min(1.0, rho))  # clamp

    var_cross = sigma_cross**2

    # Implied betas of USD legs
    r1 = compute_implied_beta(spot_approx, [TenorInput(tenor, sigma1, rr1)])[0]
    r2 = compute_implied_beta(spot_approx, [TenorInput(tenor, sigma2, rr2)])[0]

    # Vol weights
    vol_w1 = (sigma1 - rho * sigma2) / sigma_cross
    vol_w2 = (sigma2 - rho * sigma1) / sigma_cross

    # Spot weights
    spot_w1 = (sigma1**2 - rho * sigma1 * sigma2) / var_cross
    spot_w2 = (rho * sigma1 * sigma2 - sigma2**2) / var_cross

    # Cross beta
    beta_cross = vol_w1 * r1.beta * spot_w1 + vol_w2 * r2.beta * spot_w2

    # Predicted RR
    predicted_rr = compute_fair_rr(spot_approx, tenor, sigma_cross, beta_cross)

    # Residual
    residual = None
    if rr_cross_market is not None:
        residual = rr_cross_market - predicted_rr

    return CrossResult(
        cross=cross,
        tenor=tenor,
        leg1=leg1_name,
        leg2=leg2_name,
        rho=rho,
        beta_leg1=r1.beta,
        beta_leg2=r2.beta,
        vol_w_leg1=vol_w1,
        vol_w_leg2=vol_w2,
        spot_w_leg1=spot_w1,
        spot_w_leg2=spot_w2,
        beta_cross=beta_cross,
        predicted_rr=predicted_rr,
        market_rr=rr_cross_market,
        residual=residual,
    )


def parse_vol_file(text: str) -> dict:
    """Parse a vol/RR data file into vol_data dict.

    Expected format (tab/space/comma separated):
      pair  tenor  vol(%)  rr(%)

    e.g.:
      EURUSD  1M  7.53  -1.03
      USDCNH  1M  3.55  -0.25
      EURCNH  1M  6.38  -0.65

    For USDXXX pairs, the RR sign is automatically flipped to XXXUSD convention.

    Returns: {normalized_pair: {tenor: (vol_decimal, rr_decimal)}}
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

        # Normalize to XXXUSD
        norm_pair, vol_mult, rr_mult = normalize_pair(pair)

        # For crosses, keep as-is
        if pair in CROSS_LEGS:
            norm_pair = pair
            rr_mult = 1.0

        if norm_pair not in vol_data:
            vol_data[norm_pair] = {}

        vol_data[norm_pair][tenor] = (vol * vol_mult, rr * rr_mult if rr is not None else None)

    return vol_data
