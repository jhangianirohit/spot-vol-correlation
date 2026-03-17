"""Tests for the implied beta computation."""

import pytest
from spotvol.models import TenorInput
from spotvol.implied_beta import compute_implied_beta, compute_fair_rr
from spotvol import black_scholes as bs


def test_round_trip_beta_to_rr_to_beta():
    """Pick a beta, compute fair RR, then invert to recover the original beta."""
    spot = 1.08
    label = "1M"
    atm_vol = 0.083
    beta_original = -2.5  # vol points per 1% spot

    fair_rr = compute_fair_rr(spot, label, atm_vol, beta_original)
    results = compute_implied_beta(
        spot, [TenorInput(label=label, atm_vol=atm_vol, rr_25d=fair_rr)]
    )
    assert abs(results[0].beta - beta_original) < 1e-8


def test_round_trip_multiple_tenors():
    """Round trip for multiple tenors."""
    spot = 1.08
    betas = {"1W": -3.0, "1M": -1.2, "3M": -0.5}

    for label, beta in betas.items():
        fair_rr = compute_fair_rr(spot, label, 0.10, beta)
        results = compute_implied_beta(
            spot, [TenorInput(label=label, atm_vol=0.10, rr_25d=fair_rr)]
        )
        assert abs(results[0].beta - beta) < 1e-8, f"Failed for {label}"


def test_positive_beta_gives_positive_rr():
    fair_rr = compute_fair_rr(1.08, "1M", 0.10, beta=1.0)
    assert fair_rr > 0


def test_negative_beta_gives_negative_rr():
    fair_rr = compute_fair_rr(1.08, "1M", 0.10, beta=-1.0)
    assert fair_rr < 0


def test_zero_beta_gives_zero_rr():
    fair_rr = compute_fair_rr(1.08, "1M", 0.10, beta=0.0)
    assert abs(fair_rr) < 1e-12


def test_eurusd_market_example():
    """EURUSD: negative beta, 1W larger than 1M."""
    spot = 1.08
    tenors = [
        TenorInput(label="1W", atm_vol=0.10, rr_25d=-0.020),
        TenorInput(label="1M", atm_vol=0.083, rr_25d=-0.0145),
    ]
    results = compute_implied_beta(spot, tenors)

    assert results[0].beta < 0
    assert results[1].beta < 0
    assert abs(results[0].beta) > abs(results[1].beta)  # 1W more reactive


def test_strikes_are_25_delta():
    spot = 1.08
    results = compute_implied_beta(
        spot, [TenorInput(label="1M", atm_vol=0.10, rr_25d=-0.01)]
    )
    r = results[0]
    delta_call = bs.delta(spot, r.K_call, r.T, r.atm_vol, is_call=True)
    delta_put = bs.delta(spot, r.K_put, r.T, r.atm_vol, is_call=False)
    assert abs(delta_call - 0.25) < 1e-8
    assert abs(delta_put - (-0.25)) < 1e-8


def test_beta_units_are_vol_points():
    """Beta should be in vol points, not decimals.
    If RR=-2% and the math gives beta_decimal=-0.025, we should see beta=-2.5."""
    spot = 1.08
    results = compute_implied_beta(
        spot, [TenorInput(label="1M", atm_vol=0.10, rr_25d=-0.02)]
    )
    # Beta should be order of magnitude ~1-5, not ~0.01-0.05
    assert abs(results[0].beta) > 0.1
    assert abs(results[0].beta) < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
