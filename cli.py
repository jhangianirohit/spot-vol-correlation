#!/usr/bin/env python3
"""
Spot-Vol Correlation Tool — CLI

Usage:
  # Extract implied betas from market RR:
  python cli.py --pair EURUSD --spot 1.08 \
    1W 10.0 -2.0 \
    1M 8.3 -1.45

  # Each tenor is: LABEL ATM_VOL(%) RR_25D(%)

  # Compute fair RR from a given beta:
  python cli.py --pair EURUSD --spot 1.08 --beta -0.03 \
    1W 10.0 \
    1M 8.3
"""

import argparse
import sys

from spotvol.models import TenorInput
from spotvol.implied_beta import compute_implied_beta, compute_fair_rr


def parse_tenor_args(args):
    """Parse positional args: LABEL ATM_VOL [RR] LABEL ATM_VOL [RR] ..."""
    tenors = []
    i = 0
    while i < len(args):
        label = args[i].upper()
        i += 1
        if i >= len(args):
            raise ValueError(f"Missing ATM vol for tenor {label}")
        atm_vol = float(args[i]) / 100  # % to decimal
        i += 1

        rr = None
        if i < len(args):
            # Next arg could be RR (a number) or the next tenor label
            try:
                rr = float(args[i]) / 100  # % to decimal
                i += 1
            except ValueError:
                pass  # It's a tenor label, not RR

        tenors.append((label, atm_vol, rr))
    return tenors


def main():
    parser = argparse.ArgumentParser(description="Spot-Vol Correlation Tool")
    parser.add_argument("--pair", required=True, help="Currency pair (e.g., EURUSD)")
    parser.add_argument("--spot", required=True, type=float, help="Spot rate")
    parser.add_argument("--beta", type=float, default=None,
                        help="If provided, compute fair RR from this beta (Δσ per 1%% spot)")
    parser.add_argument("--sf", type=float, default=0.5,
                        help="Scaling factor (default 0.5)")
    parser.add_argument("tenor_data", nargs="*",
                        help="Tenor data: LABEL ATM_VOL(%%) [RR_25D(%%)] ...")

    args = parser.parse_args()
    tenors = parse_tenor_args(args.tenor_data)

    print(f"\n  {args.pair}  Spot = {args.spot}  SF = {args.sf}")
    print(f"  {'═' * 70}")

    if args.beta is not None:
        # Mode 2: compute fair RR from beta
        print(f"  Beta = {args.beta} (Δσ per 1% spot)")
        print()
        print(f"  {'Tenor':>6s}  {'ATM vol':>8s}  {'Fair RR':>10s}  {'25d Call vol':>12s}  {'25d Put vol':>12s}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*12}")

        for label, atm_vol, _ in tenors:
            fair_rr = compute_fair_rr(args.spot, label, atm_vol, args.beta, scaling_factor=args.sf)
            call_vol = atm_vol + fair_rr / 2
            put_vol = atm_vol - fair_rr / 2
            print(f"  {label:>6s}  {atm_vol*100:7.1f}%  {fair_rr*100:+9.2f}%  {call_vol*100:11.2f}%  {put_vol*100:11.2f}%")

    else:
        # Mode 1: extract implied beta from market RR
        tenor_inputs = []
        for label, atm_vol, rr in tenors:
            if rr is None:
                print(f"  ERROR: Missing RR for tenor {label}", file=sys.stderr)
                sys.exit(1)
            tenor_inputs.append(TenorInput(label=label, atm_vol=atm_vol, rr_25d=rr))

        results = compute_implied_beta(args.spot, tenor_inputs, scaling_factor=args.sf)

        print()
        print(f"  {'Tenor':>6s}  {'ATM':>6s}  {'RR':>8s}  {'β (per 1%)':>11s}  "
              f"{'β_sd':>10s}  {'vol/1σ spot':>12s}  {'daily σ':>10s}")
        print(f"  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*11}  "
              f"{'─'*10}  {'─'*12}  {'─'*10}")

        for r in results:
            print(f"  {r.label:>6s}  {r.atm_vol*100:5.1f}%  {r.rr_25d*100:+7.2f}%  {r.beta:+11.6f}  "
                  f"{r.beta_sd:+10.6f}  {abs(r.beta_sd)*100:11.2f}v  "
                  f"{r.daily_std*10000:9.0f}pip")

        if len(results) >= 2:
            print()
            r0, r1 = results[0], results[1]
            print(f"  β ratio {r0.label}/{r1.label}: {abs(r0.beta_sd/r1.beta_sd):.2f}x")

        print()
        for r in results:
            direction = "vol UP when spot DOWN" if r.beta < 0 else "vol UP when spot UP"
            print(f"  {r.label}: 1 daily σ ({r.daily_std*10000:.0f} pips) → "
                  f"vol moves {abs(r.beta_sd)*100:.2f}v ({direction})")
            print(f"       e.g., {r.atm_vol*100:.1f}% → "
                  f"{(r.atm_vol - r.beta_sd)*100:.1f}% (spot down 1σ), "
                  f"{(r.atm_vol + r.beta_sd)*100:.1f}% (spot up 1σ)")

    print()


if __name__ == "__main__":
    main()
