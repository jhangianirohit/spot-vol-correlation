"""
Currency basket beta decomposition.

Instead of decomposing cross RR through USD pairs only, extract a
"currency beta" for each currency using the full cross-section of
pair RRs, weighted by the basket vol decomposition.

Model: β(pair_AB) = w_A × β_A - w_B × β_B

Where:
  w_A, w_B = basket variance weights (how much of pair's vol is A vs B)
  β_A, β_B = currency-level betas (to be solved for)

Solve by weighted least squares across all observed pairs.
"""

import numpy as np

BETA_CONST = 1.4826
TENOR_DAYS = {
    'ON':1,'1D':1,'1W':7,'2W':14,'3W':21,'1M':30,'2M':60,
    '3M':91,'6M':182,'9M':274,'1Y':365,'2Y':730
}


def tenor_to_T(tenor):
    d = TENOR_DAYS.get(tenor.upper())
    return d / 365 if d else None


def get_currencies(pair):
    """Extract LHS and RHS 3-letter currency codes."""
    return pair[:3], pair[3:]


def pair_beta(rr_pct, vol_pct, tenor):
    """Implied beta from RR and ATM vol (both in %)."""
    T = tenor_to_T(tenor)
    if not T or vol_pct == 0:
        return None
    return BETA_CONST * rr_pct / (vol_pct * np.sqrt(T))


def fair_rr(beta, vol_pct, tenor):
    """Convert beta to RR (in %)."""
    T = tenor_to_T(tenor)
    if not T:
        return None
    return beta * vol_pct * np.sqrt(T) / BETA_CONST


def compute_basket_vols(pairs_data):
    """
    Compute basket volatility for each currency.

    BasketVol(X) = sqrt(mean of σ²(X,j) for all j paired with X)

    Args:
        pairs_data: {pair: vol} where vol is in decimal (e.g., 0.075)

    Returns:
        {ccy: basket_vol_decimal}
    """
    # Collect all vols by currency
    ccy_vols = {}  # {ccy: [list of pair vols²]}
    for pair, vol in pairs_data.items():
        lhs, rhs = get_currencies(pair)
        if lhs not in ccy_vols:
            ccy_vols[lhs] = []
        if rhs not in ccy_vols:
            ccy_vols[rhs] = []
        ccy_vols[lhs].append(vol ** 2)
        ccy_vols[rhs].append(vol ** 2)

    basket_vols = {}
    for ccy, vol2_list in ccy_vols.items():
        basket_vols[ccy] = np.sqrt(np.mean(vol2_list))

    return basket_vols


def compute_pair_weights(pair, basket_vols):
    """
    Decompose pair variance into LHS and RHS currency contributions.

    w_LHS = BasketVar(LHS) / (BasketVar(LHS) + BasketVar(RHS))
    w_RHS = BasketVar(RHS) / (BasketVar(LHS) + BasketVar(RHS))

    Returns (w_lhs, w_rhs) summing to 1.
    """
    lhs, rhs = get_currencies(pair)
    bv_lhs = basket_vols.get(lhs, 0) ** 2
    bv_rhs = basket_vols.get(rhs, 0) ** 2
    total = bv_lhs + bv_rhs
    if total == 0:
        return 0.5, 0.5
    return bv_lhs / total, bv_rhs / total


def solve_currency_betas(pairs_data, tenor):
    """
    Solve for currency betas from the cross-section of pair RRs.

    For each pair AB:
      observed_β(AB) = w_A × β_A - w_B × β_B

    Solve by least squares.

    Args:
        pairs_data: {pair: (vol_pct, rr_pct)} for a single tenor
        tenor: e.g., "1M"

    Returns:
        currency_betas: {ccy: beta}
        basket_vols: {ccy: vol_decimal}
        pair_results: [{pair, observed_beta, predicted_beta, w_lhs, w_rhs, ...}]
    """
    # Compute basket vols from the vol data
    vol_dict = {pair: vol / 100 for pair, (vol, rr) in pairs_data.items()}
    basket_vols = compute_basket_vols(vol_dict)

    # Build the list of currencies
    ccys = sorted(basket_vols.keys())
    ccy_idx = {c: i for i, c in enumerate(ccys)}
    N_ccy = len(ccys)

    # Build the system: for each pair with RR data
    rows = []
    obs_betas = []

    for pair, (vol_pct, rr_pct) in pairs_data.items():
        if rr_pct is None or vol_pct <= 0:
            continue
        beta = pair_beta(rr_pct, vol_pct, tenor)
        if beta is None:
            continue

        lhs, rhs = get_currencies(pair)
        if lhs not in ccy_idx or rhs not in ccy_idx:
            continue

        w_lhs, w_rhs = compute_pair_weights(pair, basket_vols)

        # Equation: beta = w_lhs × β_lhs - w_rhs × β_rhs
        row = np.zeros(N_ccy)
        row[ccy_idx[lhs]] = w_lhs
        row[ccy_idx[rhs]] = -w_rhs
        rows.append(row)
        obs_betas.append(beta)

    if len(rows) < N_ccy:
        return None, basket_vols, []

    A = np.array(rows)
    b = np.array(obs_betas)

    # Solve by least squares
    result = np.linalg.lstsq(A, b, rcond=None)
    ccy_betas_arr = result[0]

    currency_betas = {ccys[i]: ccy_betas_arr[i] for i in range(N_ccy)}

    # Compute predicted betas and residuals for each pair
    pair_results = []
    for pair, (vol_pct, rr_pct) in pairs_data.items():
        lhs, rhs = get_currencies(pair)
        w_lhs, w_rhs = compute_pair_weights(pair, basket_vols)
        beta_lhs = currency_betas.get(lhs, 0)
        beta_rhs = currency_betas.get(rhs, 0)

        pred_beta = w_lhs * beta_lhs - w_rhs * beta_rhs
        pred_rr = fair_rr(pred_beta, vol_pct, tenor)

        obs_beta = pair_beta(rr_pct, vol_pct, tenor) if rr_pct is not None else None

        pair_results.append({
            'pair': pair,
            'lhs': lhs, 'rhs': rhs,
            'vol_pct': vol_pct,
            'rr_pct': rr_pct,
            'w_lhs': w_lhs, 'w_rhs': w_rhs,
            'obs_beta': obs_beta,
            'pred_beta': pred_beta,
            'pred_rr': pred_rr,
            'residual': (rr_pct - pred_rr) if rr_pct is not None and pred_rr is not None else None,
        })

    return currency_betas, basket_vols, pair_results


# ═══════════════════════════════════════════════════════════════
# Test with realistic data
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # 1M tenor data: (vol%, rr%)
    pairs_1m = {
        'EURUSD': (7.53, -1.03),
        'USDJPY': (8.81, -0.46),
        'GBPUSD': (6.92, -0.69),
        'AUDUSD': (10.07, -0.98),
        'USDCHF': (8.12, -0.04),
        'USDCNH': (3.55, -0.25),
        'NZDUSD': (10.20, -0.85),
        # Crosses
        'EURJPY': (10.80, -0.45),
        'EURGBP': (5.20, -0.25),
        'EURCNH': (6.38, -0.65),
        'GBPJPY': (11.50, -0.55),
        'GBPAUD': (7.54, +0.19),
        'AUDJPY': (12.20, -0.90),
        'AUDNZD': (6.80, -0.10),
        'CHFJPY': (8.43, -0.71),
        'NZDJPY': (12.80, -0.95),
    }

    tenor = '1M'

    ccy_betas, basket_vols, results = solve_currency_betas(pairs_1m, tenor)

    if ccy_betas:
        print('CURRENCY BASKET BETA DECOMPOSITION (1M)')
        print('=' * 70)

        print('\nBasket Vols:')
        print(f'  {"Ccy":>5s}  {"Basket Vol":>10s}')
        print(f'  {"─"*5}  {"─"*10}')
        for ccy in sorted(basket_vols.keys()):
            print(f'  {ccy:>5s}  {basket_vols[ccy]*100:9.2f}%')

        print('\nCurrency Betas (solved by least squares):')
        print(f'  {"Ccy":>5s}  {"Beta":>8s}  {"Interpretation":>40s}')
        print(f'  {"─"*5}  {"─"*8}  {"─"*40}')
        for ccy in sorted(ccy_betas.keys(), key=lambda c: ccy_betas[c]):
            b = ccy_betas[ccy]
            interp = f'{ccy} strengthens 1% → vol moves {b:+.2f}v'
            print(f'  {ccy:>5s}  {b:+8.2f}  {interp}')

        print(f'\nPair Decomposition:')
        print(f'  {"Pair":>8s}  {"Vol%":>6s}  {"w_L":>5s}  {"w_R":>5s}  '
              f'{"Obs β":>7s}  {"Pred β":>7s}  {"Obs RR%":>8s}  {"Pred RR%":>9s}  {"Resid":>7s}')
        print(f'  {"─"*8}  {"─"*6}  {"─"*5}  {"─"*5}  '
              f'{"─"*7}  {"─"*7}  {"─"*8}  {"─"*9}  {"─"*7}')

        for r in sorted(results, key=lambda x: abs(x['residual'] or 0), reverse=True):
            obs_b = f"{r['obs_beta']:+7.2f}" if r['obs_beta'] is not None else '     —'
            obs_rr = f"{r['rr_pct']:+8.2f}" if r['rr_pct'] is not None else '       —'
            pred_rr = f"{r['pred_rr']:+9.2f}" if r['pred_rr'] is not None else '        —'
            resid = f"{r['residual']:+7.2f}" if r['residual'] is not None else '      —'
            print(f"  {r['pair']:>8s}  {r['vol_pct']:6.2f}  {r['w_lhs']:5.2f}  {r['w_rhs']:5.2f}  "
                  f"{obs_b}  {r['pred_beta']:+7.2f}  {obs_rr}  {pred_rr}  {resid}")

        # Summary stats
        residuals = [r['residual'] for r in results if r['residual'] is not None]
        rmse = np.sqrt(np.mean(np.array(residuals)**2))
        print(f'\n  RMSE of residuals: {rmse:.4f}%')
        print(f'  Max |residual|: {max(abs(r) for r in residuals):.4f}%')
    else:
        print('Not enough data to solve for currency betas.')
