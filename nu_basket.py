"""
Currency-level ν_log basket decomposition.

Mirrors the β basket solve. For each pair AB:

    ν_log²_AB = w_A² × ν_A² + w_B² × ν_B² + 2 × w_A × w_B × ρ_vol × ν_A × ν_B

With ρ_vol = 0 (baseline: independent currency vol-of-vols):

    ν_log²_AB = w_A² × ν_A² + w_B² × ν_B²

Solve for currency-level ν_log via least squares.

Normalization: since ν is strictly positive, we anchor on
the mean (subtract mean from all, then add back after solving).
In practice: solve for ν² values, then take sqrt.
"""

import numpy as np
from bf_analysis import implied_nu_log, fair_bf, daily_vol_stdev, tenor_to_T


def compute_basket_vols(pairs_vols):
    """Same basket vol computation as in beta framework."""
    ccy_vols = {}
    for pair, vol in pairs_vols.items():
        lhs, rhs = pair[:3], pair[3:]
        ccy_vols.setdefault(lhs, []).append(vol**2)
        ccy_vols.setdefault(rhs, []).append(vol**2)
    return {c: np.sqrt(np.mean(v)) for c, v in ccy_vols.items()}


def pair_weights(pair, basket_vols):
    lhs, rhs = pair[:3], pair[3:]
    bv1 = basket_vols.get(lhs, 0)**2
    bv2 = basket_vols.get(rhs, 0)**2
    total = bv1 + bv2
    return (bv1/total, bv2/total) if total > 0 else (0.5, 0.5)


def solve_currency_nu(pairs_data, tenor, rho_vol=0.0):
    """
    Solve for currency-level ν_log from cross-section of pair BFs.

    For each pair AB:
        ν²_observed = BF / (0.075 × ATM × T)
        ν²_model    = w_A² × ν_A² + w_B² × ν_B²  (with ρ_vol=0)

    Solve: ν²_observed = A × [ν_A², ν_B², ...]  by least squares.

    Args:
        pairs_data: {pair: (atm_pct, bf_pct)} for a single tenor
        tenor: e.g., '1M'
        rho_vol: correlation between currency vol-of-vols (default 0)

    Returns:
        ccy_nu: {ccy: ν_log}
        basket_vols: {ccy: vol_decimal}
        pair_results: list of dicts
    """
    T = tenor_to_T(tenor)
    if not T:
        return None, {}, []

    # Basket vols from ATM data
    vol_dict = {pair: atm/100 for pair, (atm, bf) in pairs_data.items()}
    bvols = compute_basket_vols(vol_dict)

    ccys = sorted(bvols.keys())
    ccy_idx = {c: i for i, c in enumerate(ccys)}
    N = len(ccys)

    # Build system: for each pair with BF data
    rows = []
    obs_nu2 = []
    pair_list = []

    for pair, (atm, bf) in pairs_data.items():
        if bf is None or bf <= 0 or atm <= 0:
            continue

        nu = implied_nu_log(bf, atm, T)
        if nu is None:
            continue

        lhs, rhs = pair[:3], pair[3:]
        if lhs not in ccy_idx or rhs not in ccy_idx:
            continue

        wL, wR = pair_weights(pair, bvols)

        # Equation: ν²_obs = w_L² × ν_L² + w_R² × ν_R² + 2×w_L×w_R×ρ_vol×ν_L×ν_R
        # With ρ_vol=0: ν²_obs = w_L² × ν_L² + w_R² × ν_R²
        # This is linear in ν² values
        row = np.zeros(N)
        row[ccy_idx[lhs]] = wL**2
        row[ccy_idx[rhs]] = wR**2
        # If ρ_vol ≠ 0, we'd need to add cross terms (nonlinear — would need iterative solve)
        rows.append(row)
        obs_nu2.append(nu**2)
        pair_list.append({
            'pair': pair, 'lhs': lhs, 'rhs': rhs,
            'atm': atm, 'bf': bf, 'wL': wL, 'wR': wR,
            'obs_nu': nu, 'obs_nu2': nu**2
        })

    if len(rows) < N:
        return None, bvols, []

    A = np.array(rows)
    b = np.array(obs_nu2)

    # Solve for ν² values (must be non-negative)
    # Use non-negative least squares
    from scipy.optimize import nnls
    nu2_solution, residual = nnls(A, b)

    ccy_nu = {ccys[i]: np.sqrt(nu2_solution[i]) for i in range(N)}

    # Compute predicted values
    for p in pair_list:
        wL, wR = p['wL'], p['wR']
        nu_L = ccy_nu.get(p['lhs'], 0)
        nu_R = ccy_nu.get(p['rhs'], 0)
        pred_nu2 = wL**2 * nu_L**2 + wR**2 * nu_R**2
        pred_nu = np.sqrt(pred_nu2) if pred_nu2 > 0 else 0
        pred_bf = fair_bf(pred_nu, p['atm'], T)
        p['pred_nu'] = pred_nu
        p['pred_bf'] = pred_bf
        p['residual_bf'] = p['bf'] - pred_bf
        p['residual_nu'] = p['obs_nu'] - pred_nu

    return ccy_nu, bvols, pair_list


# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Test with realistic data
    pairs_1m = {
        'EURUSD': (7.53, 0.18),
        'USDJPY': (8.81, 0.24),
        'GBPUSD': (6.92, 0.14),
        'AUDUSD': (10.07, 0.28),
        'USDCHF': (8.12, 0.20),
        'USDCNH': (3.55, 0.08),
        'NZDUSD': (10.20, 0.30),
        'EURJPY': (10.80, 0.30),
        'EURGBP': (5.20, 0.10),
        'EURCNH': (6.38, 0.12),
        'GBPJPY': (11.50, 0.32),
        'GBPAUD': (7.54, 0.16),
        'AUDJPY': (12.20, 0.38),
        'AUDNZD': (6.80, 0.12),
        'CHFJPY': (8.43, 0.22),
        'NZDJPY': (12.80, 0.40),
    }

    ccy_nu, bvols, results = solve_currency_nu(pairs_1m, '1M')

    if ccy_nu:
        print('CURRENCY ν_log DECOMPOSITION (1M)')
        print('=' * 65)

        print(f'\n{"Ccy":>5s}  {"Basket Vol":>10s}  {"ν_log":>8s}  {"Daily σ move":>12s}')
        print(f'{"─"*5}  {"─"*10}  {"─"*8}  {"─"*12}')

        for ccy in sorted(ccy_nu.keys(), key=lambda c: -ccy_nu[c]):
            nu = ccy_nu[ccy]
            bv = bvols[ccy] * 100
            daily = daily_vol_stdev(bv, nu)
            print(f'{ccy:>5s}  {bv:9.2f}%  {nu:8.2f}  {daily:.2f}v/day')

        print(f'\n{"Pair":>8s}  {"ATM":>6s}  {"BF":>6s}  {"Obs ν":>7s}  {"Pred ν":>7s}  '
              f'{"Obs BF":>7s}  {"Pred BF":>8s}  {"Resid BF":>9s}')
        print(f'{"─"*8}  {"─"*6}  {"─"*6}  {"─"*7}  {"─"*7}  '
              f'{"─"*7}  {"─"*8}  {"─"*9}')

        for r in sorted(results, key=lambda x: abs(x['residual_bf']), reverse=True):
            print(f'{r["pair"]:>8s}  {r["atm"]:6.2f}  {r["bf"]:6.2f}  {r["obs_nu"]:7.2f}  '
                  f'{r["pred_nu"]:7.2f}  {r["bf"]:7.2f}  {r["pred_bf"]:8.2f}  '
                  f'{r["residual_bf"]:+9.4f}')

        residuals = [r['residual_bf'] for r in results]
        rmse = np.sqrt(np.mean(np.array(residuals)**2))
        print(f'\nRMSE of BF residuals: {rmse:.4f}%')
