"""
Vol of implied correlation — model-implied and market-implied.

A cross C = XY (both legs quoted vs USD) has its correlation pinned by
three vols through the variance triangle:

    σ_C² = σ_X² + σ_Y² − 2ρ·σ_X·σ_Y
    ⇒ ρ = (σ_X² + σ_Y² − σ_C²) / (2·σ_X·σ_Y)

So ρ moves only when those three vols move. Linearize:

    dρ = g_X·dσ_X + g_Y·dσ_Y + g_C·dσ_C

with analytic partials (elementary from the triangle):

    g_X = ∂ρ/∂σ_X = (σ_X² − σ_Y² + σ_C²) / (2·σ_X²·σ_Y)
    g_Y = ∂ρ/∂σ_Y = (σ_Y² − σ_X² + σ_C²) / (2·σ_Y²·σ_X)
    g_C = ∂ρ/∂σ_C = −σ_C / (σ_X·σ_Y)

Each vol moves lognormally, dσ_i = σ_i·ν_i·dW_i, and the three vols
co-move through the currency factor model. So:

    Var(dρ)/dt = Σ_ij g_i·g_j·σ_i·σ_j·ν_i·ν_j·ρ_vol(i,j)

  vol-of-correlation = √(Var(dρ)/dt)   (annualized, in correlation units)

Two flavours, differing only in which ν's feed the diagonal:
  • MODEL-implied : ν_i = the factor model's pair vol-of-vol
                    (ν_i² = ν_g² + w² η²). What the fit says ρ *should* do.
  • MARKET-implied: ν_i = observed pair vol-of-vol (from each pair's own
                    BF). What the traded cross-fly-vs-leg-flies prices.
The cross-pair correlations ρ_vol(i,j) can't be observed directly, so
both flavours use the factor model for the OFF-diagonal:

    Cov(d ln σ_i, d ln σ_j) = ν_g² + Σ_{s shared} w_{i,s}·w_{j,s}·η_s²
    ρ_vol(i,j) = Cov / (ν_i^model · ν_j^model)

market − model is the tradeable signal: cross fly rich vs its legs ⇒
market pricing more vol-of-correlation than the factor model implies.

Caveat: the off-diagonal ρ_vol is model-based, and the linearization is
first-order in vol moves. Treat the level as indicative; the
market−model *gap* is the robust read.
"""

import numpy as np
from bf_analysis import implied_nu_log, tenor_to_T
from nu_basket import solve_currency_nu_global, compute_basket_vols, pair_weights


def _find_pair(pairs_data, a, b):
    """Return the key matching currencies a,b in either order, else None."""
    for p in pairs_data:
        if {p[:3], p[3:]} == {a, b}:
            return p
    return None


def vol_of_correlation(pairs_data, tenor, g=None):
    """
    Args:
        pairs_data: {pair: (atm_pct, bf_pct)} single tenor
        tenor: e.g. '1M'
        g: optional precomputed solve_currency_nu_global result

    Returns list of dicts (one per cross), each with rho, vovc_model,
    vovc_market (annualized correlation-vol), daily versions, and signal.
    """
    T = tenor_to_T(tenor)
    if not T:
        return []
    if g is None:
        g = solve_currency_nu_global(pairs_data, tenor)
    if g is None:
        return []

    bvols = g['basket_vols']
    nu_g = g['nu_g']
    eta = g['ccy_eta']

    # Per-pair observed ν and model ν, plus currency weights
    def pair_info(pair):
        atm, bf = pairs_data[pair]
        lhs, rhs = pair[:3], pair[3:]
        wL, wR = pair_weights(pair, bvols)
        obs_nu = implied_nu_log(bf, atm, T) if (bf and bf > 0) else None
        eL, eR = eta.get(lhs, 0.0), eta.get(rhs, 0.0)
        model_nu = np.sqrt(nu_g**2 + wL**2 * eL**2 + wR**2 * eR**2)
        weights = {lhs: wL, rhs: wR}
        return {'atm': atm, 'lhs': lhs, 'rhs': rhs, 'w': weights,
                'obs_nu': obs_nu, 'model_nu': model_nu}

    def cov_logvol(pi, pj):
        """Model covariance of the two pairs' log-vol shocks."""
        cov = nu_g**2  # shared global factor
        shared = set(pi['w']) & set(pj['w'])
        for s in shared:
            cov += pi['w'][s] * pj['w'][s] * eta.get(s, 0.0)**2
        return cov

    results = []
    for pair, (atm_c, bf_c) in pairs_data.items():
        x_ccy, y_ccy = pair[:3], pair[3:]
        if 'USD' in (x_ccy, y_ccy):
            continue  # only crosses
        legX = _find_pair(pairs_data, x_ccy, 'USD')
        legY = _find_pair(pairs_data, y_ccy, 'USD')
        if not legX or not legY:
            continue

        sX = pairs_data[legX][0]
        sY = pairs_data[legY][0]
        sC = atm_c
        denom = 2 * sX * sY
        rho = (sX**2 + sY**2 - sC**2) / denom
        rho = max(-0.999, min(0.999, rho))

        # analytic partials ∂ρ/∂σ_i
        gX = (sX**2 - sY**2 + sC**2) / (2 * sX**2 * sY)
        gY = (sY**2 - sX**2 + sC**2) / (2 * sY**2 * sX)
        gC = -sC / (sX * sY)

        pi = {legX: pair_info(legX), legY: pair_info(legY), pair: pair_info(pair)}
        order = [legX, legY, pair]
        sig = {legX: sX, legY: sY, pair: sC}
        grad = {legX: gX, legY: gY, pair: gC}

        # ρ_vol(i,j) from the model (off-diagonal), 1 on the diagonal
        def rho_vol(a, b):
            if a == b:
                return 1.0
            return cov_logvol(pi[a], pi[b]) / (pi[a]['model_nu'] * pi[b]['model_nu'])

        def var_drho(use_obs):
            tot = 0.0
            for a in order:
                na = pi[a]['obs_nu'] if use_obs else pi[a]['model_nu']
                if na is None:
                    return None
                for b in order:
                    nb = pi[b]['obs_nu'] if use_obs else pi[b]['model_nu']
                    if nb is None:
                        return None
                    tot += (grad[a] * grad[b] * sig[a] * sig[b]
                            * na * nb * rho_vol(a, b))
            return tot

        v_model = var_drho(False)
        v_market = var_drho(True)
        if v_model is None or v_market is None:
            continue

        vovc_model = np.sqrt(v_model) if v_model > 0 else 0.0
        # market variance can go negative (cross fly cheap vs legs) → flag
        vovc_market = np.sqrt(v_market) if v_market > 0 else -np.sqrt(-v_market)

        results.append({
            'cross': pair, 'legX': legX, 'legY': legY,
            'rho': rho,
            'vovc_model': vovc_model,            # annualized, corr units
            'vovc_market': vovc_market,
            'daily_model': vovc_model / np.sqrt(252),
            'daily_market': (abs(vovc_market) / np.sqrt(252)
                             * (1 if vovc_market >= 0 else -1)),
            'signal': vovc_market - vovc_model,  # market rich (+) / cheap (−)
        })

    return results


if __name__ == '__main__':
    pairs_1m = {
        'EURUSD': (7.53, 0.18), 'USDJPY': (8.81, 0.24), 'GBPUSD': (6.92, 0.14),
        'AUDUSD': (10.07, 0.28), 'USDCHF': (8.12, 0.20), 'USDCNH': (3.55, 0.08),
        'NZDUSD': (10.20, 0.30), 'EURJPY': (10.80, 0.30), 'EURGBP': (5.20, 0.10),
        'EURCNH': (6.38, 0.12), 'GBPJPY': (11.50, 0.32), 'GBPAUD': (7.54, 0.16),
        'AUDJPY': (12.20, 0.38), 'AUDNZD': (6.80, 0.12), 'CHFJPY': (8.43, 0.22),
        'NZDJPY': (12.80, 0.40),
    }

    res = vol_of_correlation(pairs_1m, '1M')
    print('VOL OF IMPLIED CORRELATION (1M)')
    print('=' * 78)
    print(f'\n{"Cross":>8s}  {"Legs":>16s}  {"ρ_impl":>7s}  '
          f'{"VoVC model":>10s}  {"VoVC mkt":>9s}  {"daily mkt":>9s}  {"signal":>8s}')
    print(f'{"─"*8}  {"─"*16}  {"─"*7}  {"─"*10}  {"─"*9}  {"─"*9}  {"─"*8}')
    for r in sorted(res, key=lambda x: -abs(x['signal'])):
        legs = f'{r["legX"]}/{r["legY"]}'
        print(f'{r["cross"]:>8s}  {legs:>16s}  {r["rho"]:7.3f}  '
              f'{r["vovc_model"]:10.3f}  {r["vovc_market"]:9.3f}  '
              f'{r["daily_market"]:8.4f}  {r["signal"]:+8.3f}')

    print(f'\n  VoVC = annualized stdev of implied correlation (corr units).')
    print(f'  daily ≈ VoVC / 16.  signal = market − model (rich +, cheap −).')
