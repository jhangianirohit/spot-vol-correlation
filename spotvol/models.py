from dataclasses import dataclass
from typing import Optional


@dataclass
class TenorInput:
    """Market data for a single tenor."""
    label: str          # "1W", "1M", etc.
    atm_vol: float      # decimal, e.g. 0.10 for 10%
    rr_25d: float       # decimal, e.g. -0.020 for -2.0% (negative = puts over calls)


@dataclass
class BetaResult:
    """Implied beta result for a single tenor."""
    label: str
    T: float            # time to expiry in years
    n_days: int         # business days
    atm_vol: float
    rr_25d: float
    K_call: float
    K_put: float
    vanna_call: float   # dVega for 1% spot bump
    vanna_put: float
    vanna_rr: float     # call - put
    vega_call: float
    vega_put: float
    avg_vega: float
    beta: float         # Δσ (decimal) per 1% spot move
    beta_sd: float      # Δσ (decimal) per 1 daily std dev of spot
    daily_std: float    # daily std dev in spot terms (S × σ × √(1/252))
    daily_std_pct: float  # daily std dev as % of spot

    def summary(self) -> str:
        direction = "vol UP when spot DOWN" if self.beta < 0 else "vol UP when spot UP"
        return (
            f"{self.label:>4s}  ATM={self.atm_vol*100:.1f}%  RR={self.rr_25d*100:+.2f}%  "
            f"β_sd={self.beta_sd:+.6f} ({abs(self.beta_sd)*100:.2f}v per 1σ spot)  [{direction}]"
        )
