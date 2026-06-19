# Intern Project Brief — The Global Vol Factor

**Working title:** *The Global Vol Factor — is it real, is it stable, is it tradeable?*

**Duration:** 6 weeks · **Background assumed:** strong quantitative/maths, Python literate · **Data:** full historical daily ATM / BF / RR across the FX pair universe is available.

---

## 1. One-paragraph summary

The desk has built an analytical framework that decomposes FX implied volatility-of-volatility, skew and implied correlation into **one global vol factor plus currency-specific idiosyncratic components**, and uses it to flag which butterflies, risk reversals and cross-vols are rich or cheap *today*. Every output so far is a **point-in-time snapshot** — the model asserts fair value at one instant and has never been validated through time. This project takes the existing model and answers the three questions that decide whether anyone should trust it: **does the global factor actually exist in the data, is it stable, and do its rich/cheap signals make money.**

## 2. Why this project

- **For senior management:** it produces a single headline — a *Global FX Vol Factor index* — that summarises global volatility risk appetite in one line, and a credibility statement ("the desk's model is empirically validated").
- **For the desk:** it upgrades the existing live signals from "interesting model" to "signal with a backtested track record," and tells traders *how much to trust the model in the current regime.*
- **For the intern:** it is real estimation and inference (factor models, rolling identification, mean-reversion testing) layered on top of already-validated code, so even partial results are presentable and there is no risk of "invent a model from scratch."

## 3. What already exists (starting point — do not rebuild)

| Component | File | What it gives you |
|---|---|---|
| Global vol-of-vol factor | `nu_basket.py` | ν² = ν_g² + w²η²: one global factor + per-currency idiosyncratic η |
| Currency betas (spot-vol leverage) | `basket_beta.py` | per-currency β from the RR cross-section |
| β² → butterfly link | `bf_beta_convexity.py` | MC-validated: skew/leverage feeds the fly |
| Vol of implied correlation | `vol_of_correlation.py` | model vs market cross-fly-vs-legs signal |
| BF → vol-of-vol inversion | `bf_analysis.py` | implied ν_log from a butterfly |
| Live dashboard | `app.py`, `index.html` | surfaces model − observed residuals ("+ = buy") |
| Theory write-ups | `nu-explainer.html`, `explainer.html` | full derivations, incl. the "one factor isn't enough / blocs" note (§11) |

**The intern reads, does not rewrite, this code.** The project consumes these as a black box that maps a daily vol snapshot → factor estimates and residual signals.

## 4. The three workstreams

Each answers one question at one altitude and ships one deliverable.

### Workstream 1 — Is it real? *(the management story)*
Run PCA / factor analysis on the historical panel of FX vol-of-vol (and ATM / BF / RR).
- Show empirically that **one dominant factor** explains the bulk of common variation in vol-of-vol across pairs.
- Confirm that this statistical factor *is* the model's ν_g (correlate the PC1 time series against the fitted ν_g series — they should track).
- Tie the factor level to risk regimes (VIX, DXY, risk-on/off episodes).
- **Deliverable:** the *Global FX Vol Factor index* — one time series, one chart, the headline slide.

### Workstream 2 — Is it stable? *(the maths)*
Rolling / recursive estimation of ν_g and the per-currency η and β.
- Confidence intervals and standard errors on the estimates.
- Regime dependence: does the factor structure hold in calm vs stressed markets?
- Does the bloc structure (nu-explainer §11, "one factor isn't enough") surface as a clean, persistent **second factor**?
- **Deliverable:** a stability report — how much, and when, to trust the signal. This is the trader-facing risk-control output.

### Workstream 3 — Is it tradeable? *(the desk capstone)*
Backtest the model − observed residual signals over the available history.
- Do rich flies cheapen? Does the vol-of-correlation signal mean-revert toward the model?
- Transaction-cost-aware PnL, hit-rate, Sharpe; signal-decay / holding-period analysis.
- **Deliverable:** a backtested track record attached to the existing live signals.

*Scope note for 6 weeks:* Workstreams 1 and 2 are the core and must land. Workstream 3 is the capstone — because full history is available it is in scope, but kept deliberately lightweight (one or two signals, simple cost model). If time is short, a clean mean-reversion test of the BF residual alone is an acceptable minimum for W3.

## 5. Six-week milestone plan

| Week | Focus | Checkpoint |
|---|---|---|
| 1 | Build the historical vol panel; reproduce one day of model output vs the live app to confirm the pipeline | Panel assembled, model reproduced |
| 2 | W1: PCA on the panel, variance explained, PC1 extraction | "One factor" chart, %-variance table |
| 3 | W1: identify PC1 with ν_g, overlay VIX/DXY → the Global Vol Factor index | Headline index + first management chart |
| 4 | W2: rolling estimation of ν_g / η / β, standard errors | Stability time series with error bands |
| 5 | W2 → W3: regime/second-factor check; set up the residual backtest | Second-factor verdict; backtest harness running |
| 6 | W3: run backtest (BF residual + vol-of-corr), PnL/Sharpe; assemble deck | Track record + final deck + desk write-up |

## 6. Final deliverables

1. **Management deck** — the factor narrative and the Global Vol Factor index (workstream 1), plus a one-line credibility statement from workstreams 2–3.
2. **Desk output** — the stability report (W2) and the backtested track record (W3), feeding the existing dashboard.
3. **Reproducible code** — a small analysis package sitting on top of the existing model files, runnable end-to-end on the historical panel.

## 7. Risks / things to watch

- **Identification drift:** ν_g is identified under specific conditions (nu-explainer §9). Rolling estimation can break those — the intern must monitor, not assume, identification each window.
- **Look-ahead in the backtest:** signals must use only data available at decision time; basket vols and the fit are themselves estimated from the cross-section.
- **Cost realism:** vol products are not frictionless; an over-optimistic cost model invalidates W3. Keep it conservative.
- **Scope creep:** the temptation is to improve the model. The job is to *validate* it. Model changes are out of scope for 6 weeks.

---

## Appendix — How to pitch this to senior management

**The hook (one sentence):** *"There is a single global volatility factor moving FX vol markets, we can measure it daily, and we can tell you when individual products are mispriced relative to it."*

**The arc of the pitch (≤ 4 slides):**

1. **The big idea — one factor.** Lead with the variance-explained result and the Global Vol Factor index chart. The message is altitude-appropriate: *most* of what looks like dozens of independent vol markets is one common driver. No volga, no SABR — just "one number summarises global FX vol risk appetite." This is the slide that earns the room's attention.
2. **It's not a curve-fit — it's stable and it ties to risk regimes.** Overlay the factor on VIX / risk-off episodes. Show it strengthens in stress. This is where management's intuition ("vol is a risk-appetite thing") gets confirmed by the desk's own model — credibility.
3. **It pays the desk.** The backtested track record: rich products cheapen, the signal mean-reverts, here's the hit-rate. This converts an *interesting* story into a *commercial* one — the reason to fund the desk's research.
4. **What it gives us going forward.** A live risk-appetite gauge and a rich/cheap monitor the desk already uses. Frame the intern's work as *validation of an existing edge*, not a speculative new bet — lower-risk framing lands better with senior audiences.

**Tone:** every quantitative claim should reduce to one plain-English sentence. The maths (factor identification, rolling SEs, mean-reversion tests) lives in the appendix and the desk report, not the headline deck. Senior management buys the *story and the track record*; the desk buys the *stability report and the signal.* The project is deliberately built to serve both from the same analysis.
