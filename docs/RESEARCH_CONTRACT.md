# HilmarCorp — Research Contract

## Study

**Bitcoin est-il devenu un actif macro ?**

Sample frozen at:

- start: 2017-08-17
- final Nasdaq session: 2026-09-04

## Primary estimand

The study measures the evolution of Bitcoin's contemporaneous statistical
integration with macro-financial factors.

It is not a forecasting study and it is not a causal identification design.

## Locked primary conclusions

Supported:

1. Macro-financial explanatory power is near zero in the pre-2020 sample and
   materially higher after 2020.
2. The dominant structural change is associated with the US equity factor.
3. Nasdaq-100 and S&P 500 specifications lead to the same broad conclusion.
4. The result survives removal of the 2020 Covid crash window.
5. The 2024 US spot-ETF period does not show a statistically established
   additional joint beta break versus 2020-2023.
6. Conditional event responses are directionally consistent with stronger
   contemporaneous risk-on/risk-off synchronization.

Not supported:

1. ETF launches caused the structural change.
2. Dollar, real rates and credit all experienced equally robust structural
   transformations.
3. Macro shocks robustly predict Bitcoin returns over the following
   1-20 sessions.
4. A precise unique causal break date is identified.

## Reproducibility constraints

- No forward information in rolling estimations.
- No silent interpolation of BTC market-close boundaries.
- FRED exact-date specification retained as robustness.
- Multiple-testing corrections preserved.
- Event windows must not overlap within the same research period.
- Frozen artifacts must match the SHA256 manifest.
- Refactors must preserve certified numerical outputs within explicit
  tolerances.
