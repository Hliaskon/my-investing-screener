
# Factor Explanations (v2)

**Buffett Score** — High-ROIC, FCF margin/yield, EBIT margin, low net debt, strong interest coverage.
**Lynch Score** — Low PEG/PEGY (cheaper vs growth), solid growth proxy, low leverage.
**Icahn Score** — Cash-rich, low P/FCF, shrinking shares, manageable leverage (activist unlock potential).
**Soros Score** — Cheap on FCF + high macro sensitivity (bigger reflexivity/mispricing optionality) with enough coverage.
**Simons Score** — Predictability via return autocorr, volatility clustering, and Sharpe-like return/stdev proxy.
**IP Boost** — Optional patents.csv adds a moat/innovation tilt to Buffett/Simons scores.

Notes:
- Growth proxy uses analyst 'earningsGrowth' when available, else 3-yr revenue CAGR fallback.
- ROIC is an operating proxy (NOPAT/avg invested capital) excluding cash, goodwill, intangibles.
- Use macro_overrides.json if you want to fix macro inputs for backtesting consistency.
