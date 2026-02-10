# -*- coding: utf-8 -*-
"""
screen_v2.py — Upgraded multi-framework equity screener
Author: I. Koninis (Energy Efficiency), with Copilot assist
Date: 2026-02-10

What it does
------------
- Pulls prices & financials via yfinance
- Computes quality/cash metrics (ROIC, FCF Yield/Margin, EBIT margin)
- Adds Lynch GARP metrics (PEG, PEGY) with robust fallbacks
- Adds Icahn-style activist unlock heuristics
- Adds Soros macro-stress sensitivity proxies (rates, FX, commodities)
- Adds Simons quant-edge anomaly metrics (autocorr, vol clustering, return predictability)
- Optionally merges patents/IP signals from patents.csv
- Builds sub-scores per “legendary” investor and a combined meta-score
- Writes CSV + Markdown summaries

Notes
-----
- Uses public yfinance endpoints; some fields may be missing. Fallbacks keep it robust.
- For audited ROIC and patent data, integrate a fundamentals/IP provider or maintain your own csv.
- Educational research tool. Not investment advice.
"""

import os, json, warnings, datetime as dt
import numpy as np, pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------- Helper functions -------------------------

def safe_div(a, b):
    try:
        a = float(a); b = float(b)
        if b == 0 or np.isnan(b): return np.nan
        return a / b
    except Exception:
        return np.nan

def cagr(first, last, years):
    try:
        if first is None or last is None or first <= 0 or years <= 0:
            return np.nan
        return (last / first) ** (1/years) - 1
    except Exception:
        return np.nan

def pct_change(a, b):
    try:
        return (b - a) / abs(a) if a not in (0, None, np.nan) else np.nan
    except Exception:
        return np.nan

def series_autocorr(s, lag=1):
    s = pd.Series(s).dropna()
    if len(s) <= lag+1: return np.nan
    return s.autocorr(lag=lag)

def load_optional_csv(path, index_col="ticker"):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if index_col in df.columns:
                df[index_col] = df[index_col].astype(str).str.upper()
            return df
        except Exception:
            return None
    return None

def read_macro_overrides(path="macro_overrides.json"):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# ------------------------- Universe -------------------------

u = pd.read_csv("tickers.csv")
u["ticker"] = u["ticker"].astype(str).str.upper()

# Optional IP data
pat = load_optional_csv("patents.csv")

# Optional macro overrides
macro_ovr = read_macro_overrides()

# ------------------------- Macro proxies (Soros) -------------------------
# We fetch simple proxies via yfinance. If overrides exist, we use them.
def get_macro_proxy(ticker, fallback=None):
    try:
        px = yf.Ticker(ticker).history(period="1mo", interval="1d")["Close"]
        if px.empty: return fallback
        return float(px.iloc[-1])
    except Exception:
        return fallback

ten_year = macro_ovr.get("ten_year_yield", None)  # e.g., 0.042 (=4.2%)
if ten_year is None:
    # ^TNX prints 10-yr yield * 100; divide by 100 to get decimal
    try:
        tnx = get_macro_proxy("^TNX", None)
        ten_year = (tnx/100.0) if tnx is not None else np.nan
    except Exception:
        ten_year = np.nan

usd_dxy = macro_ovr.get("usd_dxy", get_macro_proxy("DX-Y.NYB", get_macro_proxy("^DXY", np.nan)))
wti = macro_ovr.get("wti", get_macro_proxy("CL=F", np.nan))
gold = macro_ovr.get("gold", get_macro_proxy("GC=F", np.nan))

# ------------------------- Collect metrics per ticker -------------------------

rows = []
today = dt.date.today().isoformat()

for _, row in u.iterrows():
    t = row["ticker"]
    region = row.get("region", "")
    notes = row.get("notes", "")

    tk = yf.Ticker(t)
    info = tk.info or {}

    # -------- Prices & basic valuation --------
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    mcap  = info.get("marketCap")
    pe    = info.get("trailingPE") or info.get("forwardPE")
    div_y = info.get("dividendYield")  # decimal, e.g., 0.015 for 1.5%

    # -------- Price history for Simons-style anomaly metrics --------
    hist = tk.history(period="3y", interval="1d")
    if hist is None or hist.empty:
        weekly_ac = np.nan; vol_clust = np.nan; ret_pred = np.nan
    else:
        wk = hist["Close"].resample("W-FRI").last().dropna()
        wk_ret = wk.pct_change().dropna()

        # Autocorrelation of weekly returns (lag-1)
        weekly_ac = series_autocorr(wk_ret, lag=1)

        # Volatility clustering: autocorr of squared returns
        vol_clust = series_autocorr((wk_ret**2), lag=1)

        # Simple predictability proxy: |mean return| / std dev (Sharpe-like without rf)
        ret_pred = abs(wk_ret.mean()) / (wk_ret.std()+1e-12)

    # -------- Financial statements --------
    cf = tk.cashflow
    is_ = tk.income_stmt
    bs_ = tk.balance_sheet

    # TTM FCF approximation (quarterly preferred)
    try:
        qcf = tk.quarterly_cashflow
        ocf = qcf.loc["Total Cash From Operating Activities"].iloc[:4].sum()
        capex = qcf.loc["Capital Expenditures"].iloc[:4].sum()
    except Exception:
        try:
            ocf = cf.loc["Total Cash From Operating Activities"].iloc[0]
            capex = cf.loc["Capital Expenditures"].iloc[0]
        except Exception:
            ocf, capex = np.nan, np.nan
    fcf = (ocf or 0) - (capex or 0)

    # Revenue, EBIT, interest expense for coverage
    try:
        rev = is_.loc["Total Revenue"].iloc[0]
    except Exception:
        rev = np.nan
    try:
        ebit = is_.loc["Ebit"].iloc[0]
    except Exception:
        ebit = np.nan
    try:
        interest_exp = is_.loc["Interest Expense"].iloc[0]
    except Exception:
        interest_exp = np.nan

    # Balance sheet components for invested capital (use average of two periods if possible)
    try:
        ta_now = bs_.loc["Total Assets"].iloc[0]
        cl_now = bs_.loc["Total Current Liabilities"].iloc[0]
        cash_now = bs_.loc.get("Cash And Cash Equivalents", pd.Series([np.nan])).iloc[0]
        sti_now = bs_.loc.get("Short Term Investments", pd.Series([0])).iloc[0]
        gw_now  = bs_.loc.get("Goodwill", pd.Series([0])).iloc[0]
        intang_now = bs_.loc.get("Intangible Assets", pd.Series([0])).iloc[0]
    except Exception:
        ta_now=cl_now=cash_now=sti_now=gw_now=intang_now=np.nan

    try:
        ta_prev = bs_.loc["Total Assets"].iloc[1]
        cl_prev = bs_.loc["Total Current Liabilities"].iloc[1]
        cash_prev = bs_.loc.get("Cash And Cash Equivalents", pd.Series([np.nan, np.nan])).iloc[1]
        sti_prev = bs_.loc.get("Short Term Investments", pd.Series([0,0])).iloc[1]
        gw_prev  = bs_.loc.get("Goodwill", pd.Series([0,0])).iloc[1]
        intang_prev = bs_.loc.get("Intangible Assets", pd.Series([0,0])).iloc[1]
    except Exception:
        ta_prev=cl_prev=cash_prev=sti_prev=gw_prev=intang_prev=np.nan

    def invested_cap(ta, cl, cash, sti, gw, inta):
        if all(pd.isna([ta, cl])): return np.nan
        # Conservative operating IC: TA - CL - (Cash+STI) - (Goodwill+Intangibles)
        cashlike = (cash or 0) + (sti or 0)
        soft = (gw or 0) + (inta or 0)
        return (ta - cl) - cashlike - soft

    ic_now  = invested_cap(ta_now, cl_now, cash_now, sti_now, gw_now, intang_now)
    ic_prev = invested_cap(ta_prev, cl_prev, cash_prev, sti_prev, gw_prev, intang_prev)
    ic_avg  = np.nanmean([ic_now, ic_prev])

    # ROIC with NOPAT = EBIT * (1 - tax rate). Use fallback flat tax 21% if not available.
    tax_rate = 0.21
    nopat = ebit * (1 - tax_rate) if pd.notna(ebit) else np.nan
    roic = safe_div(nopat, ic_avg)

    # Ratios
    fcf_yield   = safe_div(fcf, mcap)
    fcf_margin  = safe_div(fcf, rev)
    ebit_margin = safe_div(ebit, rev)
    net_debt = None
    try:
        tot_debt = bs_.loc.get("Total Debt", pd.Series([np.nan])).iloc[0]
        net_debt = (tot_debt or 0) - (cash_now or 0)
    except Exception:
        net_debt = np.nan

    # Interest coverage = EBIT / |Interest expense|
    int_cov = safe_div(ebit, abs(interest_exp) if pd.notna(interest_exp) else np.nan)

    # ---------------- Lynch PEG/PEGY with fallbacks ----------------
    # Try analyst growth (info['earningsGrowth']); else compute 3y revenue CAGR
    growth = info.get("earningsGrowth")

    try:
        # Build 4-year revenue series (annual)
        if is_ is not None and not is_.empty:
            rev_hist = is_.loc["Total Revenue"].iloc[:4].dropna()
            if len(rev_hist) >= 2:
                years = len(rev_hist) - 1
                # conservative: use earliest vs latest
                growth_rev = cagr(rev_hist.iloc[-1*years], rev_hist.iloc[0], years)
            else:
                growth_rev = np.nan
        else:
            growth_rev = np.nan
    except Exception:
        growth_rev = np.nan

    if growth is None or pd.isna(growth):
        growth = growth_rev

    # PEG and PEGY (use trailing P/E; if missing, compute P/E with EPS approximate)
    pe_eff = pe
    if (pe_eff is None or pd.isna(pe_eff)) and info.get("trailingEps"):
        eps = info.get("trailingEps")
        pe_eff = safe_div(price, eps)

    # Ensure growth is decimal; avoid non-positive
    g = growth if (growth is not None and not pd.isna(growth)) else np.nan
    if g is not np.nan and g is not None and g <= 0:
        g = np.nan

    PEG = np.nan
    if pe_eff and g:
        # Classic Lynch: P/E divided by growth in %
        if g < 1:
            PEG = pe_eff / (g*100.0)
        else:
            PEG = pe_eff / g

    div_pct = (div_y or 0)*100.0 if div_y and div_y < 1 else div_y  # convert to %
    PEGY = None
    if pd.notna(pe_eff) and pd.notna(g):
        denom = (g*100.0 if g and g < 1 else g)
        denom += (div_pct or 0)
        PEGY = safe_div(pe_eff, denom)

    # ---------------- Icahn-style activist unlock proxies ----------------
    try:
        shares = tk.get_shares_full(start=dt.date.today().replace(year=dt.date.today().year-3))
        shares = shares.dropna()
        shr_change = pct_change(shares.iloc[0], shares.iloc[-1]) if len(shares) >= 2 else np.nan
    except Exception:
        shr_change = np.nan

    cash_to_mcap = safe_div((bs_.loc.get("Cash And Cash Equivalents", pd.Series([np.nan])).iloc[0] if bs_ is not None else np.nan), mcap)
    p_to_fcf = safe_div(mcap, fcf) if pd.notna(fcf) and fcf > 0 else np.nan

    # ---------------- Soros macro-stress proxies ----------------
    sector = (info.get("sector") or "").lower()
    cyclical = any(k in sector for k in ["financial", "materials", "industr", "energy", "discretionary"])
    rate_sensitive = ("financial" in sector) or (net_debt and net_debt > 0)
    fx_sensitive = ("materials" in sector or "industr" in sector or "technology" in sector)

    macro_sensitivity = sum([
        1.0 if cyclical else 0.0,
        1.0 if rate_sensitive else 0.0,
        1.0 if fx_sensitive else 0.0
    ])

    # ---------------- Simons quant-edge proxies already computed ----------------

    # ---------------- Optional IP/Patents merge ----------------
    patent_count = forward_cit = rd_to_sales = np.nan
    if pat is not None:
        rec = pat[pat["ticker"].str.upper()==t]
        if not rec.empty:
            patent_count = rec["patent_count"].iloc[0] if "patent_count" in rec.columns else np.nan
            forward_cit  = rec["forward_citations"].iloc[0] if "forward_citations" in rec.columns else np.nan
            rd_to_sales  = rec["rd_to_sales"].iloc[0] if "rd_to_sales" in rec.columns else np.nan

    rows.append({
        "date": today, "ticker": t, "region": region, "notes": notes,
        "price": price, "market_cap": mcap, "pe": pe_eff, "div_yield": div_y,
        "revenue_ttm": rev, "ebit_ttm": ebit, "fcf_ttm": fcf,
        "ebit_margin": ebit_margin, "fcf_margin": fcf_margin, "fcf_yield": fcf_yield,
        "roic_est": roic, "interest_coverage": int_cov, "net_debt": net_debt,
        "PEG": PEG, "PEGY": PEGY, "growth_proxy": g,
        "cash_to_mcap": cash_to_mcap, "p_to_fcf": p_to_fcf, "shares_chg_3y": shr_change,
        "weekly_ac": weekly_ac, "vol_clust": vol_clust, "ret_pred": ret_pred,
        "macro_sensitivity": macro_sensitivity,
        "patent_count": patent_count, "forward_citations": forward_cit, "rd_to_sales": rd_to_sales,
        "sector": info.get("sector")
    })

df = pd.DataFrame(rows)

# ------------------------- Build sub-scores per investor -------------------------

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - np.nanmean(s)) / (np.nanstd(s) + 1e-9)

def pos_clip(s):  # map z to 0..100
    return 100*(s - np.nanmin(s))/(np.nanmax(s)-np.nanmin(s) + 1e-9)

# Buffett Quality: ROIC, FCF margin, FCF yield, EBIT margin, low leverage, coverage
df["buffett_raw"] = (
    0.35*zscore(df["roic_est"]) +
    0.20*zscore(df["fcf_margin"]) +
    0.15*zscore(df["fcf_yield"]) +
    0.15*zscore(df["ebit_margin"]) +
    0.10*(-zscore(df["net_debt"])) +   # lower net debt better
    0.05*zscore(df["interest_coverage"])
)
df["buffett_score"] = pos_clip(df["buffett_raw"])

# Lynch GARP: inverse PEG/PEGY (lower better), growth, low debt
inv_PEG = 1/(df["PEG"]+1e-9)
inv_PEGY = 1/(df["PEGY"]+1e-9)
df["lynch_raw"] = (
    0.35*zscore(inv_PEG) +
    0.25*zscore(inv_PEGY) +
    0.25*zscore(df["growth_proxy"]) +
    0.15*(-zscore(df["net_debt"]))
)
df["lynch_score"] = pos_clip(df["lynch_raw"])

# Icahn Activist Unlock: cash_to_mcap high, p_to_fcf low, shares_chg negative (buybacks), leverage not extreme
inv_p_to_fcf = 1/(df["p_to_fcf"]+1e-9)
df["icahn_raw"] = (
    0.35*zscore(df["cash_to_mcap"]) +
    0.35*zscore(inv_p_to_fcf) +
    0.20*(-zscore(df["shares_chg_3y"])) +  # more negative (shrinking shares) is better
    0.10*(-zscore(df["net_debt"]))
)
df["icahn_score"] = pos_clip(df["icahn_raw"])

# Soros Macro-Stress: cheap on FCF + high macro sensitivity (+ coverage)
df["soros_raw"] = (
    0.50*zscore(inv_p_to_fcf) +           # cheap on FCF = upside if macro flips
    0.30*zscore(df["macro_sensitivity"]) +# higher sensitivity => bigger reflexive moves
    0.20*zscore(df["interest_coverage"])  # ability to survive rates
)
df["soros_score"] = pos_clip(df["soros_raw"])

# Simons Quant-Edge: autocorr, vol clustering, return predictability
df["simons_raw"] = (
    0.34*zscore(df["weekly_ac"]) +
    0.33*zscore(df["vol_clust"]) +
    0.33*zscore(df["ret_pred"])
)
df["simons_score"] = pos_clip(df["simons_raw"])

# Optional IP boost (applies to Buffett & Simons)
if any(col in df.columns for col in ["patent_count","forward_citations","rd_to_sales"]):
    ip_z = (
        0.50*zscore(df["patent_count"]) +
        0.30*zscore(df["forward_citations"]) +
        0.20*zscore(df["rd_to_sales"])
    )
    df["buffett_score"] = np.where(np.isnan(ip_z), df["buffett_score"], df["buffett_score"]*0.9 + pos_clip(ip_z)*0.1)
    df["simons_score"]  = np.where(np.isnan(ip_z), df["simons_score"],  df["simons_score"]*0.9  + pos_clip(ip_z)*0.1)

# Combined meta-score — equal-ish weights by default (tune to taste)
df["meta_score"] = (
    0.25*df["buffett_score"] +
    0.20*df["lynch_score"] +
    0.20*df["icahn_score"] +
    0.15*df["soros_score"] +
    0.20*df["simons_score"]
)

# Quality & Cash flags (from v1, retained)
df["quality_flag"] = (df["roic_est"]>=0.15) & (df["ebit_margin"]>=0.15)
df["cash_flag"] = (df["fcf_margin"]>=0.10) | (df["fcf_yield"]>=0.05)

# ------------------------- Outputs -------------------------

out_cols = [
    "date","ticker","region","sector","notes","price","market_cap",
    "roic_est","ebit_margin","fcf_margin","fcf_yield","interest_coverage","net_debt",
    "PEG","PEGY","growth_proxy",
    "cash_to_mcap","p_to_fcf","shares_chg_3y",
    "weekly_ac","vol_clust","ret_pred",
    "patent_count","forward_citations","rd_to_sales",
    "buffett_score","lynch_score","icahn_score","soros_score","simons_score","meta_score",
    "quality_flag","cash_flag"
]
df[out_cols].sort_values("meta_score", ascending=False).to_csv(
    os.path.join(OUT_DIR, "screen_results_v2.csv"), index=False
)

# Markdown summary
top = df.sort_values("meta_score", ascending=False).head(20)
md = top[[
    "ticker","region","sector","meta_score",
    "buffett_score","lynch_score","icahn_score","soros_score","simons_score",
    "roic_est","fcf_yield","PEGY","cash_to_mcap","p_to_fcf"
]].to_markdown(index=False)

with open(os.path.join(OUT_DIR, "screen_report_v2.md"), "w", encoding="utf-8") as f:
    f.write(f"# Weekly Screen — v2 ({today})\n\n")
    f.write("Top 20 by **Meta‑Score** (combined Buffett/Lynch/Icahn/Soros/Simons):\n\n")
    f.write(md+"\n")

# Factor explain
explain = """
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
"""
with open(os.path.join(OUT_DIR, "factor_explain_v2.md"), "w", encoding="utf-8") as f:
    f.write(explain)

print("Done. Wrote outputs/screen_results_v2.csv, screen_report_v2.md, factor_explain_v2.md")

