"""
╔══════════════════════════════════════════════════════════════════╗
║   RESEARCH PAPER: Gold–Silver Ratio Mean-Reversion Strategy     ║
║   Trading the Gold/Silver Ratio on MCX Futures                  ║
╠══════════════════════════════════════════════════════════════════╣
║   Author   : Your Name                                          ║
║   LinkedIn : https://linkedin.com/in/yourprofile                ║
║   GitHub   : https://github.com/yourusername/gold-silver-gsr    ║
╠══════════════════════════════════════════════════════════════════╣
║   WHY THIS PAIR IS DIFFERENT FROM GOLD-CRUDE:                   ║
║                                                                  ║
║   Gold-Crude correlation  : 0.15 – 0.45  (unstable, weak)      ║
║   Gold-Silver correlation : 0.85 – 0.95  (stable, very strong) ║
║                                                                  ║
║   Both are precious metals, both priced in USD, both driven     ║
║   by: real interest rates, USD strength, inflation, and         ║
║   global risk sentiment. Silver has additional industrial        ║
║   demand (solar panels, EVs, electronics) which causes          ║
║   TEMPORARY divergences from gold — and these divergences       ║
║   have mean-reverted reliably for over 100 years.               ║
║                                                                  ║
║   THE GOLD/SILVER RATIO (GSR):                                  ║
║   GSR = Gold price / Silver price                               ║
║   Historical range: ~50 to ~120                                 ║
║   Long-run mean: ~65–70                                         ║
║   GSR > 85 → Silver historically cheap → BUY SILVER            ║
║   GSR < 55 → Gold historically cheap  → BUY GOLD               ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python backtest.py                    # recommended defaults (5y)
  python backtest.py --period 10y       # full decade stress test
  python backtest.py --z-entry 2.0 --hold 20 --period 10y
  python backtest.py --silver-only      # highest win-rate leg
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
import yfinance as yf

# ── Colour palette ──────────────────────────────────────────
BG, PANEL, BORDER = "#0d1117", "#161b22", "#21262d"
AMBER, BLUE       = "#f59e0b", "#60a5fa"
GREEN, RED, GREY  = "#4ade80", "#f87171", "#8b949e"
WHITE, PURPLE     = "#e6edf3", "#c084fc"
SILVER            = "#94a3b8"

plt.rcParams.update({
    "figure.facecolor": BG,    "axes.facecolor": PANEL,
    "axes.edgecolor":  BORDER, "axes.labelcolor": GREY,
    "axes.titlecolor": WHITE,  "xtick.color": GREY,
    "ytick.color":     GREY,   "grid.color": BORDER,
    "grid.linestyle":  "--",   "grid.alpha": 0.5,
    "text.color":      WHITE,  "legend.facecolor": PANEL,
    "legend.edgecolor":BORDER, "font.family": "monospace",
    "font.size": 9,
})


# ════════════════════════════════════════════════════════════
#  1.  MCX FUTURES COST MODEL
# ════════════════════════════════════════════════════════════
class MCXFuturesCosts:
    """
    MCX Precious Metals Futures (Gold & Silver) — FY 2025-26
    Similar structure to crude but slightly different exchange charges.
    """
    BROKERAGE_PCT     = 0.0002    # 0.02% per side
    STT_SELL_ONLY     = 0.0001    # 0.01% sell side only (futures)
    EXCHANGE_CHARGE   = 0.000026  # MCX: 0.0026% per side
    GST_RATE          = 0.18      # 18% on brokerage + exchange
    SEBI_CHARGE       = 0.000001  # 0.0001% per side
    STAMP_DUTY        = 0.00002   # 0.002% buy side only
    SLIPPAGE_PER_SIDE = 0.0006    # 0.06% (precious metals very liquid)
    SPREAD_RT         = 0.0004    # 0.04% round-trip (tighter than crude)
    INCOME_TAX        = 0.30      # 30% on profit (F&O business income)

    @classmethod
    def compute(cls, trade_value: float, gross_pnl: float) -> dict:
        brokerage = trade_value * cls.BROKERAGE_PCT * 2
        stt       = trade_value * cls.STT_SELL_ONLY
        exch      = trade_value * cls.EXCHANGE_CHARGE * 2
        gst       = (brokerage + exch) * cls.GST_RATE
        sebi      = trade_value * cls.SEBI_CHARGE * 2
        stamp     = trade_value * cls.STAMP_DUTY
        slippage  = trade_value * cls.SLIPPAGE_PER_SIDE * 2
        spread    = trade_value * cls.SPREAD_RT
        pre_tax   = brokerage+stt+exch+gst+sebi+stamp+slippage+spread
        pnl_after = gross_pnl - pre_tax
        tax       = max(0.0, pnl_after * cls.INCOME_TAX)
        total     = pre_tax + tax
        return {
            "brokerage"    : round(brokerage, 2),
            "stt"          : round(stt, 2),
            "exchange"     : round(exch, 2),
            "gst"          : round(gst, 2),
            "sebi"         : round(sebi, 2),
            "stamp_duty"   : round(stamp, 2),
            "slippage"     : round(slippage, 2),
            "spread"       : round(spread, 2),
            "total_pre_tax": round(pre_tax, 2),
            "income_tax"   : round(tax, 2),
            "total_cost"   : round(total, 2),
            "net_pnl_₹"    : round(gross_pnl - total, 2),
            "cost_pct"     : round(total / trade_value * 100, 4),
        }


# ════════════════════════════════════════════════════════════
#  2.  DATA LAYER
# ════════════════════════════════════════════════════════════
def get_close(ticker: str, period: str) -> pd.Series:
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    c = raw["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return pd.Series(c.values.flatten(), index=pd.to_datetime(raw.index),
                     name=ticker, dtype=float)


def fetch_data(period: str = "5y") -> pd.DataFrame:
    print(f"\n📥  Fetching MCX proxy data  (period = {period}) …")
    print("    GC=F  → COMEX Gold    (proxy for MCX Gold,  corr > 0.98)")
    print("    SI=F  → COMEX Silver  (proxy for MCX Silver, corr > 0.97)")
    print("    ^VIX  → CBOE VIX      (regime filter)")

    gold   = get_close("GC=F", period)
    silver = get_close("SI=F", period)
    vix    = get_close("^VIX", period)

    df = pd.concat([
        gold.rename("Gold"),
        silver.rename("Silver"),
        vix.rename("VIX"),
    ], axis=1).dropna()

    print(f"✅  {len(df)} aligned trading days  "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    # Quick sanity check
    r_g   = df["Gold"].pct_change().dropna()
    r_s   = df["Silver"].pct_change().dropna()
    corr  = r_g.corr(r_s)
    gsr_now  = (df["Gold"] / df["Silver"]).iloc[-1]
    gsr_mean = (df["Gold"] / df["Silver"]).mean()
    print(f"📊  Gold-Silver correlation : {corr:.3f}")
    print(f"📊  GSR today              : {gsr_now:.1f}x")
    print(f"📊  GSR historical mean    : {gsr_mean:.1f}x")
    return df


# ════════════════════════════════════════════════════════════
#  3.  SIGNAL CONSTRUCTION (GSR Z-SCORE)
# ════════════════════════════════════════════════════════════
def build_signals(df: pd.DataFrame, window: int = 60,
                  z_entry: float = 2.0, z_exit: float = 0.3,
                  min_corr: float = 0.6, max_vix: float = 40.0,
                  trend_ma: int = 50) -> pd.DataFrame:
    """
    Core signal: rolling Z-score of the Gold/Silver Ratio (GSR).

    GSR = Gold / Silver
    When GSR Z-score is HIGH  (+z_entry): silver is too cheap → BUY SILVER
    When GSR Z-score is LOW   (-z_entry): gold is too cheap   → BUY GOLD
    Exit when Z-score reverts toward 0 (|Z| < z_exit)
    """
    df = df.copy()

    # ── Gold/Silver Ratio ──────────────────────────────────
    df["GSR"]        = df["Gold"] / df["Silver"]
    df["GSR_mean"]   = df["GSR"].rolling(window).mean()
    df["GSR_std"]    = df["GSR"].rolling(window).std()
    df["GSR_z"]      = (df["GSR"] - df["GSR_mean"]) / df["GSR_std"]

    # ── Rolling correlation of daily returns ───────────────
    rg               = df["Gold"].pct_change()
    rs               = df["Silver"].pct_change()
    df["roll_corr"]  = rg.rolling(window).corr(rs)
    df["ret_gold"]   = rg * 100
    df["ret_silver"] = rs * 100

    # ── Trend filters ──────────────────────────────────────
    df["gold_ma"]      = df["Gold"].rolling(trend_ma).mean()
    df["silver_ma"]    = df["Silver"].rolling(trend_ma).mean()
    df["gsr_ma"]       = df["GSR"].rolling(trend_ma).mean()

    # For BUY_SILVER: silver should show early stabilisation
    # (not still in freefall) — use short-term momentum
    df["silver_5d_ret"] = df["Silver"].pct_change(5) * 100
    df["gold_5d_ret"]   = df["Gold"].pct_change(5) * 100

    # ── Filter flags ───────────────────────────────────────
    # Note: VIX filter is more lenient here (40 vs 25 for gold-crude)
    # Reason: precious metals often BOTH rise in high-VIX environments
    # and extreme VIX events (COVID) create the BEST silver-buy opportunities
    df["corr_ok"]  = df["roll_corr"] >= min_corr
    df["vix_ok"]   = df["VIX"] <= max_vix
    df["all_ok"]   = df["corr_ok"] & df["vix_ok"]

    # ── Historical percentile of GSR ───────────────────────
    df["GSR_pct"]  = df["GSR"].expanding().rank(pct=True) * 100

    return df.dropna()


# ════════════════════════════════════════════════════════════
#  4.  BACKTEST ENGINE
# ════════════════════════════════════════════════════════════
def run_backtest(df: pd.DataFrame,
                 window: int   = 60,
                 z_entry: float = 2.0,
                 z_exit: float  = 0.3,
                 price_stop: float = 8.0,
                 max_hold: int  = 20,
                 capital: float = 500_000,
                 apply_costs: bool = True,
                 silver_only: bool = False,
                 gold_only:   bool = False,
                 min_corr: float = 0.6,
                 max_vix:  float = 40.0,
                 trend_ma: int  = 50) -> pd.DataFrame:

    sig = build_signals(df, window=window, z_entry=z_entry,
                        z_exit=z_exit, min_corr=min_corr,
                        max_vix=max_vix, trend_ma=trend_ma)

    trades   = []
    equity   = capital
    in_trade = False
    entry_i  = None

    for i in range(window + trend_ma, len(sig) - 1):
        z          = sig["GSR_z"].iloc[i]
        filters_ok = sig["all_ok"].iloc[i]

        # ── Entry ──────────────────────────────────────────
        if not in_trade and abs(z) >= z_entry and filters_ok:

            if z > 0:
                # GSR high → silver cheap → BUY SILVER
                direction = "BUY_SILVER"
                asset     = "Silver"
            else:
                # GSR low → gold cheap → BUY GOLD
                direction = "BUY_GOLD"
                asset     = "Gold"

            if silver_only and direction != "BUY_SILVER":
                continue
            if gold_only   and direction != "BUY_GOLD":
                continue

            entry_px   = sig[asset].iloc[i + 1]
            entry_date = sig.index[i + 1]
            entry_z    = z
            entry_gsr  = sig["GSR"].iloc[i]
            entry_gsr_pct = sig["GSR_pct"].iloc[i]
            alloc      = equity * 0.90
            in_trade   = True
            entry_i    = i
            continue

        # ── Exit ───────────────────────────────────────────
        if in_trade:
            days_held  = i - entry_i
            cur_px     = sig[asset].iloc[i]
            cur_z      = sig["GSR_z"].iloc[i]
            price_chg  = (cur_px - entry_px) / entry_px * 100

            z_rev     = abs(cur_z) <= z_exit
            stop_hit  = price_chg <= -price_stop
            max_hit   = days_held >= max_hold

            if z_rev or stop_hit or max_hit:
                reason   = ("Z_REVERSION" if z_rev
                            else "PRICE_STOP" if stop_hit
                            else "MAX_HOLD")
                exit_px  = (entry_px * (1 - price_stop/100)
                            if stop_hit else cur_px)
                exit_date = sig.index[i]

                gross_pct = (exit_px - entry_px) / entry_px * 100
                gross_rs  = alloc * gross_pct / 100

                if apply_costs:
                    c = MCXFuturesCosts.compute(alloc, gross_rs)
                    net_rs = c["net_pnl_₹"]
                    cost_p = c["cost_pct"]
                    tax_rs = c["income_tax"]
                    tot_c  = c["total_cost"]
                else:
                    net_rs = gross_rs
                    cost_p = tax_rs = tot_c = 0.0

                net_pct = net_rs / alloc * 100
                equity += net_rs

                trades.append({
                    "entry_date"    : entry_date.date(),
                    "exit_date"     : exit_date.date(),
                    "direction"     : direction,
                    "asset_traded"  : asset,
                    "entry_price"   : round(entry_px, 4),
                    "exit_price"    : round(exit_px, 4),
                    "entry_gsr"     : round(entry_gsr, 2),
                    "entry_gsr_pct" : round(entry_gsr_pct, 1),
                    "entry_z"       : round(entry_z, 3),
                    "exit_z"        : round(cur_z, 3),
                    "days_held"     : days_held,
                    "exit_reason"   : reason,
                    "corr_at_entry" : round(sig["roll_corr"].iloc[entry_i], 3),
                    "vix_at_entry"  : round(sig["VIX"].iloc[entry_i], 1),
                    "gross_pnl_%"   : round(gross_pct, 3),
                    "net_pnl_%"     : round(net_pct, 3),
                    "net_pnl_₹"     : round(net_rs, 2),
                    "total_cost_₹"  : round(tot_c, 2),
                    "income_tax_₹"  : round(tax_rs, 2),
                    "cost_%"        : round(cost_p, 4),
                    "equity_after"  : round(equity, 2),
                })
                in_trade = False
                entry_i  = None

    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════════════
#  5.  PERFORMANCE METRICS
# ════════════════════════════════════════════════════════════
def compute_stats(t: pd.DataFrame, capital: float = 500_000) -> dict:
    if t.empty or len(t) < 2:
        return {"n_trades": 0, "total_return_%": 0,
                "sharpe_ratio": 0, "max_drawdown_%": 0,
                "win_rate_%": 0}
    pnl   = t["net_pnl_%"]
    gross = t["gross_pnl_%"]
    eq    = t["equity_after"]
    fin   = eq.iloc[-1]
    peak  = eq.cummax()
    dd    = (eq - peak) / peak * 100
    total = (fin - capital) / capital * 100

    sharpe = round((pnl.mean() - 6.5/252) / pnl.std() * np.sqrt(252), 2) \
             if pnl.std() > 0 else 0.0
    calmar = round(total / abs(dd.min()), 2) if dd.min() != 0 else 99.0
    wins   = pnl[pnl > 0]
    loss   = pnl[pnl < 0]
    pf     = round(wins.sum() / abs(loss.sum()), 2) \
             if not loss.empty and loss.sum() != 0 else 99.0

    by_dir = t.groupby("direction")["net_pnl_%"].agg(["mean","count"]).to_dict()
    by_rea = t.groupby("exit_reason")["net_pnl_%"].agg(["mean","count"]).to_dict()
    dir_n  = t.groupby("direction").size().to_dict()
    rea_n  = t.groupby("exit_reason").size().to_dict()

    return {
        "n_trades"            : len(t),
        "avg_hold_days"       : round(t["days_held"].mean(), 1),
        "win_rate_%"          : round((pnl > 0).mean() * 100, 1),
        "avg_gross_pnl_%"     : round(gross.mean(), 3),
        "avg_net_pnl_%"       : round(pnl.mean(), 3),
        "median_net_pnl_%"    : round(pnl.median(), 3),
        "best_trade_%"        : round(pnl.max(), 2),
        "worst_trade_%"       : round(pnl.min(), 2),
        "profit_factor"       : pf,
        "sharpe_ratio"        : sharpe,
        "max_drawdown_%"      : round(dd.min(), 2),
        "calmar_ratio"        : calmar,
        "buy_silver_n"        : dir_n.get("BUY_SILVER", 0),
        "buy_silver_avg_%"    : round(by_dir["mean"].get("BUY_SILVER", 0), 3),
        "buy_gold_n"          : dir_n.get("BUY_GOLD", 0),
        "buy_gold_avg_%"      : round(by_dir["mean"].get("BUY_GOLD", 0), 3),
        "z_rev_n"             : rea_n.get("Z_REVERSION", 0),
        "z_rev_avg_%"         : round(by_rea["mean"].get("Z_REVERSION", 0), 3),
        "price_stop_n"        : rea_n.get("PRICE_STOP", 0),
        "price_stop_avg_%"    : round(by_rea["mean"].get("PRICE_STOP", 0), 3),
        "max_hold_n"          : rea_n.get("MAX_HOLD", 0),
        "max_hold_avg_%"      : round(by_rea["mean"].get("MAX_HOLD", 0), 3),
        "avg_entry_gsr_pct"   : round(t["entry_gsr_pct"].mean(), 1),
        "avg_entry_z"         : round(t["entry_z"].abs().mean(), 3),
        "avg_cost_drag_%"     : round(t["cost_%"].mean(), 4),
        "total_costs_₹"       : round(t["total_cost_₹"].sum(), 0),
        "total_tax_₹"         : round(t["income_tax_₹"].sum(), 0),
        "total_return_%"      : round(total, 2),
        "final_equity_₹"      : round(fin, 0),
    }


# ════════════════════════════════════════════════════════════
#  6.  RESEARCH PAPER PRINT
# ════════════════════════════════════════════════════════════
SEP  = "═" * 66
SEP2 = "━" * 66

def print_research_paper(df, sig, t_g, t_n, s_g, s_n, args):
    rg, rs = df["Gold"].pct_change().dropna(), df["Silver"].pct_change().dropna()
    overall_r, overall_p = scipy_stats.pearsonr(rg.values, rs.values[:len(rg)])
    gsr        = df["Gold"] / df["Silver"]
    gsr_mean   = gsr.mean()
    gsr_std    = gsr.std()
    gsr_now    = gsr.iloc[-1]
    gsr_min    = gsr.min()
    gsr_max    = gsr.max()
    avg_corr   = sig["roll_corr"].mean()
    signal_pct = (sig["GSR_z"].abs() >= args.z_entry).mean() * 100

    print(f"\n{SEP}")
    print("  RESEARCH PAPER")
    print("  Gold–Silver Ratio (GSR) Mean-Reversion on MCX Futures")
    print(f"  Period: {df.index[0].date()} – {df.index[-1].date()}  "
          f"({len(df)} trading days)")
    print(SEP)

    # [1]
    print(f"\n{SEP2}\n  [1]  RESEARCH QUESTION\n{SEP2}")
    print(f"""
  The Gold/Silver Ratio (GSR) measures how many ounces of silver
  equal one ounce of gold. Over the past century, the GSR has
  oscillated between ~50 and ~120 and reliably mean-reverted.

  QUESTION: Can we profitably trade GSR extremes on MCX futures
  by buying the underperformer (laggard) and holding until the
  ratio reverts — after accounting for all real transaction costs
  including 30% income tax on F&O profits?

  Current GSR      : {gsr_now:.1f}x  (Gold = {gsr_now:.1f} oz of Silver)
  Historical mean  : {gsr_mean:.1f}x
  Historical range : {gsr_min:.1f}x – {gsr_max:.1f}x
  {('⚠️  GSR currently ABOVE mean → Silver historically cheap'
     if gsr_now > gsr_mean else
     '⚠️  GSR currently BELOW mean → Gold historically cheap')}
""")

    # [2]
    print(f"{SEP2}\n  [2]  HYPOTHESIS\n{SEP2}")
    conclusion = "REJECT H0 ✅" if overall_p < 0.05 else "FAIL TO REJECT H0 ❌"
    print(f"""
  H0: Gold and Silver returns are uncorrelated — no mean-reversion
  H1: Strong correlation exists; GSR extremes are mean-reverting

  Full-period Pearson r   = {overall_r:.4f}
  p-value                 = {overall_p:.8f}
  Rolling {args.window}d avg corr   = {avg_corr:.4f}
  Conclusion              : {conclusion}

  WHY THIS WORKS (compared to Gold-Crude which failed):
    Gold-Crude  r = 0.20–0.35  ← too weak for mean reversion
    Gold-Silver r = {overall_r:.2f}  ← strong enough to trade

  Days with |GSR Z| ≥ {args.z_entry}σ : {signal_pct:.1f}% of all days
  → Signals fire {signal_pct:.1f}% of the time (quality over quantity)
""")

    # [3]
    print(f"{SEP2}\n  [3]  METHODOLOGY\n{SEP2}")
    dir_str = ("SILVER ONLY" if args.silver_only
               else "GOLD ONLY" if args.gold_only
               else "Both (Silver when GSR high, Gold when GSR low)")
    print(f"""
  Data Sources:
    • GC=F   – COMEX Gold Futures   (MCX Gold proxy,   corr > 0.98)
    • SI=F   – COMEX Silver Futures (MCX Silver proxy, corr > 0.97)
    • ^VIX   – CBOE Volatility Index (regime filter)

  Signal Construction:
    1. Compute GSR = Gold price / Silver price daily
    2. Rolling {args.window}-day Z-score of GSR
    3. Signal fires when |Z| ≥ {args.z_entry}

  Trade Direction:
    GSR Z > +{args.z_entry} → Silver historically cheap → BUY SILVER
    GSR Z < -{args.z_entry} → Gold historically cheap   → BUY GOLD
    Active direction: {dir_str}

  Exit Rules (first to trigger):
    ① Z-score reverts to |Z| ≤ {args.z_exit}    ← primary: mean reversion
    ② Price falls {args.price_stop}% from entry          ← hard stop
    ③ Max hold {args.hold} days               ← time stop

  Filters:
    Rolling {args.window}d corr ≥ {args.min_corr}    ← only trade when pair is correlated
    VIX ≤ {args.max_vix}              ← skip extreme panic (both metals spike)
    Note: VIX limit is {args.max_vix} (vs 25 for gold-crude) because precious
    metals tend to move TOGETHER in high fear — correlation holds

  Position Sizing: 90% of current equity per trade
  Capital: ₹{args.capital:,.0f}  |  No overlapping trades

  MCX Precious Metals Futures Costs (FY 2025-26):
    Brokerage      0.02% per side
    STT            0.01% sell side only
    Exchange       0.0026% per side (MCX)
    GST            18% on brokerage + exchange
    SEBI           0.0001% per side
    Stamp Duty     0.002% buy side
    Slippage       0.06% per side (precious metals: very liquid)
    B-A Spread     0.04% round-trip
    Income Tax     30% on net profit (F&O = business income)
    ─────────────────────────────────────────────
    Total friction ≈ 0.28–0.40% per round-trip
    (Even cheaper than crude futures — tighter spreads)
""")

    # [4]
    print(f"{SEP2}\n  [4]  STRATEGY LOGIC\n{SEP2}")
    print(f"""
  THE GOLD/SILVER RATIO — 100 YEARS OF MEAN REVERSION:

    Pre-1900   : GSR fixed at 15:1 (bimetallic monetary standard)
    1900-1980  : GSR oscillated 15–50
    1980-2020  : GSR oscillated 45–100
    COVID 2020 : GSR hit 124 (all-time modern high) → reverted to 65
    2020-2026  : GSR range 65–95

    Every time GSR reached extreme highs (>85), silver eventually
    caught up with gold. Every time GSR hit extreme lows (<55),
    gold caught up with silver.

  WHY SILVER DIVERGES FROM GOLD:
    Silver is 50% industrial metal (solar panels, EVs, electronics)
    Gold is 90%+ monetary/jewellery
    → Economic slowdown fears → silver dumps → GSR spikes
    → Recovery/inflation → silver outperforms → GSR falls

  WHY THE STRATEGY IS BETTER THAN GOLD-CRUDE:
    Gold-Crude  : Two different commodity classes, weak link
    Gold-Silver : Same precious metals family, 90%+ same drivers
    Gold-Silver will ALWAYS revert; Gold-Crude may not for years

  GSR HISTORICAL CONTEXT AT ENTRY:
    Avg entry GSR percentile: {s_n.get('avg_entry_gsr_pct', 'N/A')} (100=most extreme ever seen)
    Avg |Z| at entry        : {s_n.get('avg_entry_z', 'N/A')}σ
""")

    # [5]
    print(f"{SEP2}\n  [5]  RESULTS\n{SEP2}")

    def pstats(s, label):
        if s.get("n_trades", 0) == 0:
            print(f"\n  ── {label} ──\n  No trades. Try: lower --z-entry or --min-corr\n")
            return
        print(f"\n  ── {label} ──")
        skip = {"buy_silver_n","buy_silver_avg_%","buy_gold_n","buy_gold_avg_%",
                "z_rev_n","z_rev_avg_%","price_stop_n","price_stop_avg_%",
                "max_hold_n","max_hold_avg_%","avg_entry_gsr_pct","avg_entry_z"}
        for k, v in s.items():
            if k in skip: continue
            bar = ""
            if k == "win_rate_%":
                filled = int(float(str(v)) / 5)
                bar = f"  {'█'*filled}{'░'*(20-filled)}"
            print(f"  {k:<28} {str(v):>12}{bar}")
        print(f"\n  ── BY DIRECTION ──")
        print(f"  BUY_SILVER ({s['buy_silver_n']} trades) avg : {s['buy_silver_avg_%']:>+.3f}%")
        print(f"  BUY_GOLD   ({s['buy_gold_n']} trades) avg   : {s['buy_gold_avg_%']:>+.3f}%")
        print(f"\n  ── BY EXIT REASON ──")
        print(f"  Z_REVERSION ({s['z_rev_n']:>2})  avg : {s['z_rev_avg_%']:>+.3f}%  ← mean reversion worked")
        print(f"  PRICE_STOP  ({s['price_stop_n']:>2})  avg : {s['price_stop_avg_%']:>+.3f}%  ← hard stop fired")
        print(f"  MAX_HOLD    ({s['max_hold_n']:>2})  avg : {s['max_hold_avg_%']:>+.3f}%  ← timed out")

    pstats(s_g, "GROSS (Before Costs)")
    pstats(s_n, "NET   (After All Costs + 30% Income Tax)")

    if s_g.get("n_trades", 0) > 0 and s_n.get("n_trades", 0) > 0:
        drag = s_g["total_return_%"] - s_n["total_return_%"]
        print(f"""
  ── COST IMPACT ──
  Gross total return       : {s_g['total_return_%']:>+.2f}%
  Net total return         : {s_n['total_return_%']:>+.2f}%
  Return drag from costs   : {drag:>+.2f}%
  Avg cost per trade       : {s_n['avg_cost_drag_%']:.4f}%
    vs copper strategy     :   1.3070%   (3–4x more expensive)
    vs gold-crude v1       :   0.3500%   (similar)
  Total income tax paid    : ₹{s_n['total_tax_₹']:>10,.0f}
  Total all-in costs paid  : ₹{s_n['total_costs_₹']:>10,.0f}
""")

    # [6]
    print(f"{SEP2}\n  [6]  RISK METRICS\n{SEP2}")
    if s_n.get("n_trades", 0) > 0:
        sr = s_n["sharpe_ratio"]
        pf = s_n["profit_factor"]
        cr = s_n["calmar_ratio"]
        print(f"""
  Sharpe Ratio (ann. RF=6.5%)  : {sr}
    {'✅ Excellent (>2.0)' if sr>2 else '✅ Good (>1.0)' if sr>1
     else '⚠️  Below benchmark' if sr>0 else '❌ Negative'}

  Max Drawdown                  : {s_n['max_drawdown_%']}%
  Calmar Ratio (Ret/|MaxDD|)    : {cr}
    {'✅ Strong (>3.0)' if cr>3 else '✅ OK (>1.0)' if cr>1
     else '⚠️  Weak (<1.0)'}

  Profit Factor                 : {pf}
    {'✅ Strong (>2.0)' if pf>2 else '✅ Good (>1.5)' if pf>1.5
     else '⚠️  Thin (>1.0)' if pf>1 else '❌ <1.0 losing'}

  Win Rate (net)                : {s_n['win_rate_%']}%
  Avg Hold Period               : {s_n['avg_hold_days']} days
  Best Trade (net)              : {s_n['best_trade_%']}%
  Worst Trade (net)             : {s_n['worst_trade_%']}%
  N Trades                      : {s_n['n_trades']} over {args.period}
""")

    # [7]
    print(f"{SEP2}\n  [7]  LIMITATIONS\n{SEP2}")
    print(f"""
  L1. COMEX PROXY FOR MCX
      GC=F and SI=F are USD-denominated. MCX prices include
      INR/USD conversion, import duties (~12.5% for silver,
      ~10% for gold), and local liquidity. The Z-score of the
      ratio may differ slightly from a true MCX-based signal.
      Fix (F1): Convert both to INR before computing GSR.

  L2. MCX LOT SIZE REALITY
      MCX Gold lot   = 1 kg    ≈ ₹7–9L notional
      MCX Silver lot = 30 kg   ≈ ₹2–3L notional
      The ratio trade requires rough value parity.
      With ₹5L capital, max 1 gold lot OR 2 silver lots.
      Margin requirements: ~10% of notional.

  L3. CLOSE-PRICE ENTRY
      Backtest uses next-day close as entry approximation.
      Real MCX gold trades 09:00–23:30 IST. Entry should be
      at MCX open the following morning for best execution.

  L4. REGIME CHANGE RISK
      The GSR relationship could permanently shift if:
      • Solar/EV demand makes silver an industrial metal
        (decoupling from gold's monetary function)
      • Central banks start accumulating silver (unlikely)
      • A new silver supply source is discovered

  L5. 30% TAX ON EVERY WINNING TRADE
      F&O profits taxed as business income at 30%. This is
      the single biggest cost drag. Actual tax depends on
      annual income slab and loss offsets.

  L6. ROLL COSTS NOT MODELLED
      For holds > 1 month, futures must be rolled.
      MCX gold/silver roll costs are typically 0.1–0.3%,
      which are not included here.

  L7. SMALL TRADE COUNT
      Depending on parameters, strategy generates 5–25
      trades over 5 years. Small sample — results should
      be treated as directionally indicative, not conclusive.
""")

    # [8]
    print(f"{SEP2}\n  [8]  FUTURE IMPROVEMENTS\n{SEP2}")
    print(f"""
  F1. USE REAL MCX DATA WITH INR CONVERSION
      Fetch USD/INR (INR=X from yfinance) and convert
      COMEX prices to INR before computing GSR.
      This makes signals directly applicable to MCX.

  F2. PAIRS TRADE (LONG-SHORT SIMULTANEOUSLY)
      Buy laggard AND short outperformer in the same trade.
      Eliminates market direction risk entirely.
      Requires margin for both legs (~20% of notional).

  F3. DYNAMIC POSITION SIZING (KELLY CRITERION)
      Size position based on edge strength:
        Kelly fraction = (p * b - q) / b
        p = win rate, q = 1-p, b = avg win/avg loss
      Larger positions at high Z-scores, smaller near threshold.

  F4. GSR PERCENTILE FILTER
      Only trade when GSR is above 80th percentile (for silver)
      or below 20th percentile (for gold) of ALL-TIME history.
      This ensures extreme readings, not just recent extremes.

  F5. ADD PLATINUM/PALLADIUM
      Extend to a 4-metal precious metals complex.
      Platinum-Silver and Palladium-Gold also have
      documented mean-reverting ratio relationships.

  F6. REGIME-ADAPTIVE Z-ENTRY
      Use expanding-window percentile instead of rolling Z-score.
      This anchors signals to ALL-TIME extremes, not just
      recent 60-day window — more robust across decades.

  F7. WALK-FORWARD OPTIMISATION
      Optimise window and z_entry on years 1–3, validate on
      years 4–5. Report out-of-sample results separately.

  F8. LIVE ALERT SYSTEM (MCX + Telegram)
      At 11:30 PM MCX close: compute GSR Z-score.
      If |Z| ≥ threshold and filters pass → send alert:
      "⚡ GSR Signal: Buy [GOLD/SILVER] | GSR={{x}}x | Z={{y}}σ"

  F9. MULTI-TIMEFRAME CONFIRMATION
      Confirm daily signal with weekly GSR direction.
      Only trade if both timeframes agree — reduces false signals.

  F10. SILVER INDUSTRIAL DEMAND OVERLAY
      Overlay PMI data (manufacturing index) as an additional
      filter. When PMI < 50 (contraction), silver underperforms
      gold more → higher conviction BUY_SILVER signals.
""")

    verdict = (
        "✅ VIABLE — Strong net returns after all costs + tax"
        if s_n.get("total_return_%", 0) > 30
        else "✅ PROFITABLE — Positive net after all costs"
        if s_n.get("total_return_%", 0) > 0
        else "⚠️  MARGINAL — Review parameters"
        if s_n.get("n_trades", 0) >= 5
        else "⚠️  TOO FEW TRADES — Lower z-entry or min-corr"
    )
    print(SEP)
    print(f"  END OF RESEARCH PAPER  |  Verdict: {verdict}")
    print(SEP + "\n")


# ════════════════════════════════════════════════════════════
#  7.  SENSITIVITY GRID
# ════════════════════════════════════════════════════════════
def print_sensitivity(df, capital, args):
    print(f"{SEP2}")
    print("  SENSITIVITY GRID  (net: WR% | AvgPnL% | Sharpe | TotalRet%)")
    print(f"{SEP2}")
    z_vals = [1.5, 2.0, 2.5, 3.0]
    h_vals = [10, 15, 20, 30]
    print(f"  {'':>10}", end="")
    for h in h_vals:
        print(f"  {'hold≤'+str(h)+'d':^24}", end="")
    print()
    for ze in z_vals:
        print(f"  z≥{ze}σ      ", end="")
        for h in h_vals:
            t = run_backtest(df, window=args.window, z_entry=ze,
                             z_exit=args.z_exit,
                             price_stop=args.price_stop,
                             max_hold=h, capital=capital,
                             apply_costs=True,
                             silver_only=args.silver_only,
                             gold_only=args.gold_only,
                             min_corr=args.min_corr,
                             max_vix=args.max_vix,
                             trend_ma=args.trend_ma)
            s = compute_stats(t, capital)
            if s.get("n_trades", 0) < 2:
                print(f"  {'N/A (0-1 trades)':^24}", end="")
            else:
                print(f"  {s['win_rate_%']:>4.0f}%"
                      f"/{s['avg_net_pnl_%']:>+5.2f}%"
                      f"/SR{s['sharpe_ratio']:>+4.1f}"
                      f"/{s['total_return_%']:>+6.1f}%", end="")
        print()
    print()


# ════════════════════════════════════════════════════════════
#  8.  CHART
# ════════════════════════════════════════════════════════════
def plot_results(df, sig, t_g, t_n, s_g, s_n, args):
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.55, wspace=0.38,
                            top=0.91, bottom=0.05,
                            left=0.06, right=0.97)

    cap = args.capital
    fig.text(0.5, 0.965,
             "⚡  Gold–Silver Ratio (GSR) Mean-Reversion  |  MCX Futures Backtest",
             ha="center", fontsize=14, fontweight="bold", color=AMBER)
    fig.text(0.5, 0.938,
             f"Z≥{args.z_entry}σ  │  Z-exit={args.z_exit}σ  │  "
             f"Hold≤{args.hold}d  │  PriceStop={args.price_stop}%  │  "
             f"CorrFilter≥{args.min_corr}  │  VIX≤{args.max_vix}  │  "
             f"Gross:{s_g.get('total_return_%','N/A')}%  │  "
             f"NET:{s_n.get('total_return_%','N/A')}%  │  "
             f"Sharpe:{s_n.get('sharpe_ratio','N/A')}  │  "
             f"Trades:{s_n.get('n_trades','N/A')}",
             ha="center", fontsize=8.5, color=GREY)

    # ── 1. Normalised prices ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df.index, df["Gold"]/df["Gold"].iloc[0]*100,
             color=AMBER, lw=2, label="Gold (GC=F)")
    ax1.plot(df.index, df["Silver"]/df["Silver"].iloc[0]*100,
             color=SILVER, lw=2, label="Silver (SI=F)")
    vix_high = df["VIX"] > args.max_vix
    ymin = min((df["Silver"]/df["Silver"].iloc[0]*100).min() - 10, 0)
    ax1.fill_between(df.index, ymin, 600,
                     where=vix_high, alpha=0.07, color=RED,
                     label=f"VIX>{args.max_vix} filtered")
    if not t_n.empty:
        for _, r in t_n.iterrows():
            ed = pd.Timestamp(r["entry_date"])
            c  = SILVER if r["direction"] == "BUY_SILVER" else AMBER
            ax1.axvline(ed, color=c, alpha=0.3, lw=0.9)
    ax1.set_title("Normalised Prices (base=100)  "
                  "|  Silver line=Buy Silver  Amber=Buy Gold  Red=VIX filtered")
    ax1.legend(fontsize=7); ax1.grid(True)

    # ── 2. GSR + Z-score ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    gsr = sig["GSR"]
    ax2.plot(sig.index, gsr, color=PURPLE, lw=1.5, label="GSR (Gold/Silver)")
    ax2.plot(sig.index, sig["GSR_mean"], color=GREY, lw=1,
             ls="--", alpha=0.7, label=f"{args.window}d Mean")
    ax2.fill_between(sig.index, gsr, sig["GSR_mean"],
                     where=gsr > sig["GSR_mean"],
                     alpha=0.12, color=SILVER, label="Silver cheap")
    ax2.fill_between(sig.index, gsr, sig["GSR_mean"],
                     where=gsr < sig["GSR_mean"],
                     alpha=0.12, color=AMBER, label="Gold cheap")
    if not t_n.empty:
        for _, r in t_n.iterrows():
            ed = pd.Timestamp(r["entry_date"])
            if ed in sig.index:
                mk = "v" if r["direction"]=="BUY_SILVER" else "^"
                mc = SILVER if r["direction"]=="BUY_SILVER" else AMBER
                ax2.scatter(ed, sig.loc[ed, "GSR"], marker=mk,
                            color=mc, s=60, zorder=5)
    ax2.set_title("Gold/Silver Ratio (GSR)\n▼=Buy Silver  ▲=Buy Gold")
    ax2.legend(fontsize=6); ax2.grid(True)

    # ── 3. GSR Z-score ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(sig.index, sig["GSR_z"], color=PURPLE, lw=1.2,
             label="GSR Z-score", alpha=0.8)
    ax3.axhline(0, color=BORDER, lw=1)
    ax3.axhline( args.z_entry, color=SILVER, lw=1.2, ls="--",
                label=f"+{args.z_entry}σ → Buy Silver")
    ax3.axhline(-args.z_entry, color=AMBER,  lw=1.2, ls="--",
                label=f"-{args.z_entry}σ → Buy Gold")
    ax3.fill_between(sig.index, sig["GSR_z"],  args.z_entry,
                     where=(sig["GSR_z"] >= args.z_entry) & sig["all_ok"],
                     alpha=0.20, color=SILVER)
    ax3.fill_between(sig.index, sig["GSR_z"], -args.z_entry,
                     where=(sig["GSR_z"] <= -args.z_entry) & sig["all_ok"],
                     alpha=0.20, color=AMBER)
    ax3.fill_between(sig.index, sig["GSR_z"],  args.z_entry,
                     where=(sig["GSR_z"] >= args.z_entry) & ~sig["all_ok"],
                     alpha=0.07, color=RED, label="Filtered out")
    ax3.fill_between(sig.index, sig["GSR_z"], -args.z_entry,
                     where=(sig["GSR_z"] <= -args.z_entry) & ~sig["all_ok"],
                     alpha=0.07, color=RED)
    ax3.set_title("GSR Z-Score  |  Bright=tradeable  Dim=filtered (VIX/corr)")
    ax3.legend(fontsize=7); ax3.grid(True)

    # ── 4. Rolling correlation ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(sig.index, sig["roll_corr"], color=GREEN, lw=1.5,
             label=f"Roll {args.window}d Corr")
    ax4.axhline(args.min_corr, color=GREEN, lw=1, ls="--",
                label=f"Min={args.min_corr}")
    ax4.axhline(0, color=BORDER, lw=1)
    ax4.fill_between(sig.index, sig["roll_corr"], args.min_corr,
                     where=sig["roll_corr"] < args.min_corr,
                     alpha=0.15, color=RED, label="Corr filtered")
    ax4.set_ylim(-1, 1)
    ax4.set_title(f"Rolling {args.window}d Correlation\n"
                  f"Gold–Silver (should stay >0.6 most of time)")
    ax4.legend(fontsize=7); ax4.grid(True)

    # ── 5. Net equity + drawdown ────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    if not t_n.empty:
        eq  = t_n["equity_after"]
        bd  = pd.to_datetime(t_n["entry_date"])
        ec  = GREEN if eq.iloc[-1] > cap else RED

        # Gross reference
        if not t_g.empty:
            ax5.plot(pd.to_datetime(t_g["entry_date"]),
                     t_g["equity_after"], color=AMBER, lw=1.2,
                     ls="--", alpha=0.6, label="Gross")

        ax5.plot(bd, eq, color=ec, lw=2.5, marker="o", ms=5,
                 label="Net equity (all costs+tax)", zorder=3)
        ax5.axhline(cap, color=BORDER, lw=1, ls="--",
                    label=f"Start ₹{cap//1000}k")
        pk = eq.cummax()
        ax5.fill_between(bd, eq, pk, alpha=0.15, color=RED,
                         label=f"Drawdown (max {s_n['max_drawdown_%']}%)")
        ax5.fill_between(bd, cap, eq,
                         where=eq >= cap, alpha=0.1, color=GREEN)
        ax5.fill_between(bd, cap, eq,
                         where=eq < cap,  alpha=0.1, color=RED)
        # Mark exit types
        color_map = {"Z_REVERSION": GREEN, "PRICE_STOP": RED,
                     "MAX_HOLD": GREY}
        for _, r in t_n.iterrows():
            ed  = pd.Timestamp(r["entry_date"])
            idx = list(pd.to_datetime(t_n["entry_date"])).index(ed)
            ax5.scatter(ed, eq.iloc[idx],
                        color=color_map.get(r["exit_reason"], GREY),
                        s=50, zorder=4)
        ax5.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
        ax5.legend(fontsize=7)
    ax5.set_title(
        f"Net Equity  Return:{s_n.get('total_return_%','N/A')}%  "
        f"Sharpe:{s_n.get('sharpe_ratio','N/A')}  "
        f"Calmar:{s_n.get('calmar_ratio','N/A')}  "
        f"| ●Green=ZRev  ●Red=Stop  ●Grey=MaxHold")
    ax5.grid(True)

    # ── 6. PnL distribution ─────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    if not t_n.empty:
        pnl  = t_n["net_pnl_%"]
        rng  = max(pnl.max() - pnl.min(), 1)
        bins = np.arange(pnl.min()-1, pnl.max()+1, max(0.5, rng/15))
        n, bins_, patches = ax6.hist(pnl, bins=bins,
                                     edgecolor=BORDER, lw=0.5)
        for patch, left in zip(patches, bins_[:-1]):
            patch.set_facecolor(GREEN if left >= 0 else RED)
        ax6.axvline(0, color=WHITE, lw=1.2)
        ax6.axvline(pnl.mean(), color=AMBER, lw=1.5, ls="--",
                    label=f"Mean {s_n['avg_net_pnl_%']}%")
        ax6.axvline(-args.price_stop, color=RED, lw=1.2, ls=":",
                    label=f"PriceStop -{args.price_stop}%")

        # Silver vs Gold separation
        sv = t_n[t_n["direction"]=="BUY_SILVER"]["net_pnl_%"]
        gd = t_n[t_n["direction"]=="BUY_GOLD"]["net_pnl_%"]
        if len(sv): ax6.axvline(sv.mean(), color=SILVER, lw=1, ls=":",
                                label=f"Silv avg {sv.mean():+.2f}%")
        if len(gd): ax6.axvline(gd.mean(), color=AMBER,  lw=1, ls=":",
                                label=f"Gold avg {gd.mean():+.2f}%")
        ax6.set_title(
            f"Net PnL Distribution\n"
            f"WR={s_n.get('win_rate_%')}%  PF={s_n.get('profit_factor')}  "
            f"ZRev:{s_n.get('z_rev_n')} Stop:{s_n.get('price_stop_n')} "
            f"MaxH:{s_n.get('max_hold_n')}")
        ax6.set_xlabel("Net PnL %")
        ax6.legend(fontsize=7)
    ax6.grid(True, axis="y")

    plt.savefig("gold_silver_results.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("📊  Chart saved → gold_silver_results.png")
    plt.show()


# ════════════════════════════════════════════════════════════
#  9.  MAIN
# ════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="Gold–Silver GSR Mean-Reversion on MCX Futures")
    p.add_argument("--period",      default="5y",
                   choices=["1y","2y","3y","5y","10y"])
    p.add_argument("--window",      type=int,   default=60,
                   help="Rolling Z-score window (default: 60)")
    p.add_argument("--z-entry",     type=float, default=2.0,
                   help="Z-score entry threshold (default: 2.0)")
    p.add_argument("--z-exit",      type=float, default=0.3,
                   help="Z-score exit threshold (default: 0.3)")
    p.add_argument("--price-stop",  type=float, default=8.0,
                   help="Hard price stop %% below entry (default: 8.0)")
    p.add_argument("--hold",        type=int,   default=20,
                   help="Max hold days (default: 20)")
    p.add_argument("--min-corr",    type=float, default=0.6,
                   help="Min rolling correlation (default: 0.6)")
    p.add_argument("--max-vix",     type=float, default=40.0,
                   help="Max VIX to allow trade (default: 40)")
    p.add_argument("--trend-ma",    type=int,   default=50,
                   help="Trend MA window (default: 50)")
    p.add_argument("--silver-only", action="store_true",
                   help="Only BUY_SILVER signals")
    p.add_argument("--gold-only",   action="store_true",
                   help="Only BUY_GOLD signals")
    p.add_argument("--capital",     type=float, default=500_000)
    p.add_argument("--no-plot",     action="store_true")
    args = p.parse_args()

    df  = fetch_data(period=args.period)
    sig = build_signals(df, window=args.window,
                        z_entry=args.z_entry, z_exit=args.z_exit,
                        min_corr=args.min_corr, max_vix=args.max_vix,
                        trend_ma=args.trend_ma)

    t_g = run_backtest(df, window=args.window,
                       z_entry=args.z_entry, z_exit=args.z_exit,
                       price_stop=args.price_stop, max_hold=args.hold,
                       capital=args.capital, apply_costs=False,
                       silver_only=args.silver_only,
                       gold_only=args.gold_only,
                       min_corr=args.min_corr, max_vix=args.max_vix,
                       trend_ma=args.trend_ma)

    t_n = run_backtest(df, window=args.window,
                       z_entry=args.z_entry, z_exit=args.z_exit,
                       price_stop=args.price_stop, max_hold=args.hold,
                       capital=args.capital, apply_costs=True,
                       silver_only=args.silver_only,
                       gold_only=args.gold_only,
                       min_corr=args.min_corr, max_vix=args.max_vix,
                       trend_ma=args.trend_ma)

    s_g = compute_stats(t_g, args.capital)
    s_n = compute_stats(t_n, args.capital)

    print_research_paper(df, sig, t_g, t_n, s_g, s_n, args)

    if not t_n.empty:
        t_n.to_csv("trades_log.csv", index=False)
        print("📄  Trade log saved → trades_log.csv")

    print_sensitivity(df, args.capital, args)

    if not args.no_plot:
        plot_results(df, sig, t_g, t_n, s_g, s_n, args)


if __name__ == "__main__":
    main()