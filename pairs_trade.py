"""
╔══════════════════════════════════════════════════════════════════╗
║  PAIRS TRADE: Long Silver + Short Gold Simultaneously           ║
║  Gold–Silver GSR Strategy — Market-Neutral Version              ║
╠══════════════════════════════════════════════════════════════════╣
║  HOW IT WORKS:                                                  ║
║                                                                  ║
║  Single-leg (old):                                              ║
║    GSR high → BUY SILVER only                                   ║
║    Risk: if both gold & silver crash together, you lose         ║
║                                                                  ║
║  Pairs trade (new):                                             ║
║    GSR high → BUY SILVER + SHORT GOLD (equal notional)         ║
║    GSR low  → BUY GOLD   + SHORT SILVER (equal notional)       ║
║    Risk: only exposed to SPREAD movement, not direction         ║
║                                                                  ║
║  Example:                                                        ║
║    Gold +5%, Silver +8% → P&L = +8% - 5% = +3% on spread      ║
║    Gold -5%, Silver -2% → P&L = -2% - (-5%) = +3% on spread   ║
║    Both crash 20% → P&L = -20% - (-20%) = 0% (protected!)     ║
║                                                                  ║
║  MCX FUTURES SHORTING:                                          ║
║    Selling MCX Gold/Silver futures = short position             ║
║    Perfectly legal, standard practice                           ║
║    Margin required: ~10% of notional per leg                    ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python pairs_trade.py
  python pairs_trade.py --period 10y
  python pairs_trade.py --period 10y --z-entry 1.5 --hold 20
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
#  1.  MCX FUTURES COST MODEL — PAIRS VERSION
# ════════════════════════════════════════════════════════════
def compute_progressive_tax(annual_profit_rs: float) -> float:
    """Indian income tax — New Regime + 4% cess."""
    if annual_profit_rs <= 0:
        return 0.0
    SLABS = [(400_000,0.00),(400_000,0.05),(400_000,0.10),
             (400_000,0.15),(400_000,0.20),(400_000,0.25),
             (float("inf"),0.30)]
    tax = 0; rem = annual_profit_rs
    for sz, r in SLABS:
        if rem <= 0: break
        t = min(rem, sz); tax += t*r; rem -= t
    return round(tax * 1.04, 2)


def compute_pairs_costs(leg_value: float, net_spread_pnl: float,
                        annual_profit_so_far: float = 0.0) -> dict:
    """
    Cost model for BOTH legs of a pairs trade.
    Each leg has its own fixed costs.
    Tax applied once on net spread P&L.

    leg_value: notional of each leg (long + short = 2 × leg_value total)
    net_spread_pnl: gross profit from the spread in ₹
    """
    def single_leg_costs(tv):
        brok  = tv * 0.0002 * 2        # buy + close
        stt   = tv * 0.0001            # sell-side only
        exch  = tv * 0.000026 * 2
        gst   = (brok + exch) * 0.18
        sebi  = tv * 0.000001 * 2
        stamp = tv * 0.00002
        slip  = tv * 0.0006 * 2
        sprd  = tv * 0.0004
        return brok+stt+exch+gst+sebi+stamp+slip+sprd

    # Both legs pay fixed costs
    long_costs  = single_leg_costs(leg_value)
    short_costs = single_leg_costs(leg_value)
    total_fixed = long_costs + short_costs

    pnl_after   = net_spread_pnl - total_fixed
    if pnl_after > 0:
        tax_with    = compute_progressive_tax(annual_profit_so_far + pnl_after)
        tax_without = compute_progressive_tax(annual_profit_so_far)
        tax         = max(0.0, tax_with - tax_without)
    else:
        tax = 0.0

    total_cost = total_fixed + tax
    net_pnl    = net_spread_pnl - total_cost

    return {
        "long_costs"       : round(long_costs, 2),
        "short_costs"      : round(short_costs, 2),
        "total_fixed"      : round(total_fixed, 2),
        "income_tax"       : round(tax, 2),
        "total_cost"       : round(total_cost, 2),
        "net_pnl_₹"        : round(net_pnl, 2),
        "cost_pct_of_leg"  : round(total_cost / leg_value * 100, 4),
        "eff_tax_%"        : round(tax/pnl_after*100, 2) if pnl_after>0 else 0,
    }


# ════════════════════════════════════════════════════════════
#  2.  DATA
# ════════════════════════════════════════════════════════════
def get_close(ticker: str, period: str) -> pd.Series:
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    c = raw["Close"]
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    return pd.Series(c.values.flatten(), index=pd.to_datetime(raw.index),
                     name=ticker, dtype=float)


def fetch_data(period: str = "5y") -> pd.DataFrame:
    print(f"\n📥  Fetching data (period={period}) …")
    gold   = get_close("GC=F", period)
    silver = get_close("SI=F", period)
    vix    = get_close("^VIX", period)
    df = pd.concat([gold.rename("Gold"), silver.rename("Silver"),
                    vix.rename("VIX")], axis=1).dropna()
    print(f"✅  {len(df)} days  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ════════════════════════════════════════════════════════════
#  3.  SIGNAL CONSTRUCTION
# ════════════════════════════════════════════════════════════
def build_signals(df: pd.DataFrame, window=60, z_entry=1.5,
                  z_exit=0.3, min_corr=0.5, max_vix=50.0,
                  trend_ma=50) -> pd.DataFrame:
    d = df.copy()
    d["GSR"]       = d["Gold"] / d["Silver"]
    d["GSR_mean"]  = d["GSR"].rolling(window).mean()
    d["GSR_std"]   = d["GSR"].rolling(window).std()
    d["GSR_z"]     = (d["GSR"] - d["GSR_mean"]) / d["GSR_std"]
    rg             = d["Gold"].pct_change()
    rs             = d["Silver"].pct_change()
    d["roll_corr"] = rg.rolling(window).corr(rs)
    d["GSR_pct"]   = d["GSR"].expanding().rank(pct=True) * 100
    d["corr_ok"]   = d["roll_corr"] >= min_corr
    d["vix_ok"]    = d["VIX"] <= max_vix
    d["all_ok"]    = d["corr_ok"] & d["vix_ok"]
    d["ret_gold"]  = rg * 100
    d["ret_silver"]= rs * 100
    return d.dropna()


# ════════════════════════════════════════════════════════════
#  4.  PAIRS BACKTEST ENGINE
# ════════════════════════════════════════════════════════════
def run_pairs_backtest(df: pd.DataFrame,
                       window: int    = 60,
                       z_entry: float = 1.5,
                       z_exit: float  = 0.3,
                       spread_stop: float = 5.0,  # stop on SPREAD loss, not single leg
                       max_hold: int  = 20,
                       capital: float = 500_000,
                       apply_costs: bool = True,
                       min_corr: float = 0.5,
                       max_vix: float  = 50.0,
                       trend_ma: int   = 50) -> pd.DataFrame:
    """
    Pairs trade backtest.

    Each trade allocates:
      - 45% of equity to LONG leg  (buy laggard)
      - 45% of equity to SHORT leg (sell outperformer)
      - Total exposure: 90% (same as single-leg)
      - Net market exposure: ~0 (market neutral)

    P&L = long_return - short_return (spread return)
    Stop: if spread return ≤ -spread_stop%
    """
    sig = build_signals(df, window=window, z_entry=z_entry,
                        z_exit=z_exit, min_corr=min_corr,
                        max_vix=max_vix, trend_ma=trend_ma)

    trades        = []
    equity        = capital
    in_trade      = False
    entry_i       = None
    annual_profit = 0.0
    current_fy    = None

    for i in range(window + trend_ma, len(sig) - 1):
        z          = sig["GSR_z"].iloc[i]
        filters_ok = sig["all_ok"].iloc[i]
        dt         = sig.index[i]

        # Indian FY tracker
        fy = dt.year if dt.month >= 4 else dt.year - 1
        if current_fy is None: current_fy = fy
        if fy != current_fy:
            annual_profit = 0.0
            current_fy    = fy

        # ── Entry ─────────────────────────────────────────
        if not in_trade and abs(z) >= z_entry and filters_ok:
            # Direction based on GSR Z-score
            if z > 0:
                # GSR high: gold overperformed → LONG silver, SHORT gold
                long_asset  = "Silver"
                short_asset = "Gold"
                direction   = "LONG_SILVER_SHORT_GOLD"
            else:
                # GSR low: silver overperformed → LONG gold, SHORT silver
                long_asset  = "Gold"
                short_asset = "Silver"
                direction   = "LONG_GOLD_SHORT_SILVER"

            # Entry prices (next day)
            long_entry  = sig[long_asset].iloc[i + 1]
            short_entry = sig[short_asset].iloc[i + 1]

            # Each leg gets 45% of equity
            leg_value   = equity * 0.45
            entry_z     = z
            entry_gsr   = sig["GSR"].iloc[i]
            entry_gsr_p = sig["GSR_pct"].iloc[i]
            annual_snap = annual_profit

            in_trade = True
            entry_i  = i
            continue

        # ── Exit ──────────────────────────────────────────
        if in_trade:
            days_held   = i - entry_i
            long_cur    = sig[long_asset].iloc[i]
            short_cur   = sig[short_asset].iloc[i]
            cur_z       = sig["GSR_z"].iloc[i]

            # Spread return = long return - short return
            long_ret    = (long_cur  - long_entry)  / long_entry  * 100
            short_ret   = (short_cur - short_entry) / short_entry * 100
            spread_ret  = long_ret - short_ret   # net pairs P&L %

            z_rev    = abs(cur_z) <= z_exit
            stop_hit = spread_ret <= -spread_stop
            max_hit  = days_held >= max_hold

            if z_rev or stop_hit or max_hit:
                reason = ("Z_REVERSION" if z_rev
                          else "SPREAD_STOP" if stop_hit
                          else "MAX_HOLD")

                # Cap spread at stop level if triggered
                if stop_hit:
                    spread_ret = -spread_stop

                # Gross P&L in ₹ (on leg_value notional)
                gross_rs = leg_value * spread_ret / 100

                if apply_costs:
                    c = compute_pairs_costs(leg_value, gross_rs, annual_snap)
                    net_rs  = c["net_pnl_₹"]
                    cost_p  = c["cost_pct_of_leg"]
                    tax_rs  = c["income_tax"]
                    tot_c   = c["total_cost"]
                    eff_tax = c["eff_tax_%"]
                else:
                    net_rs = gross_rs; cost_p = tax_rs = tot_c = eff_tax = 0.0

                net_pct       = net_rs / leg_value * 100
                equity       += net_rs
                annual_profit += max(0.0, gross_rs - (tot_c - tax_rs))

                # Market-neutral check
                long_ret_final  = (long_cur - long_entry)  / long_entry * 100
                short_ret_final = (short_cur - short_entry) / short_entry * 100
                mkt_exposure    = (long_ret_final + short_ret_final) / 2  # should be ~0

                trades.append({
                    "entry_date"      : sig.index[entry_i + 1].date(),
                    "exit_date"       : sig.index[i].date(),
                    "direction"       : direction,
                    "long_asset"      : long_asset,
                    "short_asset"     : short_asset,
                    "long_entry"      : round(long_entry, 4),
                    "short_entry"     : round(short_entry, 4),
                    "long_exit"       : round(long_cur, 4),
                    "short_exit"      : round(short_cur, 4),
                    "long_ret_%"      : round(long_ret_final, 3),
                    "short_ret_%"     : round(short_ret_final, 3),
                    "spread_ret_%"    : round(spread_ret, 3),
                    "mkt_exposure_%"  : round(mkt_exposure, 3),
                    "entry_gsr"       : round(entry_gsr, 2),
                    "entry_gsr_pct"   : round(entry_gsr_p, 1),
                    "entry_z"         : round(entry_z, 3),
                    "exit_z"          : round(cur_z, 3),
                    "days_held"       : days_held,
                    "exit_reason"     : reason,
                    "leg_value_₹"     : round(leg_value, 2),
                    "gross_pnl_%"     : round(spread_ret, 3),
                    "net_pnl_%"       : round(net_pct, 3),
                    "net_pnl_₹"       : round(net_rs, 2),
                    "total_cost_₹"    : round(tot_c, 2),
                    "income_tax_₹"    : round(tax_rs, 2),
                    "eff_tax_%"       : round(eff_tax, 2),
                    "cost_%"          : round(cost_p, 4),
                    "equity_after"    : round(equity, 2),
                })
                in_trade = False
                entry_i  = None

    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════════════
#  5.  PERFORMANCE METRICS
# ════════════════════════════════════════════════════════════
def compute_stats(t: pd.DataFrame, capital: float = 500_000) -> dict:
    if t.empty or len(t) < 2:
        return {"n_trades": 0}
    pnl   = t["net_pnl_%"]
    eq    = t["equity_after"]
    fin   = eq.iloc[-1]
    peak  = eq.cummax()
    dd    = (eq - peak) / peak * 100
    total = (fin - capital) / capital * 100
    sharpe = round((pnl.mean()-6.5/252)/pnl.std()*np.sqrt(252), 2) \
             if pnl.std() > 0 else 0.0
    calmar = round(total / abs(dd.min()), 2) if dd.min() != 0 else 99.0
    wins = pnl[pnl > 0]; loss = pnl[pnl < 0]
    pf   = round(wins.sum()/abs(loss.sum()), 2) \
           if not loss.empty and loss.sum()!=0 else 99.0
    by_d = t.groupby("direction")["net_pnl_%"].agg(["mean","count"]).to_dict()
    by_r = t.groupby("exit_reason")["net_pnl_%"].agg(["mean","count"]).to_dict()
    return {
        "n_trades"          : len(t),
        "avg_hold_days"     : round(t["days_held"].mean(), 1),
        "win_rate_%"        : round((pnl>0).mean()*100, 1),
        "avg_gross_%"       : round(t["gross_pnl_%"].mean(), 3),
        "avg_net_%"         : round(pnl.mean(), 3),
        "best_%"            : round(pnl.max(), 2),
        "worst_%"           : round(pnl.min(), 2),
        "profit_factor"     : pf,
        "sharpe_ratio"      : sharpe,
        "max_drawdown_%"    : round(dd.min(), 2),
        "calmar_ratio"      : calmar,
        "avg_mkt_exposure_%" : round(t["mkt_exposure_%"].abs().mean(), 3),
        "avg_cost_%"        : round(t["cost_%"].mean(), 4),
        "total_costs_₹"     : round(t["total_cost_₹"].sum(), 0),
        "total_tax_₹"       : round(t["income_tax_₹"].sum(), 0),
        "total_return_%"    : round(total, 2),
        "final_equity_₹"    : round(fin, 0),
    }


# ════════════════════════════════════════════════════════════
#  6.  PRINT REPORT
# ════════════════════════════════════════════════════════════
SEP = "═" * 66

def print_report(t_g, t_n, s_g, s_n, t_single, s_single, args):
    print(f"\n{SEP}")
    print("  PAIRS TRADE RESULTS — Gold–Silver GSR Market-Neutral")
    print(SEP)

    def pstat(s, label):
        if s.get("n_trades", 0) == 0:
            print(f"\n  ── {label} ──\n  No trades.\n"); return
        print(f"\n  ── {label} ──")
        skip = set()
        for k, v in s.items():
            if k in skip: continue
            bar = ""
            if k == "win_rate_%":
                filled = int(float(str(v))/5)
                bar = f"  {'█'*filled}{'░'*(20-filled)}"
            print(f"  {k:<28} {str(v):>12}{bar}")

    pstat(s_g, "GROSS (Before Costs)")
    pstat(s_n, "NET   (After All Costs + Progressive Tax)")

    if s_g.get("n_trades",0)>0 and s_n.get("n_trades",0)>0:
        drag = s_g["total_return_%"] - s_n["total_return_%"]
        print(f"""
  ── COST IMPACT ──
  Gross return     : {s_g['total_return_%']:>+.2f}%
  Net return       : {s_n['total_return_%']:>+.2f}%
  Cost drag        : {drag:>+.2f}%  (2 legs × fixed costs)
  Avg cost/trade   : {s_n['avg_cost_%']:.4f}% of leg value
  Total tax paid   : ₹{s_n['total_tax_₹']:>10,.0f}
  Total all costs  : ₹{s_n['total_costs_₹']:>10,.0f}

  ── MARKET NEUTRALITY ──
  Avg net market exposure : {s_n['avg_mkt_exposure_%']:.3f}%
  (should be close to 0 — confirms pairs trade is working)""")

    # Compare single vs pairs
    print(f"\n{'─'*66}")
    print("  COMPARISON: Single-Leg vs Pairs Trade")
    print(f"{'─'*66}")
    print(f"  {'Metric':<25} {'Single-Leg':>14} {'Pairs Trade':>14} {'Better?':>8}")
    print(f"  {'─'*62}")
    metrics = [
        ("Net Return %",    s_single.get("total_return_%",0), s_n.get("total_return_%",0), "higher"),
        ("Sharpe Ratio",    s_single.get("sharpe_ratio",0),   s_n.get("sharpe_ratio",0),   "higher"),
        ("Max Drawdown %",  s_single.get("max_drawdown_%",0), s_n.get("max_drawdown_%",0), "higher"),
        ("Win Rate %",      s_single.get("win_rate_%",0),     s_n.get("win_rate_%",0),      "higher"),
        ("Profit Factor",   s_single.get("profit_factor",0),  s_n.get("profit_factor",0),   "higher"),
        ("Avg Net PnL %",   s_single.get("avg_net_%",  s_single.get("avg_net_pnl_%",0)),
                            s_n.get("avg_net_%",0),    "higher"),
    ]
    for m, sv, pv, better in metrics:
        if better == "higher":
            winner = "Pairs ✅" if pv > sv else "Single ✅" if sv > pv else "Tie"
        else:
            winner = "Pairs ✅" if pv < sv else "Single ✅" if sv < pv else "Tie"
        print(f"  {m:<25} {sv:>14.2f} {pv:>14.2f} {winner:>8}")


# ════════════════════════════════════════════════════════════
#  7.  CHART
# ════════════════════════════════════════════════════════════
def plot_results(df, sig, t_g, t_n, t_single, s_g, s_n, s_single, args):
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(BG)
    gs_ = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.55, wspace=0.38,
                            top=0.91, bottom=0.05,
                            left=0.06, right=0.97)
    cap = args.capital

    fig.text(0.5, 0.965,
             "⚡  Gold–Silver Pairs Trade (Market-Neutral)  |  Long+Short Simultaneously",
             ha="center", fontsize=14, fontweight="bold", color=AMBER)
    fig.text(0.5, 0.938,
             f"Z≥{args.z_entry}σ  │  SpreadStop={args.spread_stop}%  │  Hold≤{args.hold}d  │  "
             f"Gross:{s_g.get('total_return_%','N/A')}%  │  "
             f"NET:{s_n.get('total_return_%','N/A')}%  │  "
             f"Sharpe:{s_n.get('sharpe_ratio','N/A')}  │  "
             f"MaxDD:{s_n.get('max_drawdown_%','N/A')}%  │  "
             f"Trades:{s_n.get('n_trades','N/A')}",
             ha="center", fontsize=8.5, color=GREY)

    # ── 1. Normalised prices ─────────────────────────────
    ax1 = fig.add_subplot(gs_[0, :2])
    ax1.plot(df.index, df["Gold"]/df["Gold"].iloc[0]*100, color=AMBER, lw=2, label="Gold")
    ax1.plot(df.index, df["Silver"]/df["Silver"].iloc[0]*100, color=SILVER, lw=2, label="Silver")
    if not t_n.empty:
        for _, r in t_n.iterrows():
            ed = pd.Timestamp(r["entry_date"])
            c  = SILVER if "SILVER" in r["direction"] else AMBER
            ax1.axvline(ed, color=c, alpha=0.3, lw=0.9)
    ax1.set_title("Normalised Prices (base=100)  |  Lines=trade entries")
    ax1.legend(fontsize=8); ax1.grid(True)

    # ── 2. GSR Z-score ─────────────────────────────────
    ax2 = fig.add_subplot(gs_[0, 2])
    ax2.plot(sig.index, sig["GSR_z"], color=PURPLE, lw=1.2, alpha=0.8)
    ax2.axhline(0, color=BORDER, lw=1)
    ax2.axhline( args.z_entry, color=SILVER, lw=1.2, ls="--", label=f"+{args.z_entry}σ Long Silv")
    ax2.axhline(-args.z_entry, color=AMBER,  lw=1.2, ls="--", label=f"-{args.z_entry}σ Long Gold")
    ax2.fill_between(sig.index, sig["GSR_z"], args.z_entry,
                     where=(sig["GSR_z"]>=args.z_entry)&sig["all_ok"], alpha=0.2, color=SILVER)
    ax2.fill_between(sig.index, sig["GSR_z"], -args.z_entry,
                     where=(sig["GSR_z"]<=-args.z_entry)&sig["all_ok"], alpha=0.2, color=AMBER)
    ax2.set_title("GSR Z-Score  (Signal Zones)")
    ax2.legend(fontsize=7); ax2.grid(True)

    # ── 3. Spread returns per trade ─────────────────────
    ax3 = fig.add_subplot(gs_[1, 0])
    if not t_n.empty:
        colors_t = [GREEN if v>0 else RED for v in t_n["net_pnl_%"]]
        ax3.bar(range(len(t_n)), t_n["net_pnl_%"], color=colors_t, edgecolor=BORDER, lw=0.5)
        ax3.axhline(0, color=BORDER, lw=1)
        ax3.axhline(t_n["net_pnl_%"].mean(), color=AMBER, lw=1.5, ls="--",
                    label=f"Mean {s_n['avg_net_%']}%")
        ax3.set_title(f"Net Spread P&L per Trade\nWR={s_n.get('win_rate_%')}%  PF={s_n.get('profit_factor')}")
        ax3.set_xlabel("Trade #"); ax3.set_ylabel("Net PnL %")
        ax3.legend(fontsize=7); ax3.grid(True, axis="y")

    # ── 4. Market exposure check ───────────────────────
    ax4 = fig.add_subplot(gs_[1, 1])
    if not t_n.empty:
        exp = t_n["mkt_exposure_%"]
        colors_e = [GREEN if abs(v)<2 else AMBER if abs(v)<5 else RED for v in exp]
        ax4.bar(range(len(exp)), exp, color=colors_e, edgecolor=BORDER, lw=0.5)
        ax4.axhline(0, color=WHITE, lw=1.2)
        ax4.axhline( 2, color=GREEN, lw=0.8, ls="--", alpha=0.5, label="±2% target zone")
        ax4.axhline(-2, color=GREEN, lw=0.8, ls="--", alpha=0.5)
        ax4.set_title(f"Market Exposure per Trade\n(avg={s_n.get('avg_mkt_exposure_%')}% — should be ~0)")
        ax4.set_xlabel("Trade #"); ax4.legend(fontsize=7); ax4.grid(True, axis="y")

    # ── 5. Single vs Pairs equity comparison ──────────
    ax5 = fig.add_subplot(gs_[1, 2])
    if not t_single.empty and not t_n.empty:
        eq_s = t_single["equity_after"] if "equity_after" in t_single.columns else t_single.get("equity", pd.Series())
        eq_p = t_n["equity_after"]
        ax5.plot(range(len(eq_s)), eq_s/cap*100, color=AMBER, lw=1.5, ls="--",
                 label=f"Single-Leg ({s_single.get('total_return_%',0):+.1f}%)")
        ax5.plot(range(len(eq_p)), eq_p/cap*100, color=GREEN, lw=2,
                 label=f"Pairs Trade ({s_n.get('total_return_%',0):+.1f}%)")
        ax5.axhline(100, color=BORDER, lw=1, ls="--")
        ax5.set_title("Single-Leg vs Pairs Trade\nEquity Comparison (base=100%)")
        ax5.legend(fontsize=7); ax5.grid(True)

    # ── 6. Net equity + drawdown ───────────────────────
    ax6 = fig.add_subplot(gs_[2, :2])
    if not t_n.empty:
        eq  = t_n["equity_after"]
        bd  = pd.to_datetime(t_n["entry_date"])
        ec  = GREEN if eq.iloc[-1] > cap else RED
        ax6.plot(bd, eq, color=ec, lw=2.5, marker="o", ms=5,
                 label="Pairs Net equity", zorder=3)
        if not t_single.empty:
            es = t_single["equity_after"] if "equity_after" in t_single.columns else None
            if es is not None:
                bs = pd.to_datetime(t_single.get("entry_date", t_single.index))
                ax6.plot(bs, es, color=AMBER, lw=1.5, ls="--", alpha=0.6,
                         label="Single-Leg (ref)")
        ax6.axhline(cap, color=BORDER, lw=1, ls="--", label=f"Start ₹{cap//1000}k")
        pk = eq.cummax()
        ax6.fill_between(bd, eq, pk, alpha=0.15, color=RED,
                         label=f"DD (max {s_n['max_drawdown_%']}%)")
        ax6.fill_between(bd, cap, eq, where=eq>=cap, alpha=0.08, color=GREEN)
        ax6.fill_between(bd, cap, eq, where=eq<cap,  alpha=0.08, color=RED)
        ax6.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x,_: f"₹{x/1000:.0f}k"))
        ax6.legend(fontsize=7)
    ax6.set_title(f"Pairs Trade Net Equity  "
                  f"Return:{s_n.get('total_return_%')}%  "
                  f"Sharpe:{s_n.get('sharpe_ratio')}  "
                  f"Calmar:{s_n.get('calmar_ratio')}")
    ax6.grid(True)

    # ── 7. PnL distribution ───────────────────────────
    ax7 = fig.add_subplot(gs_[2, 2])
    if not t_n.empty:
        pnl  = t_n["net_pnl_%"]
        rng  = max(pnl.max()-pnl.min(), 1)
        bins = np.arange(pnl.min()-0.5, pnl.max()+0.5, max(0.3, rng/15))
        n__, bins_, patches = ax7.hist(pnl, bins=bins, edgecolor=BORDER, lw=0.5)
        for patch, left in zip(patches, bins_[:-1]):
            patch.set_facecolor(GREEN if left >= 0 else RED)
        ax7.axvline(0, color=WHITE, lw=1.2)
        ax7.axvline(pnl.mean(), color=AMBER, lw=1.5, ls="--",
                    label=f"Mean {s_n['avg_net_%']}%")
        ax7.axvline(-args.spread_stop, color=RED, lw=1.2, ls=":",
                    label=f"SpreadStop -{args.spread_stop}%")
        ax7.set_title(f"Net Spread PnL Distribution\n"
                      f"WR={s_n.get('win_rate_%')}%  PF={s_n.get('profit_factor')}")
        ax7.legend(fontsize=7)
    ax7.grid(True, axis="y")

    plt.savefig("pairs_trade_results.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("📊  Chart saved → pairs_trade_results.png")
    plt.show()


# ════════════════════════════════════════════════════════════
#  8.  MAIN
# ════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="Gold–Silver Pairs Trade (Long+Short Simultaneously)")
    p.add_argument("--period",       default="10y",
                   choices=["1y","2y","3y","5y","10y"])
    p.add_argument("--window",       type=int,   default=60)
    p.add_argument("--z-entry",      type=float, default=1.5)
    p.add_argument("--z-exit",       type=float, default=0.3)
    p.add_argument("--spread-stop",  type=float, default=5.0,
                   help="Stop on SPREAD loss %% (default: 5.0) — tighter than single-leg")
    p.add_argument("--hold",         type=int,   default=20)
    p.add_argument("--min-corr",     type=float, default=0.5)
    p.add_argument("--max-vix",      type=float, default=50.0)
    p.add_argument("--capital",      type=float, default=500_000)
    p.add_argument("--no-plot",      action="store_true")
    args = p.parse_args()

    df  = fetch_data(period=args.period)
    sig = build_signals(df, window=args.window,
                        z_entry=args.z_entry, z_exit=args.z_exit,
                        min_corr=args.min_corr, max_vix=args.max_vix)

    # ── Run pairs backtest ─────────────────────────────
    t_g = run_pairs_backtest(df, window=args.window,
                             z_entry=args.z_entry, z_exit=args.z_exit,
                             spread_stop=args.spread_stop,
                             max_hold=args.hold, capital=args.capital,
                             apply_costs=False, min_corr=args.min_corr,
                             max_vix=args.max_vix)
    t_n = run_pairs_backtest(df, window=args.window,
                             z_entry=args.z_entry, z_exit=args.z_exit,
                             spread_stop=args.spread_stop,
                             max_hold=args.hold, capital=args.capital,
                             apply_costs=True, min_corr=args.min_corr,
                             max_vix=args.max_vix)
    s_g = compute_stats(t_g, args.capital)
    s_n = compute_stats(t_n, args.capital)

    # ── Run single-leg for comparison ──────────────────
    # Import from backtest.py
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        import backtest as bt
        df_bt = bt.fetch_data(period=args.period)
        t_single = bt.run_backtest(df_bt, window=args.window,
                                   z_entry=args.z_entry, z_exit=args.z_exit,
                                   price_stop=8.0, max_hold=args.hold,
                                   capital=args.capital, apply_costs=True,
                                   min_corr=args.min_corr, max_vix=args.max_vix)
        s_single = bt.compute_stats(t_single, args.capital)
        print("✅  Single-leg comparison loaded from backtest.py")
    except Exception as e:
        print(f"⚠️  Could not load single-leg for comparison: {e}")
        t_single = pd.DataFrame(); s_single = {}

    print_report(t_g, t_n, s_g, s_n, t_single, s_single, args)

    if not t_n.empty:
        t_n.to_csv("pairs_trades_log.csv", index=False)
        print("📄  Pairs trade log → pairs_trades_log.csv")

    if not args.no_plot:
        plot_results(df, sig, t_g, t_n, t_single, s_g, s_n, s_single, args)

    print(f"\n{'═'*66}")
    verdict = ("✅ PAIRS TRADE VIABLE" if s_n.get("total_return_%",0) > 0
               else "❌ Pairs trade underperforms — check spread_stop")
    print(f"  {verdict}")
    print(f"  Run OOS test: python oos_test.py")
    print("═"*66)

if __name__ == "__main__":
    main()
