"""
╔══════════════════════════════════════════════════════════════════╗
║  KELLY CRITERION POSITION SIZING                                ║
║  Gold–Silver GSR Strategy — Dynamic Capital Allocation          ║
╠══════════════════════════════════════════════════════════════════╣
║  WHAT IS KELLY CRITERION?                                       ║
║                                                                  ║
║  Formula: f* = (p × b - q) / b                                 ║
║    p  = win rate (e.g. 0.632)                                   ║
║    q  = loss rate = 1 - p                                       ║
║    b  = avg win / avg loss ratio (payoff ratio)                 ║
║    f* = optimal fraction of capital to risk per trade           ║
║                                                                  ║
║  Example with this strategy:                                    ║
║    p = 0.632, avg win = 3.5%, avg loss = 2.1%                  ║
║    b = 3.5 / 2.1 = 1.667                                       ║
║    f* = (0.632 × 1.667 - 0.368) / 1.667 = 0.41 = 41%         ║
║                                                                  ║
║  THREE VARIANTS TESTED:                                          ║
║    Fixed 90%    : current approach (baseline)                   ║
║    Full Kelly   : f* from rolling 20-trade history              ║
║    Half Kelly   : f* / 2  (safer, recommended in practice)     ║
║    Dynamic Kelly: Half Kelly × Z-score strength multiplier      ║
║                                                                  ║
║  Z-SCORE SCALING:                                               ║
║    |Z| = 1.5σ (entry threshold) → 1.0x base allocation        ║
║    |Z| = 2.0σ                   → 1.33x base allocation        ║
║    |Z| = 2.5σ                   → 1.67x base allocation        ║
║    |Z| = 3.0σ+                  → 2.0x base allocation (cap)  ║
║    Stronger signal → bigger position                            ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python kelly_sizing.py
  python kelly_sizing.py --period 10y
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
#  1.  KELLY FUNCTIONS
# ════════════════════════════════════════════════════════════

def compute_kelly(trade_history: list, window: int = 20) -> float:
    """
    Rolling Kelly fraction from recent trade history.
    Uses last `window` trades to estimate edge.

    Returns: Kelly fraction (clipped 0.05 – 0.95)
    """
    if len(trade_history) < max(5, window // 2):
        return 0.30   # conservative default until enough history

    recent = trade_history[-window:]
    pnl    = [t["net_pnl_%"] for t in recent]
    wins   = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]

    if not wins or not losses:
        return 0.30

    p = len(wins)   / len(pnl)
    q = len(losses) / len(pnl)
    b = np.mean(wins) / abs(np.mean(losses))   # payoff ratio

    kelly = (p * b - q) / b
    return float(np.clip(kelly, 0.05, 0.95))


def z_scale(base_alloc: float, z_abs: float,
            z_entry: float, max_alloc: float = 0.95) -> float:
    """
    Scale allocation by signal strength.
    At z_entry → 1.0× base.
    At 2×z_entry → 2.0× base (capped at max_alloc).
    """
    strength   = z_abs / z_entry          # 1.0 at threshold, higher for stronger
    strength   = min(strength, 2.0)       # cap multiplier at 2×
    scaled     = base_alloc * strength
    return float(np.clip(scaled, 0.05, max_alloc))


# ════════════════════════════════════════════════════════════
#  2.  DATA + COSTS  (same as backtest.py)
# ════════════════════════════════════════════════════════════

def get_close(ticker: str, period: str) -> pd.Series:
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    c = raw["Close"]
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    return pd.Series(c.values.flatten(), index=pd.to_datetime(raw.index),
                     name=ticker, dtype=float)


def fetch_data(period: str = "10y") -> pd.DataFrame:
    print(f"\n📥  Fetching {period} data …")
    gold   = get_close("GC=F", period)
    silver = get_close("SI=F", period)
    vix    = get_close("^VIX", period)
    df = pd.concat([gold.rename("Gold"), silver.rename("Silver"),
                    vix.rename("VIX")], axis=1).dropna()
    print(f"✅  {len(df)} days  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def compute_progressive_tax(profit: float) -> float:
    if profit <= 0: return 0.0
    SLABS = [(400_000,0),(400_000,.05),(400_000,.10),(400_000,.15),
             (400_000,.20),(400_000,.25),(float("inf"),.30)]
    tax = 0; rem = profit
    for sz, r in SLABS:
        if rem <= 0: break
        t = min(rem, sz); tax += t*r; rem -= t
    return round(tax * 1.04, 2)


def calc_costs(tv: float, gp: float, annual: float = 0) -> dict:
    pre  = tv * (0.0002*2 + 0.0001 + 0.000026*2 + 0.00002
                 + 0.000001*2 + 0.0006*2 + 0.0004)
    pre += (tv*0.0002*2 + tv*0.000026*2) * 0.18
    pnl  = gp - pre
    tax  = max(0, compute_progressive_tax(annual+pnl)
               - compute_progressive_tax(annual)) if pnl > 0 else 0
    return {"net": round(gp-pre-tax, 2),
            "cost_pct": round((pre+tax)/tv*100, 4)}


def build_signals(df: pd.DataFrame, window=60, z_entry=1.5,
                  z_exit=0.3, min_corr=0.5, max_vix=50.0,
                  trend_ma=50) -> pd.DataFrame:
    d = df.copy()
    d["GSR"]       = d["Gold"] / d["Silver"]
    d["GSR_mean"]  = d["GSR"].rolling(window).mean()
    d["GSR_std"]   = d["GSR"].rolling(window).std()
    d["GSR_z"]     = (d["GSR"]-d["GSR_mean"]) / d["GSR_std"]
    rg             = d["Gold"].pct_change()
    rs             = d["Silver"].pct_change()
    d["roll_corr"] = rg.rolling(window).corr(rs)
    d["GSR_pct"]   = d["GSR"].expanding().rank(pct=True)*100
    d["corr_ok"]   = d["roll_corr"] >= min_corr
    d["vix_ok"]    = d["VIX"] <= max_vix
    d["all_ok"]    = d["corr_ok"] & d["vix_ok"]
    return d.dropna()


# ════════════════════════════════════════════════════════════
#  3.  UNIFIED BACKTEST WITH SIZING MODE
# ════════════════════════════════════════════════════════════

SIZING_MODES = ["fixed", "full_kelly", "half_kelly", "dynamic_kelly"]

def run_backtest(df: pd.DataFrame,
                 sizing: str    = "fixed",
                 fixed_alloc: float = 0.90,
                 kelly_window: int  = 20,
                 window: int    = 60,
                 z_entry: float = 1.5,
                 z_exit: float  = 0.3,
                 price_stop: float = 8.0,
                 max_hold: int  = 20,
                 capital: float = 500_000,
                 min_corr: float = 0.5,
                 max_vix: float  = 50.0,
                 trend_ma: int  = 50) -> pd.DataFrame:

    sig = build_signals(df, window=window, z_entry=z_entry,
                        z_exit=z_exit, min_corr=min_corr,
                        max_vix=max_vix, trend_ma=trend_ma)

    trades        = []
    history       = []    # running list for Kelly calc
    equity        = capital
    in_trade      = False
    entry_i       = None
    annual_profit = 0.0
    current_fy    = None

    for i in range(window + trend_ma, len(sig) - 1):
        z          = sig["GSR_z"].iloc[i]
        filters_ok = sig["all_ok"].iloc[i]
        dt         = sig.index[i]

        fy = dt.year if dt.month >= 4 else dt.year - 1
        if current_fy is None: current_fy = fy
        if fy != current_fy: annual_profit = 0.0; current_fy = fy

        # ── Entry ─────────────────────────────────────────
        if not in_trade and abs(z) >= z_entry and filters_ok:
            asset = "Silver" if z > 0 else "Gold"

            # ── Compute allocation based on sizing mode ───
            if sizing == "fixed":
                alloc_frac = fixed_alloc

            elif sizing == "full_kelly":
                k          = compute_kelly(history, kelly_window)
                alloc_frac = z_scale(k, abs(z), z_entry)

            elif sizing == "half_kelly":
                k          = compute_kelly(history, kelly_window) / 2
                alloc_frac = z_scale(k, abs(z), z_entry)

            elif sizing == "dynamic_kelly":
                k          = compute_kelly(history, kelly_window) / 2
                alloc_frac = z_scale(k, abs(z), z_entry)
                # Extra: boost by GSR percentile extremity
                gsr_ext    = abs(sig["GSR_pct"].iloc[i] - 50) / 50  # 0–1
                alloc_frac = min(alloc_frac * (1 + gsr_ext * 0.5), 0.95)

            alloc      = equity * alloc_frac
            entry_px   = sig[asset].iloc[i + 1]
            entry_z    = z
            entry_gsr  = sig["GSR"].iloc[i]
            entry_gsrp = sig["GSR_pct"].iloc[i]
            annual_snap= annual_profit
            in_trade   = True; entry_i = i
            continue

        # ── Exit ──────────────────────────────────────────
        if in_trade:
            dh  = i - entry_i
            cp  = sig[asset].iloc[i]
            cz  = sig["GSR_z"].iloc[i]
            chg = (cp - entry_px) / entry_px * 100

            z_rev    = abs(cz) <= z_exit
            stop_hit = chg <= -price_stop
            max_hit  = dh >= max_hold

            if z_rev or stop_hit or max_hit:
                reason = ("Z_REVERSION" if z_rev
                          else "PRICE_STOP" if stop_hit
                          else "MAX_HOLD")
                xp     = entry_px*(1-price_stop/100) if stop_hit else cp
                gp     = (xp-entry_px)/entry_px*100
                gr     = alloc*gp/100
                c      = calc_costs(alloc, gr, annual_snap)
                net_rs = c["net"]
                equity+= net_rs
                annual_profit += max(0.0, gr)

                rec = {"entry_date" : sig.index[entry_i+1].date(),
                       "exit_date"  : sig.index[i].date(),
                       "asset"      : asset,
                       "entry_z"    : round(entry_z,3),
                       "entry_gsr"  : round(entry_gsr,2),
                       "entry_gsrp" : round(entry_gsrp,1),
                       "alloc_pct"  : round(alloc_frac*100,1),
                       "alloc_₹"    : round(alloc,0),
                       "gross_%"    : round(gp,3),
                       "net_%"      : round(net_rs/alloc*100,3),
                       "net_₹"      : round(net_rs,2),
                       "cost_%"     : round(c["cost_pct"],4),
                       "days_held"  : dh,
                       "exit_reason": reason,
                       "equity"     : round(equity,2)}
                trades.append(rec)
                history.append(rec)
                in_trade = False; entry_i = None

    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════════════
#  4.  STATS
# ════════════════════════════════════════════════════════════

def stats(t: pd.DataFrame, capital: float = 500_000) -> dict:
    if t.empty or len(t) < 2:
        return {"n":0,"wr":0,"avg":0,"sharpe":0,"maxdd":0,"total":0,"final":capital,"pf":0}
    pnl  = t["net_%"]; eq = t["equity"]
    fin  = eq.iloc[-1]
    peak = eq.cummax(); dd = (eq-peak)/peak*100
    tot  = (fin-capital)/capital*100
    sh   = round((pnl.mean()-6.5/252)/pnl.std()*np.sqrt(252),2) if pnl.std()>0 else 0
    wins = pnl[pnl>0]; loss = pnl[pnl<0]
    pf   = round(wins.sum()/abs(loss.sum()),2) if not loss.empty and loss.sum()!=0 else 99
    return {"n":len(t),"wr":round((pnl>0).mean()*100,1),
            "avg":round(pnl.mean(),3),"sharpe":sh,
            "maxdd":round(dd.min(),2),"calmar":round(tot/abs(dd.min()),2) if dd.min()!=0 else 99,
            "total":round(tot,2),"final":round(fin,0),"pf":pf,
            "avg_alloc":round(t["alloc_pct"].mean(),1)}


# ════════════════════════════════════════════════════════════
#  5.  PRINT COMPARISON
# ════════════════════════════════════════════════════════════

SEP = "═"*66

def print_comparison(results: dict, capital: float):
    print(f"\n{SEP}")
    print("  KELLY CRITERION SIZING — COMPARISON vs FIXED 90%")
    print(SEP)

    labels = {
        "fixed"         : "Fixed 90%    (current)",
        "half_kelly"    : "Half Kelly   (recommended)",
        "full_kelly"    : "Full Kelly   (aggressive)",
        "dynamic_kelly" : "Dynamic Kelly (Half + Z-scale)",
    }

    print(f"\n  {'Method':<30} {'N':>4} {'AvgAlloc':>9} {'WR%':>6} "
          f"{'AvgNet%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'Return%':>9} {'PF':>6}")
    print(f"  {'─'*80}")

    best_return = max(r["total"] for r in results.values() if r["n"] > 0)
    for mode, s in results.items():
        if s["n"] == 0:
            print(f"  {labels[mode]:<30} {'no trades':>60}")
            continue
        star = " ←★" if s["total"] == best_return else ""
        print(f"  {labels[mode]:<30} {s['n']:>4} {s['avg_alloc']:>8.1f}% "
              f"{s['wr']:>5.1f}% {s['avg']:>+7.3f}% "
              f"{s['sharpe']:>7.2f} {s['maxdd']:>7.2f}% "
              f"{s['total']:>+8.2f}% {s['pf']:>5.2f}{star}")

    print(f"\n  ── WHAT KELLY DOES ──")
    f  = results["fixed"]
    hk = results["half_kelly"]
    dk = results["dynamic_kelly"]

    if hk["n"] > 0 and f["n"] > 0:
        ret_diff    = hk["total"] - f["total"]
        sharpe_diff = hk["sharpe"] - f["sharpe"]
        dd_diff     = hk["maxdd"]  - f["maxdd"]
        print(f"  Half Kelly vs Fixed 90%:")
        print(f"    Return     : {f['total']:>+.2f}% → {hk['total']:>+.2f}%  "
              f"({'▲' if ret_diff>0 else '▼'} {abs(ret_diff):.2f}%)")
        print(f"    Sharpe     : {f['sharpe']:>+.2f}  → {hk['sharpe']:>+.2f}  "
              f"({'▲' if sharpe_diff>0 else '▼'} {abs(sharpe_diff):.2f})")
        print(f"    Max DD     : {f['maxdd']:>+.2f}% → {hk['maxdd']:>+.2f}%  "
              f"({'▲ more DD' if dd_diff<0 else '▼ less DD'})")
        print(f"    Avg Alloc  : {f['avg_alloc']:.1f}%  → {hk['avg_alloc']:.1f}%")

    if dk["n"] > 0 and f["n"] > 0:
        print(f"\n  Dynamic Kelly vs Fixed 90%:")
        print(f"    Return : {f['total']:>+.2f}% → {dk['total']:>+.2f}%")
        print(f"    Sharpe : {f['sharpe']:>+.2f}  → {dk['sharpe']:>+.2f}")
        print(f"    Max DD : {f['maxdd']:>+.2f}% → {dk['maxdd']:>+.2f}%")

    print(f"\n  ── POSITION SIZE BY Z-SCORE (Dynamic Kelly) ──")
    # Show how allocation varies with Z-score
    sample_kelly = 0.35  # typical half-kelly
    print(f"  (Example: Kelly={sample_kelly*2:.0%} Full → {sample_kelly:.0%} Half-Kelly base)")
    print(f"\n  {'Z-Score':>10}  {'Base Alloc':>12}  {'Z-Scaled':>10}  {'GSR-Boosted':>12}")
    print(f"  {'─'*50}")
    for z in [1.5, 1.8, 2.0, 2.5, 3.0, 3.5]:
        base     = sample_kelly
        z_sc     = z_scale(base, z, 1.5)
        gsr_ext  = 0.7  # example: 85th pct → (85-50)/50 = 0.7
        boosted  = min(z_sc * (1 + gsr_ext*0.5), 0.95)
        print(f"  {z:>10.1f}σ  {base*100:>11.1f}%  {z_sc*100:>9.1f}%  {boosted*100:>11.1f}%")

    print(f"\n  ── VERDICT ──")
    best_mode  = max(results, key=lambda m: results[m]["total"])
    best_s     = results[best_mode]
    print(f"  Best sizing method  : {labels[best_mode]}")
    print(f"  Net return          : {best_s['total']:+.2f}%")
    print(f"  Sharpe ratio        : {best_s['sharpe']}")
    print(f"  Max drawdown        : {best_s['maxdd']}%")
    rec = ("✅ Use Dynamic Kelly — better risk-adjusted returns"
           if best_mode == "dynamic_kelly"
           else "✅ Use Half Kelly — most practical and well-tested"
           if best_mode == "half_kelly"
           else "⚠️  Fixed sizing wins here — Kelly not beneficial in this regime")
    print(f"  Recommendation      : {rec}")


# ════════════════════════════════════════════════════════════
#  6.  CHART
# ════════════════════════════════════════════════════════════

def plot_comparison(all_trades: dict, df: pd.DataFrame,
                    results: dict, capital: float, z_entry: float):

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.50, wspace=0.38,
                            top=0.91, bottom=0.06,
                            left=0.06, right=0.97)
    fig.text(0.5, 0.965,
             "⚡  Kelly Criterion Position Sizing  |  Gold–Silver GSR Strategy",
             ha="center", fontsize=14, fontweight="bold", color=AMBER)
    fig.text(0.5, 0.938,
             f"Fixed 90%  vs  Half-Kelly  vs  Full-Kelly  vs  Dynamic Kelly  "
             f"| 10-Year Real Data Comparison",
             ha="center", fontsize=9, color=GREY)

    colours = {"fixed": GREY, "half_kelly": AMBER,
               "full_kelly": RED, "dynamic_kelly": GREEN}
    labels  = {"fixed": "Fixed 90%", "half_kelly": "Half Kelly",
               "full_kelly": "Full Kelly", "dynamic_kelly": "Dynamic Kelly"}

    # ── 1. Equity curves ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for mode, t in all_trades.items():
        if t.empty: continue
        ax1.plot(pd.to_datetime(t["entry_date"]),
                 t["equity"] / capital * 100,
                 color=colours[mode], lw=2,
                 label=f"{labels[mode]} ({results[mode]['total']:+.1f}%)",
                 ls="--" if mode=="full_kelly" else "-")
    ax1.axhline(100, color=BORDER, lw=1, ls="--")
    ax1.set_title("Equity Curves — All Sizing Methods (base=100%)")
    ax1.set_ylabel("Equity %"); ax1.legend(fontsize=8); ax1.grid(True)

    # ── 2. Allocation per trade (Dynamic Kelly) ─────────
    ax2 = fig.add_subplot(gs[0, 2])
    t_dk = all_trades.get("dynamic_kelly", pd.DataFrame())
    t_fx = all_trades.get("fixed", pd.DataFrame())
    if not t_dk.empty:
        ax2.scatter(range(len(t_dk)), t_dk["alloc_pct"],
                    c=[GREEN if v>0 else RED for v in t_dk["net_%"]],
                    s=50, zorder=3, label="Dynamic Kelly alloc")
        ax2.axhline(90, color=GREY, lw=1.2, ls="--", label="Fixed 90%")
        ax2.axhline(t_dk["alloc_pct"].mean(), color=GREEN, lw=1, ls=":",
                    label=f"DK avg {t_dk['alloc_pct'].mean():.1f}%")
        ax2.set_ylim(0, 100)
        ax2.set_title("Dynamic Kelly: Allocation per Trade\n(Green=win, Red=loss)")
        ax2.set_xlabel("Trade #"); ax2.set_ylabel("Allocation %")
        ax2.legend(fontsize=7); ax2.grid(True)

    # ── 3. Z-score vs allocation scatter ────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if not t_dk.empty:
        sc = ax3.scatter(t_dk["entry_z"].abs(), t_dk["alloc_pct"],
                         c=t_dk["net_%"], cmap="RdYlGn",
                         s=60, edgecolors=BORDER, lw=0.5, vmin=-5, vmax=10)
        plt.colorbar(sc, ax=ax3, label="Net PnL %")
        ax3.axhline(90, color=GREY, lw=1, ls="--", alpha=0.6, label="Fixed 90%")
        z_range = np.linspace(z_entry, t_dk["entry_z"].abs().max()+0.2, 50)
        ax3.set_title("Z-Score vs Allocation (Dynamic Kelly)\n(Color = trade PnL)")
        ax3.set_xlabel("|Z-Score at Entry|"); ax3.set_ylabel("Allocation %")
        ax3.legend(fontsize=7); ax3.grid(True)

    # ── 4. Return comparison bar ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    modes  = ["fixed","half_kelly","full_kelly","dynamic_kelly"]
    rets   = [results[m]["total"] for m in modes]
    sharps = [results[m]["sharpe"] for m in modes]
    lbls   = ["Fixed\n90%","Half\nKelly","Full\nKelly","Dynamic\nKelly"]
    c_bars = [GREEN if r>0 else RED for r in rets]
    bars   = ax4.bar(lbls, rets, color=c_bars, edgecolor=BORDER, lw=0.5)
    ax4.axhline(0, color=BORDER, lw=1)
    for bar, v, sh in zip(bars, rets, sharps):
        ax4.text(bar.get_x()+bar.get_width()/2,
                 v + (2 if v>=0 else -5),
                 f"{v:+.1f}%\nSR:{sh}", ha="center", fontsize=8, color=WHITE)
    ax4.set_title("Net Return by Sizing Method\n(with Sharpe annotation)")
    ax4.set_ylabel("Net Return %"); ax4.grid(True, axis="y")

    # ── 5. Risk metrics radar / bar comparison ──────────
    ax5 = fig.add_subplot(gs[1, 2])
    metrics_labels = ["Win Rate %", "Profit Factor", "Sharpe", "Calmar"]
    x     = np.arange(len(metrics_labels))
    width = 0.2
    for j, mode in enumerate(["fixed","half_kelly","dynamic_kelly"]):
        s  = results[mode]
        if s["n"] == 0: continue
        vals = [s["wr"], s["pf"], s["sharpe"], s["calmar"]]
        ax5.bar(x + j*width, vals, width, label=labels[mode],
                color=colours[mode], edgecolor=BORDER, lw=0.5, alpha=0.85)
    ax5.set_xticks(x + width)
    ax5.set_xticklabels(metrics_labels, fontsize=8)
    ax5.set_title("Risk Metrics Comparison\n(higher is better for all)")
    ax5.legend(fontsize=7); ax5.grid(True, axis="y")

    plt.savefig("kelly_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("📊  Chart saved → kelly_comparison.png")
    plt.show()


# ════════════════════════════════════════════════════════════
#  7.  MAIN
# ════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Kelly Criterion Sizing Comparison — Gold-Silver GSR")
    p.add_argument("--period",       default="10y",
                   choices=["1y","2y","3y","5y","10y"])
    p.add_argument("--window",       type=int,   default=60)
    p.add_argument("--z-entry",      type=float, default=1.5)
    p.add_argument("--z-exit",       type=float, default=0.3)
    p.add_argument("--price-stop",   type=float, default=8.0)
    p.add_argument("--hold",         type=int,   default=20)
    p.add_argument("--kelly-window", type=int,   default=20,
                   help="Rolling window for Kelly estimation (default: 20)")
    p.add_argument("--min-corr",     type=float, default=0.5)
    p.add_argument("--max-vix",      type=float, default=50.0)
    p.add_argument("--capital",      type=float, default=500_000)
    p.add_argument("--no-plot",      action="store_true")
    args = p.parse_args()

    df = fetch_data(period=args.period)

    common = dict(window=args.window, z_entry=args.z_entry,
                  z_exit=args.z_exit, price_stop=args.price_stop,
                  max_hold=args.hold, capital=args.capital,
                  min_corr=args.min_corr, max_vix=args.max_vix,
                  kelly_window=args.kelly_window)

    all_trades = {}
    all_stats  = {}

    for mode in SIZING_MODES:
        print(f"  Running {mode} …", end=" ", flush=True)
        t = run_backtest(df, sizing=mode, **common)
        s = stats(t, args.capital)
        all_trades[mode] = t
        all_stats[mode]  = s
        print(f"done  → {s['total']:>+.2f}%  Sharpe {s['sharpe']}")

    print_comparison(all_stats, args.capital)

    # Save best result to CSV
    best_mode = max(all_stats, key=lambda m: all_stats[m]["total"])
    if not all_trades[best_mode].empty:
        all_trades[best_mode].to_csv("kelly_best_trades.csv", index=False)
        print(f"\n📄  Best trades log ({best_mode}) → kelly_best_trades.csv")

    if not args.no_plot:
        sig = {k: build_signals(df, window=args.window, z_entry=args.z_entry,
                                z_exit=args.z_exit, min_corr=args.min_corr,
                                max_vix=args.max_vix) for k in ["_"]}
        plot_comparison(all_trades, df, all_stats, args.capital, args.z_entry)

    # Update README with Kelly results
    print(f"\n{'═'*66}")
    print("  KELLY SIZING COMPLETE")
    print(f"  Best method : {best_mode}  → {all_stats[best_mode]['total']:+.2f}% net")
    print(f"  Run command : python kelly_sizing.py --period 10y")
    print("═"*66)


if __name__ == "__main__":
    main()
