"""
╔══════════════════════════════════════════════════════════════════╗
║  OUT-OF-SAMPLE VALIDATION + SHARPE INVESTIGATION               ║
║  Gold–Silver GSR Strategy                                       ║
╠══════════════════════════════════════════════════════════════════╣
║  METHOD:                                                        ║
║  Split 10 years into 3 windows:                                 ║
║    IN-SAMPLE  (IS) : 2016–2020  ← where params were chosen     ║
║    OUT-OF-SAMPLE 1 : 2021–2022  ← first blind test             ║
║    OUT-OF-SAMPLE 2 : 2023–2026  ← second blind test            ║
║                                                                  ║
║  SHARPE INVESTIGATION:                                           ║
║  Sharpe of 8.14 is very high. Check for:                        ║
║    1. Trade count bias (small N inflates Sharpe)                ║
║    2. Autocorrelation in returns                                 ║
║    3. Look-ahead in rolling Z-score construction                ║
║    4. Compare to random baseline                                 ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python oos_test.py
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Colour palette ──────────────────────────────────────────
BG, PANEL, BORDER = "#0d1117", "#161b22", "#21262d"
AMBER, BLUE       = "#f59e0b", "#60a5fa"
GREEN, RED, GREY  = "#4ade80", "#f87171", "#8b949e"
WHITE, PURPLE     = "#e6edf3", "#c084fc"

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor": PANEL,
    "axes.edgecolor":  BORDER,"axes.labelcolor": GREY,
    "axes.titlecolor": WHITE, "xtick.color": GREY,
    "ytick.color":     GREY,  "grid.color": BORDER,
    "grid.linestyle":  "--",  "grid.alpha": 0.5,
    "text.color":      WHITE, "legend.facecolor": PANEL,
    "legend.edgecolor":BORDER,"font.family": "monospace",
    "font.size": 9,
})

# ════════════════════════════════════════════════════════════
#  DATA
# ════════════════════════════════════════════════════════════
def get_close(ticker, period="10y"):
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    c = raw["Close"]
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    return pd.Series(c.values.flatten(), index=pd.to_datetime(raw.index),
                     name=ticker, dtype=float)

print("📥  Fetching 10y data for OOS validation …")
gold   = get_close("GC=F", "10y")
silver = get_close("SI=F", "10y")
vix    = get_close("^VIX", "10y")
df = pd.concat([gold.rename("Gold"), silver.rename("Silver"),
                vix.rename("VIX")], axis=1).dropna()
print(f"✅  {len(df)} days  ({df.index[0].date()} → {df.index[-1].date()})")

# ════════════════════════════════════════════════════════════
#  STRATEGY CORE  (fixed params — NO re-optimisation on OOS)
# ════════════════════════════════════════════════════════════
PARAMS = dict(window=60, z_entry=1.5, z_exit=0.3,
              price_stop=8.0, max_hold=20,
              min_corr=0.5, max_vix=50.0, trend_ma=50)

def compute_progressive_tax(profit):
    if profit <= 0: return 0.0
    SLABS = [(400_000,0),(400_000,.05),(400_000,.10),(400_000,.15),
             (400_000,.20),(400_000,.25),(float("inf"),.30)]
    tax = 0; rem = profit
    for sz, r in SLABS:
        if rem <= 0: break
        t = min(rem, sz); tax += t*r; rem -= t
    return tax * 1.04

def compute_costs(tv, gp, annual=0):
    pre = tv*(0.0002*2+0.0001+0.000026*2+0.00002+0.000001*2+0.0006*2+0.0004)
    pre += (tv*0.0002*2 + tv*0.000026*2)*0.18  # GST
    pnl = gp - pre
    tax = max(0, compute_progressive_tax(annual+pnl) - compute_progressive_tax(annual)) if pnl>0 else 0
    return {"net": round(gp-pre-tax,2), "cost_pct": round((pre+tax)/tv*100,4)}

def build_signals(data, p):
    d = data.copy()
    d["GSR"]       = d["Gold"]/d["Silver"]
    d["GSR_mean"]  = d["GSR"].rolling(p["window"]).mean()
    d["GSR_std"]   = d["GSR"].rolling(p["window"]).std()
    d["GSR_z"]     = (d["GSR"]-d["GSR_mean"])/d["GSR_std"]
    rg = d["Gold"].pct_change(); rs = d["Silver"].pct_change()
    d["roll_corr"] = rg.rolling(p["window"]).corr(rs)
    d["gold_ma"]   = d["Gold"].rolling(p["trend_ma"]).mean()
    d["silver_ma"] = d["Silver"].rolling(p["trend_ma"]).mean()
    d["GSR_pct"]   = d["GSR"].expanding().rank(pct=True)*100
    d["corr_ok"]   = d["roll_corr"] >= p["min_corr"]
    d["vix_ok"]    = d["VIX"] <= p["max_vix"]
    d["all_ok"]    = d["corr_ok"] & d["vix_ok"]
    return d.dropna()

def backtest(data, p, capital=500_000):
    sig = build_signals(data, p)
    trades=[]; equity=capital; in_trade=False; entry_i=None
    annual_profit=0; current_fy=None
    for i in range(p["window"]+p["trend_ma"], len(sig)-1):
        z = sig["GSR_z"].iloc[i]; ok = sig["all_ok"].iloc[i]
        dt = sig.index[i]
        fy = dt.year if dt.month>=4 else dt.year-1
        if current_fy is None: current_fy=fy
        if fy != current_fy: annual_profit=0; current_fy=fy
        if not in_trade and abs(z)>=p["z_entry"] and ok:
            asset     = "Silver" if z>0 else "Gold"
            ep        = sig[asset].iloc[i+1]
            entry_z   = z; alloc=equity*0.90
            annual_snap = annual_profit
            in_trade=True; entry_i=i; entry_px=ep; entry_asset=asset; entry_alloc=alloc
            continue
        if in_trade:
            dh = i-entry_i; cp=sig[entry_asset].iloc[i]; cz=sig["GSR_z"].iloc[i]
            chg=(cp-entry_px)/entry_px*100
            if abs(cz)<=p["z_exit"] or chg<=-p["price_stop"] or dh>=p["max_hold"]:
                reason = ("Z_REVERSION" if abs(cz)<=p["z_exit"]
                          else "PRICE_STOP" if chg<=-p["price_stop"] else "MAX_HOLD")
                xp = entry_px*(1-p["price_stop"]/100) if chg<=-p["price_stop"] else cp
                gp = (xp-entry_px)/entry_px*100
                gr = entry_alloc*gp/100
                c  = compute_costs(entry_alloc, gr, annual_snap)
                equity+=c["net"]; annual_profit+=max(0,gr)
                trades.append({"net_%":c["net"]/entry_alloc*100,
                                "gross_%":gp, "equity":equity,
                                "reason":reason, "days":dh, "z_entry":entry_z})
                in_trade=False; entry_i=None
    return pd.DataFrame(trades)

def calc_stats(t, capital=500_000):
    if t.empty or len(t)<2: return {}
    pnl=t["net_%"]; eq=t["equity"]; fin=eq.iloc[-1]
    peak=eq.cummax(); dd=(eq-peak)/peak*100
    tot=(fin-capital)/capital*100
    sh=round((pnl.mean()-6.5/252)/pnl.std()*np.sqrt(252),2) if pnl.std()>0 else 0
    wins=pnl[pnl>0]; loss=pnl[pnl<0]
    pf=round(wins.sum()/abs(loss.sum()),2) if not loss.empty and loss.sum()!=0 else 99
    return {"n":len(t),"wr":round((pnl>0).mean()*100,1),
            "avg":round(pnl.mean(),3),"sharpe":sh,
            "maxdd":round(dd.min(),2),"pf":pf,
            "total":round(tot,2),"final":round(fin,0)}

# ════════════════════════════════════════════════════════════
#  1.  THREE-WAY IS / OOS SPLIT
# ════════════════════════════════════════════════════════════
print("\n" + "═"*66)
print("  OUT-OF-SAMPLE VALIDATION")
print("  Parameters FIXED at: z=1.5, corr≥0.5, VIX≤50, hold=20")
print("═"*66)

splits = {
    "IN-SAMPLE  (2016–2020)": (None, "2021-01-01"),
    "OUT-OF-SAMPLE 1 (2021–2022)": ("2021-01-01","2023-01-01"),
    "OUT-OF-SAMPLE 2 (2023–2026)": ("2023-01-01", None),
    "FULL PERIOD (all data)": (None, None),
}

results = {}
print(f"\n  {'Period':<30} {'N':>4} {'WR%':>6} {'Avg%':>7} {'Sharpe':>8} {'MaxDD%':>8} {'Total%':>9}")
print(f"  {'─'*65}")
for label, (start, end) in splits.items():
    sub = df.copy()
    if start: sub = sub[sub.index >= start]
    if end:   sub = sub[sub.index < end]
    if len(sub) < 200:
        print(f"  {label:<30} {'insufficient data':>50}")
        continue
    t = backtest(sub, PARAMS)
    s = calc_stats(t)
    results[label] = (t, s)
    if not s:
        print(f"  {label:<30} {'no trades':>50}")
        continue
    flag = ("✅" if s["sharpe"]>1 and s["total"]>0
            else "⚠️" if s["total"]>0
            else "❌")
    print(f"  {label:<30} {s['n']:>4} {s['wr']:>5.1f}% {s['avg']:>+6.3f}% "
          f"{s['sharpe']:>7.2f} {s['maxdd']:>7.2f}% {s['total']:>+8.2f}%  {flag}")

# ════════════════════════════════════════════════════════════
#  2.  SHARPE INVESTIGATION
# ════════════════════════════════════════════════════════════
print(f"\n{'═'*66}")
print("  SHARPE RATIO INVESTIGATION")
print("═"*66)

t_full = results.get("FULL PERIOD (all data)", (pd.DataFrame(), {}))[0]
if not t_full.empty:
    pnl = t_full["net_%"]
    n   = len(pnl)

    print(f"\n  ── Trade Count Effect ──")
    print(f"  N trades            : {n}")
    print(f"  Sharpe (annualised) : {calc_stats(t_full)['sharpe']}")
    print(f"  Standard Error      : {round(1/np.sqrt(n),3)}")
    print(f"  95% CI              : [{round(calc_stats(t_full)['sharpe'] - 2/np.sqrt(n),2)}, "
          f"{round(calc_stats(t_full)['sharpe'] + 2/np.sqrt(n),2)}]")
    print(f"  → With only {n} trades, Sharpe CI is WIDE. Need 100+ trades to be confident.")

    print(f"\n  ── Autocorrelation Check ──")
    acf1 = pnl.autocorr(1)
    acf2 = pnl.autocorr(2)
    print(f"  Lag-1 autocorr : {acf1:.3f}  (>0.2 would inflate Sharpe)")
    print(f"  Lag-2 autocorr : {acf2:.3f}")
    if abs(acf1) > 0.2:
        adj_sharpe = calc_stats(t_full)['sharpe'] * np.sqrt((1-acf1)/(1+acf1))
        print(f"  ⚠️  Autocorrelation detected → Adjusted Sharpe: {adj_sharpe:.2f}")
    else:
        print(f"  ✅ No significant autocorrelation — Sharpe not inflated by this")

    print(f"\n  ── Look-Ahead Bias Check ──")
    print(f"  Z-score uses rolling({PARAMS['window']}d) mean & std")
    print(f"  Entry on NEXT day close after signal (i+1) ✅")
    print(f"  Exit checked at current day close (i) ✅")
    print(f"  No forward-looking data in signal construction ✅")
    print(f"  → Look-ahead bias appears absent from code structure")

    print(f"\n  ── Random Baseline Comparison ──")
    np.random.seed(42)
    n_sims = 5000
    rand_sharpes = []
    for _ in range(n_sims):
        rand_ret = np.random.choice(
            df["Gold"].pct_change().dropna().values, n, replace=True) * 100
        sh = (rand_ret.mean()-6.5/252)/rand_ret.std()*np.sqrt(252) if rand_ret.std()>0 else 0
        rand_sharpes.append(sh)
    p_val = (np.array(rand_sharpes) >= calc_stats(t_full)['sharpe']).mean()
    print(f"  Strategy Sharpe        : {calc_stats(t_full)['sharpe']}")
    print(f"  Random baseline (5k sim) 95th pctile : {np.percentile(rand_sharpes,95):.2f}")
    print(f"  p-value vs random      : {p_val:.4f}")
    print(f"  {'✅ Sharpe is statistically significant vs random' if p_val<0.05 else '⚠️  Cannot reject random hypothesis'}")

    print(f"\n  ── Honest Assessment ──")
    s = calc_stats(t_full)
    if s['n'] < 50:
        print(f"  ⚠️  Only {s['n']} trades over full period — sample too small")
        print(f"     Sharpe of {s['sharpe']} is LIKELY overstated due to small N")
        print(f"     Need 3-5 more years of live data to confirm edge")
        print(f"     Treat current Sharpe as upper bound, not true estimate")
    if s['sharpe'] > 3:
        print(f"  ⚠️  Sharpe > 3 is rare for real strategies after costs")
        print(f"     Possible explanations:")
        print(f"       • Strategy genuinely exceptional in 2021-2026 gold bull run")
        print(f"       • Regime-specific performance (not repeatable)")
        print(f"       • Small sample variance ({s['n']} trades)")

# ════════════════════════════════════════════════════════════
#  3.  ROLLING ANNUAL PERFORMANCE
# ════════════════════════════════════════════════════════════
print(f"\n{'═'*66}")
print("  ROLLING ANNUAL PERFORMANCE (Year-by-Year)")
print("═"*66)
print(f"\n  {'Year':>6} {'Trades':>8} {'Win%':>6} {'Avg%':>8} {'Sharpe':>8} {'Return%':>9}")
print(f"  {'─'*50}")
for yr in range(df.index[0].year, df.index[-1].year+1):
    sub = df[df.index.year == yr]
    if len(sub) < 100: continue
    t = backtest(sub, PARAMS)
    s = calc_stats(t, capital=500_000)
    if not s or s['n']==0:
        print(f"  {yr:>6} {'0':>8} {'—':>6} {'—':>8} {'—':>8} {'—':>9}")
        continue
    flag = "✅" if s["total"]>0 else "❌"
    print(f"  {yr:>6} {s['n']:>8} {s['wr']:>5.1f}% {s['avg']:>+7.3f}% "
          f"{s['sharpe']:>7.2f} {s['total']:>+8.2f}%  {flag}")

# ════════════════════════════════════════════════════════════
#  4.  CHART
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                        top=0.90, bottom=0.07, left=0.07, right=0.97)
fig.text(0.5, 0.95,
         "⚡  Gold–Silver GSR  |  Out-of-Sample Validation + Sharpe Investigation",
         ha="center", fontsize=13, fontweight="bold", color=AMBER)

# Panel 1: IS vs OOS equity curves
ax1 = fig.add_subplot(gs[0, :2])
colors_map = {"IN-SAMPLE  (2016–2020)": AMBER,
              "OUT-OF-SAMPLE 1 (2021–2022)": GREEN,
              "OUT-OF-SAMPLE 2 (2023–2026)": BLUE}
for label, col in colors_map.items():
    if label in results and not results[label][0].empty:
        t, s = results[label]
        eq = t["equity"]
        ax1.plot(range(len(eq)), eq/500_000*100, color=col, lw=2, label=f"{label.split('(')[0].strip()}")
ax1.axhline(100, color=BORDER, lw=1, ls="--")
ax1.set_title("Equity Curves: IS vs OOS (base=100%)")
ax1.set_ylabel("Equity %")
ax1.legend(fontsize=7); ax1.grid(True)

# Panel 2: Sharpe by period
ax2 = fig.add_subplot(gs[0, 2])
periods = []; sharpes = []; colors_bar = []
for label in ["IN-SAMPLE  (2016–2020)",
              "OUT-OF-SAMPLE 1 (2021–2022)",
              "OUT-OF-SAMPLE 2 (2023–2026)"]:
    if label in results and results[label][1]:
        sh = results[label][1]["sharpe"]
        lbl= label.split("(")[1].replace(")","")
        periods.append(lbl); sharpes.append(sh)
        colors_bar.append(GREEN if sh > 1 else RED if sh < 0 else AMBER)
if periods:
    bars = ax2.bar(periods, sharpes, color=colors_bar, edgecolor=BORDER)
    ax2.axhline(0, color=BORDER, lw=1)
    ax2.axhline(1, color=GREEN, lw=1, ls="--", alpha=0.5, label="Min acceptable (1.0)")
    for bar, v in zip(bars, sharpes):
        ax2.text(bar.get_x()+bar.get_width()/2, v+(0.1 if v>=0 else -0.3),
                 f"{v:.2f}", ha="center", fontsize=9, color=WHITE)
ax2.set_title("Sharpe by Period\n(consistent = not overfit)")
ax2.legend(fontsize=7); ax2.grid(True, axis="y")

# Panel 3: Return distribution
ax3 = fig.add_subplot(gs[1, 0])
if not t_full.empty:
    pnl = t_full["net_%"]
    bins = np.arange(pnl.min()-0.5, pnl.max()+0.5, max(0.5, (pnl.max()-pnl.min())/15))
    n_, bins_, patches = ax3.hist(pnl, bins=bins, edgecolor=BORDER, lw=0.5)
    for patch, left in zip(patches, bins_[:-1]):
        patch.set_facecolor(GREEN if left >= 0 else RED)
    ax3.axvline(pnl.mean(), color=AMBER, lw=1.5, ls="--",
                label=f"Mean {pnl.mean():.2f}%")
    ax3.axvline(0, color=WHITE, lw=1)
    ax3.set_title(f"Trade PnL Distribution\nN={len(pnl)}, WR={round((pnl>0).mean()*100)}%")
    ax3.legend(fontsize=7); ax3.grid(True, axis="y")

# Panel 4: Rolling Sharpe (12-trade window)
ax4 = fig.add_subplot(gs[1, 1])
if not t_full.empty and len(t_full) >= 12:
    pnl = t_full["net_%"]
    roll_sh = [round((pnl.iloc[i-12:i].mean()-6.5/252)/pnl.iloc[i-12:i].std()*np.sqrt(252),2)
               if pnl.iloc[i-12:i].std()>0 else 0 for i in range(12, len(pnl)+1)]
    ax4.plot(range(len(roll_sh)), roll_sh, color=PURPLE, lw=1.5, label="Rolling 12-trade Sharpe")
    ax4.axhline(0, color=BORDER, lw=1)
    ax4.axhline(1, color=GREEN, lw=0.8, ls="--", alpha=0.6)
    ax4.fill_between(range(len(roll_sh)), roll_sh, 0,
                     where=np.array(roll_sh)>0, alpha=0.1, color=GREEN)
    ax4.fill_between(range(len(roll_sh)), roll_sh, 0,
                     where=np.array(roll_sh)<0, alpha=0.1, color=RED)
    ax4.set_title("Rolling 12-Trade Sharpe\n(stability of edge over time)")
    ax4.legend(fontsize=7); ax4.grid(True)

# Panel 5: Year-by-year returns
ax5 = fig.add_subplot(gs[1, 2])
yr_rets = []; yr_labels = []
for yr in range(df.index[0].year, df.index[-1].year+1):
    sub = df[df.index.year == yr]
    if len(sub) < 100: continue
    t = backtest(sub, PARAMS)
    s = calc_stats(t)
    if s and s['n']>0:
        yr_rets.append(s['total'])
        yr_labels.append(str(yr))
if yr_rets:
    colors_yr = [GREEN if r > 0 else RED for r in yr_rets]
    bars = ax5.bar(yr_labels, yr_rets, color=colors_yr, edgecolor=BORDER)
    ax5.axhline(0, color=BORDER, lw=1)
    for bar, v in zip(bars, yr_rets):
        ax5.text(bar.get_x()+bar.get_width()/2, v+(1 if v>=0 else -3),
                 f"{v:+.0f}%", ha="center", fontsize=8, color=WHITE)
    ax5.set_title("Year-by-Year Net Returns\n(Green = profitable year)")
    ax5.grid(True, axis="y")

plt.savefig("oos_validation.png", dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n📊  Chart saved → oos_validation.png")
plt.show()

print(f"\n{'═'*66}")
print("  OOS VALIDATION COMPLETE")
print("═"*66)
