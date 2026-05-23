# ⚡ Gold–Silver Ratio (GSR) Mean-Reversion | MCX Futures Backtest

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![yfinance](https://img.shields.io/badge/Data-yfinance-orange?style=flat-square)
![MCX](https://img.shields.io/badge/Exchange-MCX_Futures-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> Trading the Gold/Silver Ratio (GSR) on MCX futures — one of the oldest and most reliable mean-reverting spreads in commodity markets.

---

## 🧠 Why Gold–Silver (Not Gold–Crude)

| Pair | Correlation | Mean-Reverts? | Verdict |
|------|-------------|---------------|---------|
| Gold–Crude Oil | 0.15–0.45 | Unreliable | ❌ Tested, failed |
| **Gold–Silver** | **0.85–0.95** | **Yes, reliably** | **✅ This strategy** |

Gold and Silver share the same macro drivers: real interest rates, USD strength, inflation, and global risk. Silver's additional industrial demand (solar panels, EVs) creates *temporary* divergences — these are what we trade.

---

## 📊 The Gold/Silver Ratio (GSR)

```
GSR = Gold Price / Silver Price

Historical range : ~50x to ~124x
Long-run mean    : ~65–70x
COVID 2020 peak  : 124x (then reverted to 65x by 2021)

GSR > mean + 2σ  →  Silver historically cheap  →  BUY SILVER
GSR < mean - 2σ  →  Gold historically cheap    →  BUY GOLD
```

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt

# Recommended — 5 year backtest, both directions
python backtest.py --period 5y

# Full decade stress test
python backtest.py --period 10y

# Silver only (historically stronger leg)
python backtest.py --period 10y --silver-only

# Higher quality signals only
python backtest.py --period 10y --z-entry 2.5 --hold 20
```

---

## ⚙️ Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--period` | `5y` | `1y 2y 3y 5y 10y` |
| `--window` | `60` | Rolling Z-score window (days) |
| `--z-entry` | `2.0` | GSR Z-score to trigger signal |
| `--z-exit` | `0.3` | Z-score to exit (mean reversion) |
| `--price-stop` | `8.0` | Hard stop % below entry price |
| `--hold` | `20` | Max days to hold |
| `--min-corr` | `0.6` | Min rolling correlation to trade |
| `--max-vix` | `40.0` | Max VIX (higher than gold-crude — precious metals behave differently) |
| `--silver-only` | off | Only take BUY_SILVER signals |
| `--gold-only` | off | Only take BUY_GOLD signals |
| `--capital` | `500000` | Starting capital ₹ |

---

## 💰 MCX Futures Cost Model (FY 2025-26)

| Cost | Rate | Notes |
|------|------|-------|
| Brokerage | 0.02% per side | Discount broker |
| STT | 0.01% sell-side only | Futures: much lower than equity |
| Exchange | 0.0026% per side | MCX |
| GST | 18% on brokerage + exchange | |
| SEBI | 0.0001% per side | |
| Stamp Duty | 0.002% buy-side | |
| Slippage | 0.06% per side | Precious metals: very liquid |
| B-A Spread | 0.04% round-trip | Tighter than crude |
| **Income Tax** | **30% on net profit** | F&O = business income |
| **Total friction** | **≈ 0.28–0.40%** | vs 1.3% for NSE equity |

---

## 📈 Strategy Logic

```
Step 1 — COMPUTE
  GSR = Gold / Silver daily
  Rolling 60-day Z-score of GSR

Step 2 — FILTER (all must pass)
  Rolling 60d correlation ≥ 0.60  (pair must be correlated)
  VIX ≤ 40                         (no extreme panic)

Step 3 — SIGNAL
  GSR Z > +2.0σ  →  Silver cheap  →  BUY SILVER (laggard)
  GSR Z < -2.0σ  →  Gold cheap    →  BUY GOLD   (laggard)

Step 4 — EXIT (first to trigger)
  ① |GSR Z| reverts to ≤ 0.3σ     ← primary (mean reversion)
  ② Price falls 8% from entry      ← hard stop
  ③ Max 20 days held               ← time stop
```

---

## 📊 Chart Panels (6-panel figure)

1. **Normalised Prices** — Gold vs Silver (base 100), trade entries marked
2. **GSR Raw** — Ratio with rolling mean, entry signals marked
3. **GSR Z-Score** — Signal zones (bright = tradeable, dim = filtered)
4. **Rolling Correlation** — Stability over time, filter threshold shown
5. **Net Equity + Drawdown** — Green dot=Z-reversion, Red=stop, Grey=maxhold
6. **PnL Distribution** — Histogram with Silver vs Gold averages separate

---

## ⚠️ Key Limitations

- COMEX proxies (GC=F, SI=F) used — MCX prices include INR/USD and import duties
- MCX Gold lot = 1 kg (~₹8L), Silver lot = 30 kg (~₹2.5L) — position sizing matters
- 30% income tax on all F&O profits is the biggest cost drag
- Small trade count over 5 years — results are directionally indicative

---

## 🔬 Lessons From Failed Strategies

This strategy was built after:
1. **Copper → HINDCOPPER** failed: NSE equity costs (1.3%/trade) too high
2. **Gold–Crude divergence** failed: correlation too weak (r=0.25), 2020 wiped account

Gold–Silver fixes both problems: correlation is 5x stronger, futures costs 3x lower.

---
## 🙋 Author

**Parth Sarthi Saxena**
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/parthsarthisaxena)
- GitHub: [github.com/yourusername](https://github.com/parthsarthisaxena)

*⭐ Star the repo if this helped your research!*
