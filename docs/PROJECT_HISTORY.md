# Project History

This document records, in order, (A) the features and changes **requested**, and (B) the features **implemented** in response. It is a plain-English log so anyone can follow how the project grew.

---

## Section A — Features requested (chronological)

1. **A stock prediction & trading app for Indian stocks (NSE/BSE).**
   Two modes: paper trading (fake money) and real-money trading. Two core jobs: (a) buy a stock based on a future-price prediction; (b) monitor the live market against the prediction made at buy time and decide whether to keep holding or sell. Indian market, LSTM neural network, desktop app. Real trading deferred until paper proves reliable. Both swing and intraday horizons. Nifty 50 preloaded.

2. **Switch the model engine from TensorFlow to PyTorch.**
   TensorFlow refused to install on Python 3.13. PyTorch supports it.

3. **Add daily/weekly profit goals based on the chosen horizon.**
   A minimum and maximum profit per position. Sell at max (don't take extra risk); sell at min if a fresh prediction turns negative. Between min and max let a fresh prediction decide.

4. **Capital allocation rules.**
   Invest at most 75% of capital, never everything in one stock. Size positions in proportion to the predicted edge.

5. **Turn it into a fully autonomous agent.**
   No manual stock picking. Train on the whole universe at startup and keep models; retrain when predictions are frequently wrong. The agent decides what to trade and when. Alternative data: news, sentiment, sector indices, correlation, fundamentals, macro (RBI, inflation, GDP). Goal is per day / per week; stop trading once reached. Ledger + report.

6. **Implement auto-loop timing** (agent runs on a timer while the app is open).

7. **Architecture design diagram** — visual PNG showing all components, data flows, and working status auto-detected from source files.

8. **App changes:**
   - Auto-monitor always on while app is running (no toggle).
   - Clear Ledger button.
   - Negative P&L positions hold until recovery — only stop-loss can force a sell.
   - Formatted date & time in Ledger tab.
   - Remove manual "add stock to buy" from Predict tab.

9. **Comprehensive how-to HTML guide** (`docs/how_to_use.html`).

10. **LSTM walk-forward architecture search** — try 4 architectures across 2 expanding folds, pick the winner per stock/horizon. ~3–4× training time, provably better model.

11. **Parallel batch price fetch** — 50-stock price refresh from ~30 s sequential to ~3–5 s using ThreadPoolExecutor.

12. **Saved models list + clear** in Settings tab — scrollable list with directional accuracy, val loss, size, date. Clear all button. Settings tab wrapped in a scrollable frame.

13. **Fix retrain logic:**
    - Voluntary retrain (Predict tab): save new model only if walk-forward score beats the saved model.
    - Stale retrain (MAPE > 4%): always save — the existing model is already proven bad on live data.

14. **Add `.gitignore`** — exclude `model/trained/`, `__pycache__/`, `*.db`, exports. Remove 200+ binary/runtime files accidentally tracked.

15. **Move docs to `docs/` folder** — `how_to_use.html`, `PROJECT_HISTORY.md`, `design_diagram.png` moved out of root. README rewritten to reflect all changes.

---

## Section B — Features implemented (chronological)

### Round 1 — Core app
1. **Project structure & settings** (`config.py`, `nifty50.py`).
2. **Data fetcher** (`data/fetcher.py`) — Yahoo Finance, NSE `.NS` suffix.
3. **Technical indicators** (`model/features.py`) — RSI, MACD, MAs, Bollinger Bands (18 features).
4. **LSTM prediction** (`model/lstm_model.py`) — trains per stock, multi-step price path forecast.
5. **Paper trading engine** (`trading/portfolio.py`) — SQLite wallet (cash, positions, trade log).
6. **Monitor / decision engine** (`trading/monitor.py`) — HOLD / SELL per position.
7. **Desktop GUI** (`gui/app.py`, `gui/charts.py`) — dark CustomTkinter window with Dashboard, Predict & Buy, Portfolio & Monitor, Settings tabs.
8. **Real-broker placeholder** (`trading/broker_interface.py`) — disabled stub, same interface as Portfolio.
9. **Setup README**.

### Round 2 — PyTorch
10. Replaced TensorFlow with PyTorch in `model/lstm_model.py`. Updated `requirements.txt` and README.

### Round 3 — Profit goals
11. Per-horizon min/max profit goals in `config.py` (intraday 1%/3%, swing 3%/6%).
12. Goal-based sell logic in the monitor.
13. Dashboard "Period goals" cards and min/max target columns in the monitor table.

### Round 4 — Capital allocation
14. Allocation settings in `config.py` — 75% cap, 10% per-stock cap, edge-proportional sizing, 2% minimum.
15. Position sizer (`trading/sizer.py`).
16. 75% cap enforced inside `portfolio.buy()`.
17. GUI sizing display — auto-filled quantity, sizing breakdown, "Capital used" tile.

### Round 5 — Autonomous agent
18. Sector map (`data/sectors.py`) and extended fetcher (sector indices, fundamentals, news).
19. Signals layer (`signals/`) — sentiment, sector, fundamentals, macro, correlation, aggregator.
20. Trainer (`agent/trainer.py`) — bulk-trains all Nifty 50 models, retrains drifting ones.
21. Accuracy tracker (`agent/accuracy.py`) — logs predictions vs. actuals, flags models above MAPE threshold.
22. Autonomous agent (`agent/agent.py`) — scans, scores, ranks, goal circuit-breaker, buys/sells.
23. Ledger (`trading/ledger.py`) — trade log, performance stats, CSV and markdown export.
24. Portfolio extensions — set total capital, period-goal circuit-breaker tracking.
25. GUI rebuild — Agent tab and Ledger tab added; "Set total capital" control.

### Round 6 — Auto-loop timing
26. Auto-loop — agent runs every N minutes while the app is open; non-overlapping cycles; clean stop-on-close.

### Round 7 — Move & verify
27. Verified all 30 modules compile cleanly at `C:\my code\stock trader`.

### Round 8 — Architecture diagram
28. `docs/design_diagram.png` — matplotlib diagram auto-detecting component status from source files. 5 tiers, labeled arrows, green/grey status badges.

### Round 9 — App redesign
29. **Always-on monitor** — replaced toggle auto-loop with a continuous position monitor that starts 15 s after the app opens. Configurable interval; "Check now" for immediate trigger.
30. **Negative P&L hold** (`trading/monitor.py`) — step 1b: if `pnl_pct < 0`, return HOLD immediately. Only stop-loss can force a sell below breakeven.
31. **Clear Ledger** — `portfolio.clear_trades()`; red "Clear Ledger" button in Ledger tab.
32. **Formatted timestamps** in Ledger tab — `dd-Mon-YYYY HH:MM`.
33. **Removed manual buy** from Predict tab — custom symbol entry and Paper BUY button removed. Tab renamed "Predict".
34. **`docs/how_to_use.html`** — comprehensive dark-theme HTML guide covering all 6 tabs, signals, decision flow, position sizing, config reference, FAQ.

### Round 10 — Model upgrade
35. **Walk-forward architecture search** (`model/lstm_model.py`):
    - 4 candidates: small (32→16, dropout 0.1), medium (64→32, 0.2), large (128→64, 0.2), regularised (64→32, 0.3).
    - 2 expanding walk-forward folds per candidate.
    - Scoring: `0.4 × MAE + 0.6 × (1 − directional_accuracy)` — direction weighted 60%.
    - Final training uses the winning architecture.
    - `confidence` now = real walk-forward directional accuracy.
    - Backward-compatible: old `.pt` files load with default arch fallback.
36. **Parallel batch price fetch** (`data/fetcher.py`) — `ThreadPoolExecutor(max_workers=10)`, 50 stocks in ~3–5 s vs ~30 s.

### Round 11 — Saved models UI + retrain logic
37. **Saved Models section** in Settings tab — lists every `.pt` with symbol, horizon, dir accuracy, val loss, file size, training date; total count and disk usage.
38. **Settings tab scrollable** — `CTkScrollableFrame` so all sections are always reachable.
39. **Clear all models** button — deletes all `.pt` and `.json` from `model/trained/`.
40. **Save-only-if-better guard** — voluntary retrains compare walk-forward combined score; keep old model if new is not better.
41. **force_save for stale retrains** — `retrain_stale()` passes `force_save=True`; MAPE-flagged models always replaced.

### Round 12 — Repo hygiene + docs reorganisation
42. **`.gitignore`** — excludes `model/trained/`, `__pycache__/`, `*.db`, exports. Removed 200+ accidentally-tracked binary files.
43. **`docs/` folder** — `how_to_use.html`, `PROJECT_HISTORY.md`, `design_diagram.png` moved here. Root stays clean.
44. **README full rewrite** — reflects all rounds 8–12.

---

---

## Section A — Round 13 requests

16. **Single source of truth for stock universe** — price cache should import `NIFTY_50` directly from `nifty50.py` (same list as trainer). Update `requirements.txt` with clear sections.
17. **Live price without broker account** — design options; async NSE scraping cache chosen.
18. **Price cache implementation** — async NSE scraper, background thread, no Yahoo fallback for live prices.
19. **Explain buy criteria / fix negative-prediction buys** — `MIN_PREDICTION_EDGE` was defined but never checked; SBIN bought with negative LSTM because other signals overrode it.
20. **Low-risk consistent profit rules** — list all missing financial checkpoints for consistent low-risk returns.
21. **Implement all 5 risk rules** — trailing stop, Nifty trend filter, RSI gate, daily drawdown limit, stagnation exit.

---

## Section B — Round 13 implemented

### Round 13 — Risk rules + live feed
45. **Single source of truth** (`data/price_cache.py`) — imports `NIFTY_50` from `nifty50.py` directly; `gui/app.py` no longer passes the list. `requirements.txt` rewritten with sections and broker SDK notes.
46. **NSE async price cache** (`data/price_cache.py`) — `aiohttp` + `asyncio` background thread, 30 s refresh, instant cache reads for monitor/agent. No Yahoo fallback for live prices.
47. **LSTM edge hard gate** (`agent/agent.py`) — swing ≥ 1.5 %, intraday ≥ 1.0 % enforced before combined score. All three filter gates log rejection reasons to the GUI.
48. **Trailing stop-loss** (`trading/portfolio.py`, `trading/monitor.py`) — `highest_price_seen` column (auto-migrates existing DBs), `update_trailing_stop()` raises stop as price rises, `evaluate_all()` updates trail before each decision.
49. **Stagnation exit** (`trading/monitor.py`) — rule 2 in decision ladder; exits flat positions (abs P&L < 0.5 %) after half the horizon elapses.
50. **Nifty trend filter** (`agent/agent.py`, `data/fetcher.py`) — `get_nifty_trend()` checks `^NSEI` vs 20-day SMA; buy scan aborts when market is in downtrend.
51. **RSI overbought gate** (`agent/agent.py`, `data/fetcher.py`) — `get_current_rsi()` checked before LSTM call; RSI > 65 at scan time → skip.
52. **Daily drawdown circuit-breaker** (`agent/agent.py`) — today's P&L < −1.5 % of equity → pause all new buys for the day.
53. **Docs updated** — decision ladder in `how_to_use.html` extended to 6 steps; FAQ entries updated; `PROJECT_HISTORY.md` extended.

---

## Section A — Round 14 requests

22. **ETF / fund universe** — add a curated list of Indian ETFs as an alternative to individual Nifty 50 stocks; let users choose which universe to use from the GUI.
23. **Info icons on every setting** — each parameter row in the Settings tab should have a small ℹ button that pops up a detailed explanation (WHAT / WHY / RECOMMENDED / CAUTION).
24. **Fail-safes for required sequence** — warn or block if user tries to run buy scan before training, or switches universe without retraining; restart price cache on universe change.

---

## Section B — Round 14 implemented

### Round 14 — ETF universe + info icons + fail-safe sequence

54. **`funds.py`** (new file) — 23 curated Indian NSE ETFs in 5 risk tiers: Nifty 50 ETFs (NIFTYBEES, SETFNIF50, …), Nifty Next 50, Sectoral (BANKBEES, ITBEES, …), Midcap, Commodity/Gold. `ALL_FUNDS`, `RECOMMENDED_FUNDS`, and `FUND_CATEGORIES` dict exported.
55. **`config.py` universe config** — `ACTIVE_UNIVERSE` setting (`"nifty50"` / `"funds"` / `"both"`); `get_active_universe()` helper reads from `nifty50.py` or `funds.py` based on active setting; runtime changes reflected without restart.
56. **`agent/trainer.py`** — removed hardcoded `NIFTY_50` import; `untrained_symbols()` and `train_universe()` use `config.get_active_universe()` as default.
57. **`agent/agent.py`** — `self.universe` initialised and refreshed via `_cfg.get_active_universe()` each cycle; RSI gate, LSTM gate, Nifty trend, drawdown all operate on the active universe.
58. **`data/price_cache.py`** — `_default_universe()` function (lazy import of config) replaces module-level constant so universe changes at runtime are reflected in the next scrape cycle.
59. **GUI universe selector** (`gui/app.py`) — three radio buttons (Nifty 50 / ETFs / Both) in the Agent tab; ℹ Fund list button opens a categorized ETF popup; status label shows trained-model count for the active universe; `_on_universe_change()` warns when models are missing for new universe and restarts the price cache.
60. **GUI fail-safes** (`gui/app.py`) — three checks before the buy scan runs: (1) zero models → `showerror` with required sequence instructions; (2) > 30 % symbols untrained → `askyesno` caution; (3) price cache not running → `showwarning` (non-blocking).
61. **Settings ℹ info icons** (`gui/app.py`) — every parameter row in the Settings tab now has a compact ℹ button (column 2) that shows a `messagebox.showinfo` popup explaining WHAT the parameter does, WHY it matters, the RECOMMENDED range, and any CAUTION — covers 11 parameters.
62. **Docs updated** — `how_to_use.html`: required-sequence callout in first-time setup; universe selector table in Agent tab; ℹ button callout in Settings tab; new FAQ "Should I use Nifty 50 stocks or ETFs/funds?"; `PROJECT_HISTORY.md` extended to Round 14.

---

_Last updated: Round 14 — ETF universe selection, parameter info icons, fail-safe sequence checks._
