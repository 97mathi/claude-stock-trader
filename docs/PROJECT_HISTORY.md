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

_Last updated: Round 12 — docs reorganisation and README rewrite._
