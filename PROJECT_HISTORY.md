# Project History

This document records, in order, (A) the features and changes **you requested**, and (B) the features **I implemented** in response. It is a plain-English log so anyone can follow how the project grew.

---

## Section A - Features you requested (in chronological order)

1. **A stock prediction & trading app for Indian stocks (NSE/BSE).**
   - Two modes: **paper trading** (practice with fake money to monitor efficiency, reduce risk, build consistency) and **real-money trading**.
   - Two core jobs: (a) **buy** a stock based on a future-price prediction; (b) **monitor** the live market against the prediction made at buy time and decide whether to keep holding or sell.
   - Your setup choices: Indian market, LSTM (a neural network for sequences), desktop app, comfortable coding but new to trading, real trading deferred until paper proves reliable, both swing and intraday horizons, Nifty 50 preloaded plus custom symbols.

2. **Switch the model engine from TensorFlow to PyTorch.**
   - Reason: TensorFlow refused to install on your Python version (3.13). PyTorch supports it.

3. **Add daily/weekly profit goals based on the chosen horizon.**
   - A **minimum** and **maximum** profit per position.
   - Sell when either the min or the max is reached (don't take extra risk for more profit; don't give back profit already earned).
   - If profit is between min and max, decide by running a fresh prediction.

4. **Capital allocation rules + a data question.**
   - Invest at most **75% of capital**, never everything in one stock.
   - Split money across stocks in proportion to the profit expected from the prediction.
   - You also asked: which data is used for prediction - historic or live?

5. **Turn it into a fully autonomous agent.**
   - No manual stock picking: train on the whole universe at startup and keep the models; retrain when predictions are frequently wrong.
   - The agent decides what to trade and when to buy/sell, and places the trades itself.
   - Add alternative data: news headlines, sentiment, sector indices, correlation with other stocks, earnings reports, P/E ratios, fundamentals, RBI rate decisions, inflation, GDP.
   - Goal is per day (intraday) or per week (swing); once reached, stop trading until the period resets.
   - Option to set the total amount available to invest.
   - A detailed ledger / report of all trades.
   - Your setup choices: Nifty 50 universe, modular data providers with safe defaults.

6. **Question: how much freedom does the agent have / how are decisions made.** (Explained; no code change.)

7. **Implement the chosen run mode** (auto-loop while the app is open), after I recommended it.

8. **Confirm everything is verified, and point the project at the new folder path.**

---

## Section B - Features I implemented (in the order I built them)

### Round 1 - The core app
1. **Project structure & settings** (`config.py`, `nifty50.py`) - one place for every adjustable number; the Nifty 50 symbol list.
2. **Data fetcher** (`data/fetcher.py`) - pulls historic and live prices from Yahoo Finance (free, supports NSE via the `.NS` suffix).
3. **Technical indicators** (`model/features.py`) - RSI, MACD, moving averages, Bollinger Bands added as model inputs.
4. **LSTM prediction** (`model/lstm_model.py`) - trains per stock, forecasts a multi-step price path.
5. **Paper trading engine** (`trading/portfolio.py`) - a simulated wallet in a local SQLite database (cash, positions, trade log).
6. **Monitor / decision engine** (`trading/monitor.py`) - compares live price to the prediction and flags HOLD or SELL.
7. **Desktop GUI** (`gui/app.py`, `gui/charts.py`) - dark-themed CustomTkinter window with Dashboard, Predict & Buy, Portfolio & Monitor, and Settings tabs, plus price/forecast charts.
8. **Real-broker placeholder** (`trading/broker_interface.py`) - a disabled stub with the same interface, ready for Zerodha/Upstox later.
9. **Setup README** - install and usage in plain English.

### Round 2 - PyTorch
10. **Replaced TensorFlow with PyTorch** in `model/lstm_model.py` (same model behavior, same interface); updated `requirements.txt` and README.

### Round 3 - Profit goals
11. **Per-horizon min/max profit goals** in `config.py` (intraday 1%/3%, swing 3%/6%).
12. **Goal-based sell logic** in the monitor: sell at max (greed cap), sell at stop-loss (loss cap), and between min and max let a fresh prediction decide.
13. **Dashboard "Period goals"** cards (Today / This week) and min/max target columns in the monitor table.

### Round 4 - Capital allocation
14. **Allocation settings** in `config.py` - 75% invested cap, 10% per-stock cap, edge-proportional scaling, 2% minimum trade.
15. **Position sizer** (`trading/sizer.py`) - turns expected profit + equity into a suggested quantity within all caps.
16. **75% cap enforced** inside `portfolio.buy()` so no trade can breach the reserve.
17. **GUI sizing display** - auto-filled quantity, sizing breakdown, and a "Capital used" tile.
18. **Answered the data question** in the README (full historic vs live inventory).

### Round 5 - The autonomous agent
19. **Sector map** (`data/sectors.py`) and extended fetcher (sector indices, fundamentals, news).
20. **Signals layer** (`signals/`) - sentiment, sector momentum, fundamentals, macro, correlation, and an aggregator that blends them with the LSTM forecast into one score.
21. **Trainer** (`agent/trainer.py`) - bulk-trains all Nifty 50 models once, retrains only drifting ones.
22. **Accuracy tracker** (`agent/accuracy.py`) - logs each prediction, compares to the real price later, flags models above the error threshold.
23. **Autonomous agent** (`agent/agent.py`) - scans, scores, ranks, applies the goal circuit-breaker, and buys/sells on its own.
24. **Ledger & report** (`trading/ledger.py`) - full trade log, performance stats, CSV and markdown export.
25. **Portfolio extensions** - set total capital, and period-goal circuit-breaker tracking.
26. **GUI rebuild** - new Agent tab (Train all, Run cycle, reports) and Ledger tab, plus a "Set total capital" control.

### Round 6 - Timing autonomy
27. **Auto-loop** - the agent runs a cycle automatically every N minutes while the app is open, with non-overlapping cycles and clean stop-on-close.

### Round 7 - Move & verify
28. **Verified all 30 modules compile cleanly** at the new project path `C:\my code\stock trader`.

---

_Last updated after moving the project to `C:\my code\stock trader` and confirming a clean full-project compile._
