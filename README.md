# Stock Predictor & Paper Trader (India)

A desktop app that:
1. Predicts future prices of Indian (NSE/BSE) stocks using an LSTM (Long Short-Term Memory) neural network.
2. Runs an **autonomous agent** that scans all Nifty 50 stocks, decides what to buy/sell on its own, and executes - within strict capital and risk guardrails.
3. Lets you **paper trade** (fake money) so you can measure reliability BEFORE risking real cash.
4. Continuously monitors every open position - comparing the live price to the prediction made when you bought - and flags HOLD or SELL with reasons.
5. Keeps a **detailed ledger** with performance stats and CSV / report export.

A hook for **real-money trading** is in place (`trading/broker_interface.py`) but intentionally disabled. Switch only after paper trading proves consistent.

---

## What's inside (plain English)

| File / folder                  | Role |
|--------------------------------|------|
| `main.py`                      | Launches the desktop window. |
| `config.py`                    | All knobs in one place - capital, risk %, agent params, signal weights, macro inputs. |
| `nifty50.py`                   | Pre-loaded list of 50 most liquid Indian stocks (the agent's universe). |
| `data/fetcher.py`              | Pulls prices, sector indices, fundamentals, and news from Yahoo Finance. |
| `data/sectors.py`              | Maps each stock to its NSE sector index. |
| `model/features.py`            | Adds technical indicators (RSI, MACD, MAs, Bollinger Bands). |
| `model/lstm_model.py`          | LSTM training + multi-step forecast (PyTorch). |
| `signals/`                     | The alt-data signal layer (see below): sentiment, sector, fundamentals, macro, correlation, aggregator. |
| `agent/agent.py`               | The autonomous brain: scan, score, rank, buy/sell, circuit-breaker. |
| `agent/trainer.py`             | Bulk-trains all models; retrains stale ones. |
| `agent/accuracy.py`            | Logs predictions vs. actuals; flags drifting models for retraining. |
| `trading/portfolio.py`         | SQLite-backed paper wallet (cash, positions, trade log, capital, goals). |
| `trading/sizer.py`             | Position sizing under the 75% / per-stock rules. |
| `trading/monitor.py`           | Decision engine: HOLD / SELL based on goals + risk + fresh prediction. |
| `trading/ledger.py`            | Detailed ledger + performance report + CSV export. |
| `trading/broker_interface.py`  | Empty stub for real-money broker (Zerodha / Upstox / Angel One). |
| `gui/app.py`                   | The desktop window (CustomTkinter, dark mode). |
| `gui/charts.py`                | Matplotlib chart with past prices + forecast overlay. |

---

## The autonomous agent

The agent turns this from a manual tool into a hands-off system. **It decides what and when; you set the policy.**

**One cycle (triggered by the "Run trading cycle" button) does:**

1. **Resolve accuracy** - fills in actual prices for past predictions so it knows which models are drifting.
2. **Manage open positions** - runs the monitor and auto-sells anything flagged (stop-loss, max-profit, or a fresh forecast turning negative).
3. **Goal circuit-breaker** - if this period's profit goal is already reached, it STOPS opening new positions until the period resets (daily for intraday, weekly for swing).
4. **Scan & score** every Nifty 50 stock not already held.
5. **Rank & filter** - drop low scores and over-correlated names.
6. **Buy** the top picks, sized by the 75% / per-stock rules, up to a per-cycle limit.

**What the agent decides freely:** which stocks, when to buy, when to sell.
**What it can NEVER break (hard guardrails in code):**
- Max 75% of capital invested (25% reserve always).
- Max 10% in any single stock.
- Max 3 new positions per cycle.
- No buying a stock >85% correlated with a holding.
- No new trades after the period goal is reached.

### Run modes (timing)

The agent's *decisions* are autonomous; you choose how its *timing* works:

- **Manual** - click "Run trading cycle" to run one cycle. Safest; you control every run.
- **Auto-loop** - set an interval (minutes) on the Agent tab and click "Start auto-loop". The agent then runs a cycle automatically every N minutes for as long as the app window is open. Click "Stop auto-loop" or just close the app to end it. Cycles never overlap - if one is still running when the timer fires, the next is skipped until the current finishes. Default interval: `config.AGENT_AUTOLOOP_MINUTES` (20 min).

Auto-loop only runs while the app is open (it has no background schedule), which keeps it easy to supervise while the strategy is still being validated.

### How a stock is scored

For each stock, six signals (each -1 bearish .. +1 bullish) are blended into one score using weights in `config.SIGNAL_WEIGHTS`:

| Signal | Default weight | What it measures | Data source |
|--------|---------------|------------------|-------------|
| LSTM forecast | 45% | Predicted price move | Price history (works now) |
| Sentiment | 15% | News headline tone | Yahoo news (free) + NewsAPI (optional key) |
| Sector momentum | 15% | Is the sector trending up? | Sector indices via Yahoo (works now) |
| Fundamentals | 10% | P/E, margins, growth, debt | Yahoo .info (works, can be patchy for NSE) |
| Macro | 10% | RBI rate / inflation / GDP regime | Manual values in `config.MACRO_INPUTS` |
| Correlation | 5% | Diversification bonus | Computed from price history (works now) |

```
combined_score = weighted average of the six signals
```

Set any weight to 0 to switch a signal off. Tune everything in `config.py`.

### Alternative data: what's live vs. manual

- **Works out of the box (free):** LSTM, sector momentum, correlation, fundamentals (P/E etc. via Yahoo).
- **Optional key:** news sentiment uses Yahoo's free per-ticker headlines automatically; add a NewsAPI.org key (`config.NEWSAPI_KEY` or env `NEWSAPI_KEY`) for richer coverage. Without it, sentiment still works on Yahoo headlines, or returns neutral if none.
- **Manual input:** macro (RBI repo rate, inflation, GDP) has no reliable free India API, so you enter the latest numbers in `config.MACRO_INPUTS` and update them when the RBI / MoSPI publish new data. The macro signal turns them into a risk-on/risk-off score applied to every stock.

### Training & retraining

- **First run:** click **"Train all"** on the Agent tab once. It trains an LSTM for every Nifty 50 stock for the chosen horizon (roughly 30-60s each on CPU, so ~20-40 min total). Models are saved to `model/trained/` and reused forever after.
- **Retraining:** the accuracy tracker logs each prediction and later compares it to the real price. If a model's recent error (MAPE - Mean Absolute Percentage Error) exceeds `RETRAIN_ERROR_THRESHOLD_PCT` (default 4%), click **"Retrain stale models"** and only the drifting ones are retrained.

### Ledger & reports

The **Ledger tab** shows every trade with its reason and P&L, plus performance stats (win rate, profit factor, average win/loss, best/worst, net P&L by symbol). Two export buttons:
- **Export CSV** -> `trade_ledger.csv`
- **Generate report** -> `trading_report.md`

Both save into the project folder.

---

## Setup (one time)

1. Install **Python 3.10 or newer**.
2. Open a terminal in this folder and run:

   ```bash
   pip install -r requirements.txt
   ```

   The big one is **PyTorch** (the deep-learning library) — it's ~700 MB so the first install can take a few minutes. PyTorch supports Python 3.9-3.13. If install fails on Windows, install Microsoft Visual C++ Redistributable first.

3. Launch the app:

   ```bash
   python main.py
   ```

The first time you predict on a stock, the LSTM trains on 5 years of data (about 1-2 minutes on a regular laptop). The trained model is saved under `model/trained/` and reused after that.

---

## How to use it

### 1. Predict & Buy
- Pick a stock from the dropdown (Nifty 50 pre-loaded) or type your own NSE symbol (e.g. `IRCTC` — the app adds `.NS` automatically).
- Choose **swing** (predict 5 days ahead) or **intraday** (predict 4 hourly bars ahead).
- Click **Run prediction**. You'll see the predicted price, expected % return, and a chart with the forecast path.
- If the predicted edge is at least the **Min buy edge %** in settings, the **Paper BUY** button becomes active.
- Enter quantity, click **Paper BUY**. The trade is logged in your SQLite wallet.

### 2. Portfolio & Monitor (goal-based selling)
- Shows every open position with **buy price**, **live price**, **min target**, **max target**, **P&L %**.
- Click **Refresh decisions** — for each position the engine applies this ladder:

  1. **Stop-loss hit** (price drops too far below buy) → SELL — limit losses.
  2. **MAX profit hit** (P&L ≥ max for this horizon) → SELL — don't get greedy chasing more.
  3. **Between MIN and MAX profit** → Run a *fresh* LSTM forecast:
     - Forecast still positive → **HOLD** (let it run toward the MAX cap).
     - Forecast flat / negative → **SELL** (lock in the gain we have).
  4. **Below MIN profit** → existing rules: bearish fresh forecast SELL, past-horizon-with-no-progress SELL, otherwise HOLD.

- **Auto-sell all flagged** executes every SELL decision at once.

**Why two thresholds?** The MIN is the "safe profit floor" — once you cross it you've earned something worth protecting. The MAX is the "greed cap" — past this point the math says don't push your luck. In between is the only zone where the prediction model gets to vote, because that's the only zone where the choice isn't obvious.

### 3. Period goals (Dashboard)
The Dashboard shows two goal cards: **Today (intraday)** and **This week (swing)** — each with realized P&L, the per-position min-target %, an aggregate target in rupees, and a progress bar. This gives you a quick "am I on track today / this week?" snapshot.

### 4. Dashboard KPIs
- Cash, invested, market value, equity, realized & unrealized P&L.
- Recent trades log so you can see exactly what happened and why.

### 5. Settings
- Tune stop-loss, **intraday MIN/MAX profit**, **swing MIN/MAX profit**, minimum buy edge, minimum hold edge, max position size.
- **Reset paper wallet** wipes everything and restarts with the initial cash (`config.INITIAL_PAPER_CASH`, default Rs. 1,00,000).

**Default profit goals:**
| Horizon  | MIN (lock-in floor) | MAX (greed cap) |
|----------|---------------------|------------------|
| Intraday | +1% per day         | +3% per day      |
| Swing    | +3% per week        | +6% per week     |

---

## Capital allocation (the 75% rule)

The app NEVER invests all your money. It enforces three layered rules:

1. **75% invested cap** (`MAX_INVESTED_PCT`) - at least 25% of equity stays as cash reserve, no matter what. This protects you from being fully exposed when the market turns suddenly.
2. **10% per-stock cap** (`MAX_POSITION_PCT`) - no single stock can take more than 10% of your equity. Forces diversification.
3. **Edge-proportional sizing** (`ALLOCATION_SCALE`) - inside those caps, the suggested position size grows with how confident the model is:

   ```
   target_pct_of_equity = predicted_edge_pct * ALLOCATION_SCALE
   ```

   With the default scale of 4:
   - Predicted edge +1%   -> target  4% of equity
   - Predicted edge +1.5% -> target  6% of equity
   - Predicted edge +2.5% -> target 10% of equity (hits per-stock cap)
   - Predicted edge +5%   -> target 20% (still capped at 10% per stock)

4. **Minimum trade threshold** (`MIN_ALLOCATION_PCT`, default 2%) - if the math says "0.7% of equity", skip the trade entirely. Tiny trades waste mental energy and aren't worth the risk.

The Predict & Buy tab auto-fills the suggested quantity using these rules. You can override it, but the Paper BUY button still re-checks the 75% cap before executing - if your override would breach it, the trade is rejected.

The Dashboard "Capital used" tile shows how close you are to the 75% ceiling (e.g. `42.3% / 75%`).

---

## What data goes into a prediction

The LSTM model uses **only historical price data** (for training) and **near-live price data** (for the fresh prediction at evaluation time). It does NOT use news, earnings, macro events, or sector indices in the current version. Here's the full inventory:

### Historic data (for training)
Pulled from Yahoo Finance via `yfinance`:
- **Swing model**: 5 years of daily OHLCV bars (Open, High, Low, Close, Volume).
- **Intraday model**: 60 days of hourly OHLCV bars (Yahoo's hourly history limit is ~730 days).

### Live / near-live data (at prediction time)
- The latest 1 year of daily bars (swing) or 60 days of hourly bars (intraday) - re-fetched on every prediction so the model always sees the most recent action.
- Yahoo Finance is ~15 minutes delayed for NSE. Fine for swing trading. Acceptable for paper-trading intraday. NOT safe for real-money intraday - you'd switch to a broker WebSocket feed (Zerodha/Upstox) for that.

### Derived features (computed from the price bars)
Defined in `model/features.py`:

| Feature       | What it tells the model |
|---------------|--------------------------|
| Open, High, Low, Close, Volume | Raw OHLCV - the foundation |
| SMA_10, SMA_30 | Simple moving averages over 10 and 30 bars - trend direction |
| EMA_12, EMA_26 | Exponential moving averages - faster trend response than SMA |
| RSI_14         | Relative Strength Index - momentum (>70 overbought, <30 oversold) |
| MACD, MACD signal, MACD diff | Moving Average Convergence Divergence - trend strength + momentum |
| BB_high, BB_low, BB_pct | Bollinger Bands - volatility envelope; BB_pct says where price sits in the band (0 to 1) |
| Return_1, Return_5 | 1-bar and 5-bar percentage changes - short-term momentum |

These get scaled to 0-1 (MinMaxScaler) and fed as a 60-bar rolling window into the LSTM, which outputs the next bar's predicted close. The model then iterates that prediction 5 steps (swing) or 4 steps (intraday) to build the full forecast path.

### What is NOT used yet (roadmap items)
- News headlines / sentiment analysis
- Quarterly earnings, P/E ratios, fundamentals
- Macro indicators (RBI rates, inflation, GDP releases)
- Sector index movements (Nifty Bank, IT, Auto)
- Options data (open interest, put/call ratio)
- Order book depth (would need broker feed, not Yahoo)
- Other stocks' correlations

Adding any of these is a meaningful upgrade and would likely improve accuracy - but each adds complexity, more data dependencies, and more risk of overfitting (model memorizing noise instead of learning).

---

## Trading concepts you'll see (plain English)

- **Paper trading**: simulated trades with fake money. No risk, used to test strategy.
- **Stop-loss**: an automatic sell rule that limits how much you can lose on a trade. Default 3%.
- **Take-profit**: automatic sell once you reach a target gain. Default 6%.
- **Position size cap**: never invest more than X% of your wallet in a single stock. Default 10%.
- **Edge**: how much the model expects the price to move (in %). Bigger edge = stronger signal.
- **Swing trading**: holding for days to weeks.
- **Intraday trading**: opening and closing on the same day.
- **LSTM**: a kind of neural network that's good at learning patterns in time-ordered data — like prices.

---

## Important warnings (please read)

- **Yahoo Finance data is delayed ~15 minutes for NSE.** Fine for paper trading and learning. Not safe for actual intraday trading with real money — use a broker WebSocket instead.
- **LSTM models are NOT magic.** They learn from past patterns. Markets can break those patterns at any time (news, earnings, RBI action). Always combine with your own judgment.
- The default risk rules are conservative starting points. Tune them in Settings while watching paper P&L over weeks, not days.
- Real-money trading is intentionally disabled. Don't enable it until paper performance is consistent over at least 30+ trades.

---

## When you're ready for real money

1. Open `trading/broker_interface.py`.
2. Pick a broker — Zerodha Kite Connect has the cleanest API for Python.
3. Implement each method (`buy`, `sell`, `get_positions`, etc.) using the broker's SDK. Keep the SAME function signatures as `Portfolio` so the rest of the app keeps working unchanged.
4. In `gui/app.py`, replace `self.portfolio = Portfolio()` with the real adapter when `self.mode.get() == "real"`.
5. Start with the smallest possible quantity (1 share) and your tightest risk settings. Watch every trade for a week before scaling up.

---

## Roadmap ideas

- Backtesting tab — replay history through the model and see win rate, drawdown, Sharpe ratio.
- News sentiment feature — feed headlines into the model.
- Walk-forward retraining — automatically retrain weekly.
- Telegram / email alerts on SELL flags.
- Multi-stock portfolio optimization (Kelly criterion or risk-parity).

---

Built as a learning + monitoring tool, not as a guaranteed money machine. Stay curious, stay patient.
