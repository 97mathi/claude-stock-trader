# Stock Predictor & Paper Trader (India)

A desktop app that:
1. Predicts future prices of Indian (NSE/BSE) stocks using a **walk-forward-tuned LSTM** neural network.
2. Runs a **continuous market monitor** that evaluates open positions automatically while the app is open — no manual trigger needed.
3. Runs an **autonomous agent** that scans all Nifty 50 stocks, decides what to buy on its own, and executes — within strict capital and risk guardrails.
4. Lets you **paper trade** (fake money) so you can measure reliability before risking real cash.
5. Keeps a **detailed ledger** with performance stats and CSV / markdown export.

A hook for **real-money trading** is in place (`trading/broker_interface.py`) but intentionally disabled. Switch only after paper trading proves consistent.

→ Full usage guide: [`docs/how_to_use.html`](docs/how_to_use.html)
→ Architecture diagram: [`docs/design_diagram.png`](docs/design_diagram.png)
→ Project history: [`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md)

---

## What's inside

| File / folder                  | Role |
|--------------------------------|------|
| `main.py`                      | Launches the desktop window. |
| `config.py`                    | All knobs in one place — capital, risk %, agent params, signal weights, macro inputs. |
| `nifty50.py`                   | Pre-loaded list of 50 most liquid Indian stocks (the agent's universe). |
| `data/fetcher.py`              | Pulls prices, sector indices, fundamentals, and news from Yahoo Finance. Parallel batch fetch (10 threads) cuts a full 50-stock price refresh from ~30 s to ~3–5 s. |
| `data/sectors.py`              | Maps each stock to its NSE sector index. |
| `model/features.py`            | Adds 18 technical indicators per bar (RSI, MACD, MAs, Bollinger Bands, returns). |
| `model/lstm_model.py`          | LSTM training with walk-forward architecture search + multi-step forecast (PyTorch). |
| `model/trained/`               | Saved `.pt` weights + `.json` meta per stock/horizon. Auto-created; not committed to git. |
| `signals/`                     | Alt-data signal layer: sentiment, sector, fundamentals, macro, correlation, aggregator. |
| `agent/agent.py`               | Autonomous brain: scan, score, rank, buy/sell, circuit-breaker. |
| `agent/trainer.py`             | Bulk-trains all models; retrains stale ones (force-saves when MAPE-flagged). |
| `agent/accuracy.py`            | Logs predictions vs. actuals; flags drifting models for retraining. |
| `trading/portfolio.py`         | SQLite-backed paper wallet (cash, positions, trade log, capital, goals). |
| `trading/sizer.py`             | Position sizing under the 75% / per-stock rules. |
| `trading/monitor.py`           | Decision engine: HOLD / SELL based on goals + risk + fresh prediction. Negative positions held until recovery. |
| `trading/ledger.py`            | Detailed ledger + performance report + CSV export. |
| `trading/broker_interface.py`  | Empty stub for real-money broker (Zerodha / Upstox / Angel One). |
| `gui/app.py`                   | The desktop window (CustomTkinter, dark mode, 6 tabs). |
| `gui/charts.py`                | Matplotlib chart with past prices + forecast overlay. |
| `docs/`                        | Architecture diagram, HTML usage guide, project history. |

---

## The LSTM model

### Walk-forward architecture search (new)

Every time a model is trained or retrained, the engine runs a **walk-forward architecture search** before settling on a final model:

1. **4 candidate architectures** are evaluated: small (32→16), medium (64→32, former default), large (128→64), and medium with stronger dropout.
2. Each candidate is tested across **2 expanding walk-forward folds** — training always on past data, validating on the next unseen chunk. This mimics real usage.
3. Candidates are scored by: `0.4 × MAE + 0.6 × (1 − directional_accuracy)`. Directional accuracy (up/down call) is weighted 60% because direction matters more than exact price for trading decisions.
4. The winning architecture trains on the full 85% split and is saved.

This adds ~3–4× training time per stock but produces a provably better model for each stock/horizon pair.

### Save-only-if-better guard

For **voluntary retrains** (clicking "Retrain model"), the new model is only saved if its walk-forward score beats the existing saved model. If the new model is worse, the old one is kept and the result says so.

For **stale-model retrains** (MAPE > 4% threshold), the new model always overwrites — the existing model is already proven bad on live data, so there is nothing worth defending.

### Confidence score

`Prediction.confidence` now reflects the model's **real walk-forward directional accuracy** (e.g. `0.64` = model called up/down correctly 64% of the time across folds). Old models without walk-forward data fall back to the former val_loss proxy.

---

## The continuous market monitor

The app **continuously monitors open positions** while it is running — no toggle, no button. Every N minutes (default 20, configurable in the Agent tab):

1. Fetches live prices for all holdings in parallel.
2. Runs the HOLD/SELL decision engine with a fresh LSTM prediction for each position.
3. Logs SELL recommendations to the Agent tab.
4. Refreshes the Portfolio & Monitor tab automatically.

Click **"Check now"** to trigger an immediate check and reset the timer. The interval is read from the entry box on the Agent tab — change it any time.

### HOLD/SELL decision ladder (first match wins)

| Condition | Action | Reason |
|---|---|---|
| Price ≤ stop-loss (−3%) | **SELL** | Limit losses |
| Price < buy price (negative P&L) | **HOLD** | Wait for recovery — only stop-loss can force a sell here |
| P&L% ≥ MAX for horizon | **SELL** | Greed cap — don't push luck |
| MIN ≤ P&L% < MAX | Fresh LSTM prediction → positive → **HOLD** | Let it run |
| MIN ≤ P&L% < MAX | Fresh LSTM prediction → flat/negative → **SELL** | Lock the gain |
| P&L% < MIN, model turns bearish | **SELL** | Model changed its mind |
| P&L% < MIN, past horizon, below target | **SELL** | Thesis expired |
| Otherwise | **HOLD** | Wait |

---

## The autonomous agent (buy scan)

Click **"Run buy scan"** on the Agent tab to run one cycle. The continuous monitor handles selling; the buy scan handles new entries.

**One cycle does:**

1. **Resolve accuracy** — fills actual prices for past predictions to detect drifting models.
2. **Goal circuit-breaker** — if the period goal is already reached, stops opening new positions until the period resets (daily for intraday, weekly for swing).
3. **Scan & score** every Nifty 50 stock not already held.
4. **Rank & filter** — drop low scores and over-correlated names.
5. **Buy** the top picks, sized by the 75% / per-stock rules, up to 3 new positions per cycle.

### How a stock is scored

Six signals blended by weights in `config.SIGNAL_WEIGHTS`:

| Signal | Default weight | What it measures |
|--------|---------------|------------------|
| LSTM forecast | 45% | Predicted price move |
| Sentiment | 15% | News headline tone |
| Sector momentum | 15% | Is the sector trending up? |
| Fundamentals | 10% | P/E, margins, growth, debt |
| Macro | 10% | RBI rate / inflation / GDP |
| Correlation | 5% | Diversification bonus |

### Hard guardrails (cannot be overridden)

- Max 75% of capital invested (25% reserve always).
- Max 10% in any single stock.
- Max 3 new positions per cycle.
- No buying a stock >85% correlated with an existing holding.
- No new trades after the period goal is reached.

---

## Training & retraining

| Action | When | What happens |
|---|---|---|
| **Train all** (Agent tab) | First run only | Trains walk-forward LSTM for every untrained Nifty 50 stock. ~30–60 s each on CPU. |
| **Retrain model** (Predict tab) | Stock predictions feel off | Retrains one stock; saves only if new walk-forward score beats the saved model. |
| **Retrain stale** (Agent tab) | MAPE > 4% flagged | Retrains only drifting models; always saves (existing model proven bad). |

Trained models live in `model/trained/` — saved on disk permanently, reused across sessions.

The **Settings tab → Saved Models** section lists every trained model with its directional accuracy, validation loss, file size, and training date. A **Clear all models** button deletes everything (next prediction retrains from scratch).

---

## Setup

1. Install **Python 3.10 or newer**.
2. Open a terminal in this folder and run:

   ```bash
   pip install -r requirements.txt
   ```

   PyTorch (~700 MB) dominates the install time. On Windows, install Microsoft Visual C++ Redistributable first if the install fails.

3. Launch:

   ```bash
   python main.py
   ```

4. On the **Agent tab**, click **"Train all"** once. This trains an LSTM for every Nifty 50 stock (~20–40 min total on CPU). Models are saved and reused after that — you only do this once.

---

## Data & delay

- **Data source:** Yahoo Finance (free, no API key, `.NS` suffix for NSE).
- **Price delay:** ~15 minutes for NSE on the free feed. Fine for swing trading and paper trading. Not safe for real-money intraday — switch to a Zerodha/Upstox WebSocket for that.
- **Batch fetch:** all 50 Nifty stock prices are fetched in parallel (10 threads), so a full portfolio refresh takes ~3–5 s instead of ~30 s.

### Features fed to the LSTM (from `model/features.py`)

| Feature | What it tells the model |
|---|---|
| Open, High, Low, Close, Volume | Raw OHLCV |
| SMA_10, SMA_30 | Trend direction |
| EMA_12, EMA_26 | Faster trend response |
| RSI_14 | Momentum (>70 overbought, <30 oversold) |
| MACD, MACD signal, MACD diff | Trend strength + momentum |
| BB_high, BB_low, BB_pct | Volatility envelope |
| Return_1, Return_5 | Short-term price momentum |

All 18 features are scaled to 0–1 and fed as a 60-bar rolling window. The model iterates 5 steps (swing) or 4 steps (intraday) to build the full forecast path.

---

## Capital allocation

Three layered rules enforced in code — cannot be overridden:

1. **75% invested cap** — at least 25% of equity stays as cash, always.
2. **10% per-stock cap** — no single position exceeds 10% of equity.
3. **Edge-proportional sizing** — suggested position = `edge% × ALLOCATION_SCALE (default 4)`:
   - Edge +1% → 4% of equity
   - Edge +1.5% → 6% of equity
   - Edge +2.5% → 10% (hits per-stock cap)
4. **2% minimum** — trades below 2% of equity are skipped entirely.

---

## Important warnings

- **Yahoo data is ~15 min delayed for NSE.** Fine for paper/swing. Not for real intraday.
- **LSTM models are not magic.** They learn past patterns. Markets can break those patterns instantly on news or macro events.
- The default risk settings are conservative starting points. Watch paper P&L over weeks — not days — before tuning.
- Real-money trading is intentionally disabled. Enable only after paper performance is consistent over 30+ trades.

---

## When you're ready for real money

1. Open `trading/broker_interface.py`.
2. Implement each method (`buy`, `sell`, `get_positions`) using the broker's SDK (Zerodha Kite Connect has the cleanest Python API).
3. In `gui/app.py`, replace `self.portfolio = Portfolio()` with the real adapter when `self.mode.get() == "real"`.
4. Start with 1-share positions and maximum-tight risk settings. Watch every trade for a week before scaling up.

---

## Roadmap

- Backtesting tab — replay history through the model and show win rate, drawdown, Sharpe ratio.
- Angel One SmartAPI integration — real-time NSE prices (eliminates the 15-min Yahoo delay).
- Telegram / email alerts on SELL flags.
- Multi-stock portfolio optimisation (Kelly criterion or risk-parity).

---

Built as a learning + monitoring tool, not a guaranteed money machine. Stay curious, stay patient.
