"""
Central settings for the app. Change values here to tune behavior.
All amounts are in Indian Rupees (INR).
"""

# =====================================================================
# TRADING UNIVERSE
# =====================================================================
# What instruments the agent scans, trains models for, and the price
# cache scrapes.  Change via the GUI (Agent tab → Universe) — do NOT
# edit this manually unless you also retrain models for the new set.
#
# Values:
#   "nifty50"   — 50 individual NSE stocks (default, higher risk/return)
#   "funds"     — curated ETF list from funds.py (lower risk, recommended
#                  for beginners or capital preservation)
#   "both"      — union of both universes (large, slow to train)
#
# IMPORTANT SEQUENCE:
#   1. Set this value (here or via GUI)
#   2. Click "Train all" to build models for the chosen universe
#   3. Then run the buy scan
# Skipping step 2 after changing universe will cause the agent to use
# stale models trained on completely different instruments.
ACTIVE_UNIVERSE = "nifty50"

# ----- Paper trading wallet -----
# This is the TOTAL amount of capital available to invest. You can change it
# from the GUI (Settings tab -> "Set total capital") or here.
INITIAL_PAPER_CASH = 100_000.0     # Starting fake money (Rs. 1 lakh)
TOTAL_INVESTMENT_AMOUNT = 100_000.0  # Same as above on first run; GUI updates it

# ----- Trading mode -----
# "paper" = simulated trades with fake money (safe)
# "real"  = actual broker trades (NOT IMPLEMENTED - placeholder only)
DEFAULT_MODE = "paper"

# ----- Risk rules: stop-loss (the safety floor for losses) -----
STOP_LOSS_PCT = 0.03          # Initial stop-loss: 3% below buy price

# Trailing stop-loss — follows the price UP, never down.
# Once price rises, the stop rises with it locking in the gain.
# e.g. buy Rs.100, price rises to Rs.120 → stop moves to Rs.116.40
# Always set equal to or tighter than STOP_LOSS_PCT.
TRAILING_STOP_PCT = 0.03      # Trail 3% below highest price seen

# ----- Nifty market trend filter -----
# Only open NEW positions when Nifty 50 is above its N-day moving average.
# Individual stocks follow the index ~70% of the time.
# Buying in a downtrend fights the current. Set False to disable.
NIFTY_TREND_FILTER   = True
NIFTY_TREND_SMA_PERIOD = 20   # Days for the SMA (20 = ~1 trading month)

# ----- RSI overbought filter at entry -----
# Don't buy a stock whose RSI is already high — the easy move happened.
# Above 65-70 = overbought territory, likely to pull back.
MAX_RSI_AT_BUY = 65.0

# ----- Daily portfolio drawdown limit -----
# If the portfolio is down more than this % on any single day,
# stop opening new positions for the rest of the day.
MAX_DAILY_DRAWDOWN_PCT = 0.015   # 1.5% of equity

# ----- Stagnation exit (dead money) -----
# If a position has barely moved after half its planned duration, exit.
# Capital sitting flat earns nothing and blocks better opportunities.
STAGNATION_DAYS_RATIO  = 0.5    # Check after this fraction of horizon has passed
STAGNATION_MIN_MOVE_PCT = 0.5   # If abs(P&L%) < this after half-time → SELL

# ----- Profit goals (per horizon) -----
# Logic in monitor.py:
#   If P&L% >= MAX_PROFIT      -> SELL  (don't get greedy)
#   If P&L% in [MIN, MAX)      -> Run a FRESH prediction:
#                                 - forecast positive -> HOLD (let it run)
#                                 - forecast flat/neg -> SELL (lock the gain)
#   If P&L% <  MIN_PROFIT      -> existing rules apply (stop-loss + monitor)
#
# Swing horizon goals are "per week" (typical hold = 5 trading days).
# Intraday horizon goals are "per day".
SWING_MIN_PROFIT_PCT = 0.03      # Weekly safe floor - start protecting gains
SWING_MAX_PROFIT_PCT = 0.06      # Weekly cap - auto-sell, don't push luck

INTRADAY_MIN_PROFIT_PCT = 0.01   # Daily safe floor
INTRADAY_MAX_PROFIT_PCT = 0.03   # Daily cap

# ----- Position sizing & entry filters -----
# Capital allocation rules:
#   1. Never invest more than MAX_INVESTED_PCT of equity total (reserve buffer).
#   2. Never put more than MAX_POSITION_PCT of equity into a single stock.
#   3. The suggested per-stock allocation is proportional to predicted edge:
#        target_pct_of_equity = predicted_return_pct * ALLOCATION_SCALE
#      then capped at MAX_POSITION_PCT and at remaining budget under the
#      MAX_INVESTED_PCT ceiling.
#
# Example with defaults (scale=4, max_position=10%, max_invested=75%):
#   Predicted edge 1.5% -> target 6% of equity
#   Predicted edge 3.0% -> target 12% -> capped to 10% (max position)
#   Predicted edge 0.8% -> target 3.2% of equity
MAX_INVESTED_PCT = 0.75          # Keep at least 25% of equity in cash reserve
MAX_POSITION_PCT = 0.10          # Single-stock cap: max 10% of equity
ALLOCATION_SCALE = 4.0           # edge% x scale = target position % of equity
MIN_ALLOCATION_PCT = 0.02        # Below this, skip trade (too small to matter)
MIN_PREDICTION_EDGE = 0.01       # Legacy — kept for Settings tab display (decimal)

# Minimum LSTM-predicted return required before the agent will even consider
# buying. These are HARD gates — negative or weak predictions are rejected
# regardless of how good sentiment/sector/macro look.
# Values are in percentage points (e.g. 1.5 means the model must predict ≥+1.5%).
SWING_MIN_LSTM_EDGE_PCT    = 1.5   # swing: need ≥1.5% predicted over 5 days
INTRADAY_MIN_LSTM_EDGE_PCT = 1.0   # intraday: need ≥1.0% predicted over ~4 h

MIN_HOLD_CONFIDENCE = 0.0        # Below-min-profit: sell if fresh edge < this

# ----- LSTM model -----
LSTM_LOOKBACK_DAYS = 60          # Use last 60 bars to predict next step
LSTM_EPOCHS = 20                 # Training rounds (raise for better fit)
LSTM_BATCH_SIZE = 32
SWING_HORIZON_DAYS = 5           # Predict 5 days ahead for swing trading
INTRADAY_HORIZON_BARS = 4        # Predict 4 hourly bars ahead for intraday

# ----- Price cache (NSE scraping) -----
# Background thread scrapes all Nifty 50 prices from NSE every N seconds.
# Lower = fresher data but more requests; 30 s is a safe default.
# Set to 0 to disable (get_latest_price returns None until cache warms up).
PRICE_CACHE_INTERVAL_SECS = 30

# ----- Database -----
DB_PATH = "portfolio.db"         # SQLite file (created on first run)

# ----- Trained models folder -----
MODEL_DIR = "model/trained"


# ----- Helpers used by other modules -----

def get_active_universe() -> list[str]:
    """
    Return the current symbol list based on ACTIVE_UNIVERSE.
    Always imports fresh so GUI changes to ACTIVE_UNIVERSE are reflected
    without restarting.
    """
    from nifty50 import NIFTY_50
    from funds import ALL_FUNDS
    if ACTIVE_UNIVERSE == "funds":
        return ALL_FUNDS
    if ACTIVE_UNIVERSE == "both":
        seen: set[str] = set()
        combined = []
        for s in NIFTY_50 + ALL_FUNDS:
            if s not in seen:
                seen.add(s)
                combined.append(s)
        return combined
    return NIFTY_50   # default: "nifty50"


def min_lstm_edge_for(horizon: str) -> float:
    """
    Minimum LSTM-predicted return (in percentage points) required to buy.
    e.g. returns 1.5 → model must predict at least +1.5% gain.
    """
    if horizon == "intraday":
        return INTRADAY_MIN_LSTM_EDGE_PCT
    return SWING_MIN_LSTM_EDGE_PCT


def profit_goals_for(horizon: str) -> tuple[float, float]:
    """
    Return (min_profit_pct, max_profit_pct) for a given horizon.
    Always returns ratios (e.g. 0.03 for 3%), not percentage points.
    """
    if horizon == "intraday":
        return INTRADAY_MIN_PROFIT_PCT, INTRADAY_MAX_PROFIT_PCT
    return SWING_MIN_PROFIT_PCT, SWING_MAX_PROFIT_PCT


def goal_period_label(horizon: str) -> str:
    """Human-readable label for the goal period."""
    return "Daily" if horizon == "intraday" else "Weekly"


# =====================================================================
# AUTONOMOUS AGENT
# =====================================================================

# The agent trades the whole universe automatically. It scans every stock,
# scores it, ranks them, and buys the best while respecting all capital rules.
AGENT_DEFAULT_HORIZON = "swing"      # "swing" or "intraday"
AGENT_MAX_NEW_BUYS_PER_CYCLE = 3     # Don't open more than N new positions per run
AGENT_MIN_COMBINED_SCORE = 0.15      # Only buy if blended score >= this (0..1-ish)
AGENT_MAX_CORRELATION = 0.85         # Skip a buy if it's >85% correlated with a holding

# Auto-loop: when enabled in the GUI, the agent runs a cycle automatically
# every this-many minutes while the app window stays open. Stop anytime with
# the toggle, or just close the app.
AGENT_AUTOLOOP_MINUTES = 20

# Goal circuit-breaker:
#   Once the period goal (daily for intraday, weekly for swing) is reached,
#   the agent stops opening NEW positions until the period resets.
#   Existing positions are still risk-managed (stop-loss / max-profit) unless
#   FREEZE_ALL_AFTER_GOAL is True.
GOAL_TARGET_PCT_OF_CAPITAL = 0.02    # Period profit goal = 2% of total capital
FREEZE_ALL_AFTER_GOAL = False        # True = also stop protective sells after goal


# =====================================================================
# SIGNAL WEIGHTS  (how the agent blends data sources into one score)
# =====================================================================
# Each signal returns a value roughly in [-1, +1]. The combined score is a
# weighted average. Set a weight to 0 to switch a signal off completely.
SIGNAL_WEIGHTS = {
    "lstm":         0.45,   # Price forecast edge (the core signal)
    "sentiment":    0.15,   # News headline sentiment
    "sector":       0.15,   # Sector index momentum
    "fundamentals": 0.10,   # P/E and basic fundamentals sanity
    "macro":        0.10,   # RBI rate / inflation / GDP regime
    "correlation":  0.05,   # Diversification bonus (less correlated = better)
}


# =====================================================================
# RETRAINING (accuracy-driven)
# =====================================================================
# After each prediction we log it. Later we compare to what actually happened
# and compute error. If a model's recent error is too high, we retrain it.
RETRAIN_ERROR_THRESHOLD_PCT = 4.0    # Retrain if rolling MAPE > 4%
RETRAIN_MIN_SAMPLES = 5              # Need at least N scored predictions to judge
ACCURACY_DB_PATH = "accuracy.db"     # Where prediction-vs-actual logs live


# =====================================================================
# ALTERNATIVE DATA PROVIDERS
# =====================================================================
# News + sentiment. If you have a NewsAPI.org key, set it here or in the
# environment variable NEWSAPI_KEY. If empty, sentiment returns neutral (0).
NEWSAPI_KEY = ""                     # "" = sentiment disabled (neutral default)
NEWS_LOOKBACK_DAYS = 7

# Macro regime. India has no reliable free API for these, so enter the latest
# values manually (update them when the RBI / MoSPI release new numbers).
# The macro signal turns these into a risk-on / risk-off multiplier.
MACRO_INPUTS = {
    "rbi_repo_rate_pct": 6.50,       # Current RBI repo rate
    "rbi_last_move": "hold",         # "cut" (bullish), "hike" (bearish), "hold"
    "cpi_inflation_pct": 5.0,        # Latest CPI inflation YoY
    "inflation_target_pct": 4.0,     # RBI's target (for comparison)
    "gdp_growth_pct": 6.5,           # Latest GDP growth YoY
    "gdp_trend": "stable",           # "rising", "falling", "stable"
}
