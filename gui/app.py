"""
Main desktop application window.

Tabs:
  1. Dashboard           - wallet summary, period goals, capital usage, trades
  2. Agent               - autonomous: train all, run trading cycle, view report
  3. Predict & Buy       - manual: pick stock, run LSTM, see forecast, paper-buy
  4. Portfolio & Monitor - open positions, hold/sell decisions, auto-sell
  5. Ledger              - full trade log, performance stats, CSV/report export
  6. Settings            - risk thresholds, capital, allocation, reset wallet
"""

from __future__ import annotations
import os
import threading
import customtkinter as ctk
from tkinter import messagebox

import config
from nifty50 import NIFTY_50
from funds import FUND_CATEGORIES, ALL_FUNDS, RECOMMENDED_FUNDS
from data.fetcher import get_daily_history, get_latest_price, get_latest_prices
from data.price_cache import price_cache
from model.lstm_model import predict, train, Prediction
from trading.portfolio import Portfolio
from trading.monitor import evaluate_all, auto_execute
from trading.sizer import suggest_size
from trading.ledger import Ledger
from agent.agent import TradingAgent
from agent.trainer import train_universe, retrain_stale, untrained_symbols
from agent.accuracy import AccuracyTracker
from gui.charts import make_price_chart


# Where exported reports/CSVs are saved (the user's workspace folder)
EXPORT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class StockTraderApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Stock Predictor & Paper Trader (India)")
        self.geometry("1180x780")
        self.minsize(1080, 720)

        self.mode = ctk.StringVar(value=config.DEFAULT_MODE)
        self.portfolio = Portfolio()
        self.agent = TradingAgent(self.portfolio)
        self.cycle_busy = False
        self._monitor_after_id = None

        self._build_layout()
        self._refresh_dashboard()
        self._start_auto_monitor()
        self._start_price_cache()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ============================================================
    # Layout
    # ============================================================
    def _build_layout(self):
        # Top bar
        top = ctk.CTkFrame(self, height=60, corner_radius=0)
        top.pack(fill="x", side="top")

        ctk.CTkLabel(top, text="Stock Predictor & Paper Trader",
                     font=ctk.CTkFont(size=18, weight="bold")
                     ).pack(side="left", padx=20, pady=12)

        mode_frame = ctk.CTkFrame(top, fg_color="transparent")
        mode_frame.pack(side="right", padx=20)
        ctk.CTkLabel(mode_frame, text="Mode:").pack(side="left", padx=(0, 8))
        self.mode_switch = ctk.CTkSegmentedButton(
            mode_frame, values=["paper", "real"],
            variable=self.mode, command=self._on_mode_change)
        self.mode_switch.pack(side="left")

        # Tabs
        self.tabs = ctk.CTkTabview(self, corner_radius=8)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)
        for name in ("Dashboard", "Agent", "Predict",
                     "Portfolio & Monitor", "Ledger", "Settings"):
            self.tabs.add(name)

        self._build_dashboard(self.tabs.tab("Dashboard"))
        self._build_agent(self.tabs.tab("Agent"))
        self._build_predict(self.tabs.tab("Predict"))
        self._build_monitor(self.tabs.tab("Portfolio & Monitor"))
        self._build_ledger(self.tabs.tab("Ledger"))
        self._build_settings(self.tabs.tab("Settings"))

    # ============================================================
    # Dashboard tab
    # ============================================================
    def _build_dashboard(self, root):
        self.kpi_frame = ctk.CTkFrame(root)
        self.kpi_frame.pack(fill="x", padx=12, pady=12)

        self.kpi_labels = {}
        for i, key in enumerate(["Cash", "Invested", "Capital used",
                                  "Equity", "Realized P&L", "Unrealized P&L"]):
            box = ctk.CTkFrame(self.kpi_frame, corner_radius=8)
            box.grid(row=0, column=i, padx=6, pady=6, sticky="nsew")
            self.kpi_frame.grid_columnconfigure(i, weight=1)
            ctk.CTkLabel(box, text=key, text_color="#9aa4b2",
                         font=ctk.CTkFont(size=12)).pack(pady=(8, 0))
            val = ctk.CTkLabel(box, text="—",
                               font=ctk.CTkFont(size=18, weight="bold"))
            val.pack(pady=(2, 10))
            self.kpi_labels[key] = val

        # Period goals (Today / This week)
        ctk.CTkLabel(root, text="Period goals",
                     font=ctk.CTkFont(size=14, weight="bold")
                     ).pack(anchor="w", padx=14, pady=(8, 4))

        self.goals_frame = ctk.CTkFrame(root)
        self.goals_frame.pack(fill="x", padx=12, pady=(0, 8))
        self.goal_labels = {}
        for i, label in enumerate(["Today (intraday)", "This week (swing)"]):
            box = ctk.CTkFrame(self.goals_frame, corner_radius=8)
            box.grid(row=0, column=i, padx=6, pady=6, sticky="nsew")
            self.goals_frame.grid_columnconfigure(i, weight=1)
            ctk.CTkLabel(box, text=label, text_color="#9aa4b2",
                         font=ctk.CTkFont(size=12)
                         ).pack(anchor="w", padx=10, pady=(8, 0))
            realized_lbl = ctk.CTkLabel(
                box, text="Realized: —",
                font=ctk.CTkFont(size=13, weight="bold"))
            realized_lbl.pack(anchor="w", padx=10)
            target_lbl = ctk.CTkLabel(box, text="Target: —",
                                      text_color="#9aa4b2",
                                      font=ctk.CTkFont(size=11))
            target_lbl.pack(anchor="w", padx=10)
            bar = ctk.CTkProgressBar(box, height=10)
            bar.pack(fill="x", padx=10, pady=(4, 4))
            bar.set(0)
            positions_lbl = ctk.CTkLabel(box, text="Open positions: 0",
                                         text_color="#9aa4b2",
                                         font=ctk.CTkFont(size=11))
            positions_lbl.pack(anchor="w", padx=10, pady=(0, 8))
            self.goal_labels[label] = {
                "realized": realized_lbl,
                "target": target_lbl,
                "bar": bar,
                "positions": positions_lbl,
            }

        # Recent trades
        ctk.CTkLabel(root, text="Recent trades",
                     font=ctk.CTkFont(size=14, weight="bold")
                     ).pack(anchor="w", padx=14, pady=(4, 4))

        self.trades_box = ctk.CTkTextbox(root, height=260,
                                         font=ctk.CTkFont(family="Consolas",
                                                          size=12))
        self.trades_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        ctk.CTkButton(root, text="Refresh",
                      command=self._refresh_dashboard).pack(pady=(0, 12))

    def _refresh_dashboard(self):
        import datetime as dt
        positions = self.portfolio.get_positions()
        symbols = [p.symbol for p in positions]
        live_prices = get_latest_prices(symbols) if symbols else {}
        # Replace any None with buy price so equity stays sensible
        for p in positions:
            if live_prices.get(p.symbol) is None:
                live_prices[p.symbol] = p.buy_price

        summary = self.portfolio.total_value(live_prices)
        self.kpi_labels["Cash"].configure(text=f"Rs. {summary['cash']:,.2f}")
        self.kpi_labels["Invested"].configure(text=f"Rs. {summary['invested']:,.2f}")
        cap_used_pct = (summary['invested'] / max(summary['equity'], 1.0)) * 100
        cap = config.MAX_INVESTED_PCT * 100
        self.kpi_labels["Capital used"].configure(
            text=f"{cap_used_pct:.1f}% / {cap:.0f}%")
        self.kpi_labels["Equity"].configure(text=f"Rs. {summary['equity']:,.2f}")
        self.kpi_labels["Realized P&L"].configure(
            text=f"Rs. {summary['realized_pnl']:+,.2f}")
        self.kpi_labels["Unrealized P&L"].configure(
            text=f"Rs. {summary['unrealized_pnl']:+,.2f}")

        # ---- Goals panel ----
        equity = max(summary["equity"], 1.0)
        today_iso = dt.date.today().isoformat()
        week_start = (dt.date.today()
                      - dt.timedelta(days=dt.date.today().weekday())).isoformat()

        today_realized = self.portfolio.realized_pnl_since(today_iso)
        week_realized = self.portfolio.realized_pnl_since(week_start)

        intraday_positions = [p for p in positions if p.horizon == "intraday"]
        swing_positions = [p for p in positions if p.horizon == "swing"]

        intraday_target_pct = config.INTRADAY_MIN_PROFIT_PCT * 100
        swing_target_pct = config.SWING_MIN_PROFIT_PCT * 100

        intraday_target_rs = equity * config.INTRADAY_MIN_PROFIT_PCT * \
            max(0.1, len(intraday_positions) * config.MAX_POSITION_PCT)
        swing_target_rs = equity * config.SWING_MIN_PROFIT_PCT * \
            max(0.1, len(swing_positions) * config.MAX_POSITION_PCT)

        # Today
        g = self.goal_labels["Today (intraday)"]
        g["realized"].configure(
            text=f"Realized today: Rs. {today_realized:+,.2f}",
            text_color="#69db7c" if today_realized >= 0 else "#ff6b6b")
        g["target"].configure(
            text=f"Min target: +{intraday_target_pct:.1f}% per intraday position "
                 f"(~Rs. {intraday_target_rs:,.0f} aggregate)")
        progress = 0.0
        if intraday_target_rs > 0:
            progress = max(0.0, min(1.0, today_realized / intraday_target_rs))
        g["bar"].set(progress)
        g["positions"].configure(
            text=f"Open intraday positions: {len(intraday_positions)}")

        # This week
        g = self.goal_labels["This week (swing)"]
        g["realized"].configure(
            text=f"Realized this week: Rs. {week_realized:+,.2f}",
            text_color="#69db7c" if week_realized >= 0 else "#ff6b6b")
        g["target"].configure(
            text=f"Min target: +{swing_target_pct:.1f}% per swing position "
                 f"(~Rs. {swing_target_rs:,.0f} aggregate)")
        progress = 0.0
        if swing_target_rs > 0:
            progress = max(0.0, min(1.0, week_realized / swing_target_rs))
        g["bar"].set(progress)
        g["positions"].configure(
            text=f"Open swing positions: {len(swing_positions)}")

        # Trades
        self.trades_box.configure(state="normal")
        self.trades_box.delete("1.0", "end")
        header = f"{'Time':<20} {'Side':<5} {'Symbol':<14} {'Qty':>5} {'Price':>12} {'P&L':>12}  Reason\n"
        self.trades_box.insert("end", header)
        self.trades_box.insert("end", "-" * 110 + "\n")
        for t in self.portfolio.get_trades(50):
            line = (f"{t.timestamp:<20} {t.side:<5} {t.symbol:<14} "
                    f"{t.quantity:>5} {t.price:>12.2f} {t.pnl:>+12.2f}  {t.reason}\n")
            self.trades_box.insert("end", line)
        self.trades_box.configure(state="disabled")

    # ============================================================
    # Predict & Buy tab
    # ============================================================
    def _build_predict(self, root):
        top = ctk.CTkFrame(root)
        top.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(top, text="Stock:").pack(side="left", padx=(8, 4))
        self.symbol_combo = ctk.CTkComboBox(top, values=NIFTY_50, width=200)
        self.symbol_combo.pack(side="left", padx=4)
        self.symbol_combo.set("RELIANCE.NS")

        ctk.CTkLabel(top, text="Horizon:").pack(side="left", padx=(16, 4))
        self.horizon = ctk.StringVar(value="swing")
        ctk.CTkSegmentedButton(top, values=["swing", "intraday"],
                               variable=self.horizon).pack(side="left", padx=4)

        ctk.CTkButton(top, text="Run prediction",
                      command=self._run_predict_threaded
                      ).pack(side="left", padx=12)
        ctk.CTkButton(top, text="Retrain model",
                      command=self._retrain_threaded
                      ).pack(side="left", padx=4)

        # Result area
        body = ctk.CTkFrame(root)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.result_box = ctk.CTkTextbox(
            body, height=160, font=ctk.CTkFont(family="Consolas", size=13))
        self.result_box.pack(fill="x", padx=8, pady=8)
        self.result_box.insert("end",
            "Pick a stock + horizon, then click 'Run prediction'.\n"
            "First run for a new stock trains the model (1-2 min).\n")
        self.result_box.configure(state="disabled")

        self.chart_holder = ctk.CTkFrame(body, fg_color="transparent")
        self.chart_holder.pack(fill="both", expand=True, padx=8, pady=4)

    def _run_predict_threaded(self):
        symbol = self.symbol_combo.get()
        horizon = self.horizon.get()
        self._log_result(f"\nRunning {horizon} prediction for {symbol}...\n")
        threading.Thread(
            target=self._run_predict, args=(symbol, horizon), daemon=True
        ).start()

    def _run_predict(self, symbol: str, horizon: str):
        try:
            pred = predict(symbol, horizon)  # type: ignore
            hist = get_daily_history(symbol, "6mo")
            self.after(0, lambda: self._show_prediction(pred, hist))
        except Exception as e:
            self.after(0, lambda: self._log_result(f"ERROR: {e}\n"))

    def _retrain_threaded(self):
        symbol = self.symbol_combo.get()
        horizon = self.horizon.get()
        self._log_result(f"\nRetraining {horizon} model for {symbol} "
                         f"(~1-2 min)...\n")
        threading.Thread(
            target=self._retrain, args=(symbol, horizon), daemon=True
        ).start()

    def _retrain(self, symbol: str, horizon: str):
        try:
            meta = train(symbol, horizon)  # type: ignore
            if meta.get("retrain_skipped"):
                msg = (f"Existing model kept — new walk-forward score "
                       f"({meta['new_combined']:.4f}) did not beat saved model "
                       f"({meta['old_combined']:.4f}). Old model is better.\n")
            else:
                msg = (f"Retrained. New model saved. "
                       f"Walk-forward dir acc: {meta.get('wf_dir_acc', 0)*100:.1f}%  "
                       f"Val loss: {meta['val_loss']:.5f}\n")
            self.after(0, lambda: self._log_result(msg))
        except Exception as e:
            self.after(0, lambda: self._log_result(f"ERROR: {e}\n"))

    def _show_prediction(self, pred: Prediction, history):
        min_pct, max_pct = config.profit_goals_for(pred.horizon)
        min_target = pred.current_price * (1 + min_pct)
        max_target = pred.current_price * (1 + max_pct)
        period = config.goal_period_label(pred.horizon)

        # --- Position sizing ---
        positions = self.portfolio.get_positions()
        live_prices = {p.symbol: p.buy_price for p in positions}
        summary = self.portfolio.total_value(live_prices)
        sizing = suggest_size(
            equity=summary["equity"],
            current_invested=summary["invested"],
            cash=summary["cash"],
            current_price=pred.current_price,
            expected_return_pct=pred.expected_return_pct,
        )
        self._log_result(
            f"\n=== Prediction for {pred.symbol} ({pred.horizon}) ===\n"
            f"Current price       : Rs. {pred.current_price:.2f}\n"
            f"Predicted (end)     : Rs. {pred.predicted_price:.2f}\n"
            f"Expected return     : {pred.expected_return_pct:+.2f}%\n"
            f"Confidence (rough)  : {pred.confidence:.2f}\n"
            f"\n--- {period} goals if you buy now ---\n"
            f"MIN profit target   : Rs. {min_target:.2f}  (+{min_pct*100:.1f}%)\n"
            f"MAX profit target   : Rs. {max_target:.2f}  (+{max_pct*100:.1f}%)\n"
            f"Auto-sell rules     :  >= MAX  -> SELL (cap)\n"
            f"                       in [MIN, MAX) -> fresh prediction decides\n"
            f"                       < MIN  -> hold unless stop-loss\n"
            f"\n--- Position sizing ---\n"
            f"Equity              : Rs. {summary['equity']:,.2f}\n"
            f"Currently invested  : Rs. {summary['invested']:,.2f} "
            f"({sizing.capital_used_pct*100:.1f}% of equity)\n"
            f"Available budget    : Rs. {sizing.available_budget:,.2f} "
            f"(under {config.MAX_INVESTED_PCT*100:.0f}% cap)\n"
            f"Suggested quantity  : {sizing.suggested_qty} shares "
            f"= Rs. {sizing.suggested_rupees:,.2f} "
            f"({sizing.final_pct_of_equity*100:.2f}% of equity)\n"
            f"Sizing logic        : {sizing.reason}\n"
            f"\nForecast path       : "
            + ", ".join(f"{v:.2f}" for v in pred.predicted_path) + "\n"
        )

        # Replace chart
        for w in self.chart_holder.winfo_children():
            w.destroy()
        widget = make_price_chart(
            self.chart_holder, history,
            predicted_path=pred.predicted_path,
            title=f"{pred.symbol} — past 120 bars + {len(pred.predicted_path)}-step forecast")
        widget.pack(fill="both", expand=True)


    def _log_result(self, text: str):
        self.result_box.configure(state="normal")
        self.result_box.insert("end", text)
        self.result_box.see("end")
        self.result_box.configure(state="disabled")

    # ============================================================
    # Portfolio & Monitor tab
    # ============================================================
    def _build_monitor(self, root):
        top = ctk.CTkFrame(root)
        top.pack(fill="x", padx=12, pady=12)

        ctk.CTkButton(top, text="Refresh decisions",
                      command=self._refresh_monitor_threaded
                      ).pack(side="left", padx=6)

        ctk.CTkButton(top, text="Auto-sell all flagged",
                      fg_color="#b71c1c", hover_color="#7f0000",
                      command=self._auto_sell
                      ).pack(side="left", padx=6)

        self.monitor_box = ctk.CTkTextbox(
            root, font=ctk.CTkFont(family="Consolas", size=12))
        self.monitor_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._refresh_monitor()

    def _refresh_monitor_threaded(self):
        self.monitor_box.configure(state="normal")
        self.monitor_box.delete("1.0", "end")
        self.monitor_box.insert("end",
            "Evaluating positions (running LSTM for each — may take a moment)...\n")
        self.monitor_box.configure(state="disabled")
        threading.Thread(target=self._run_monitor, daemon=True).start()

    def _refresh_monitor(self):
        """Quick refresh without fresh predictions (just risk rules)."""
        self.monitor_box.configure(state="normal")
        self.monitor_box.delete("1.0", "end")
        positions = self.portfolio.get_positions()
        if not positions:
            self.monitor_box.insert("end",
                "No open positions. Go to 'Predict & Buy' to add one.\n")
            self.monitor_box.configure(state="disabled")
            return

        decisions = evaluate_all(self.portfolio, do_fresh_prediction=False)
        self._render_decisions(decisions)
        self.monitor_box.configure(state="disabled")

    def _run_monitor(self):
        try:
            decisions = evaluate_all(self.portfolio, do_fresh_prediction=True)
            self.after(0, lambda: self._render_decisions(decisions, clear=True))
        except Exception as e:
            self.after(0, lambda: self._render_decisions_error(str(e)))

    def _render_decisions(self, decisions, clear: bool = False):
        self.monitor_box.configure(state="normal")
        if clear:
            self.monitor_box.delete("1.0", "end")
        self._cached_decisions = decisions

        header = (f"\n{'Symbol':<12} {'Hor':<8} {'Action':<5} "
                  f"{'Buy':>9} {'Live':>9} {'Min↗':>9} {'Max↗':>9} "
                  f"{'P&L%':>7}  Reason\n")
        self.monitor_box.insert("end", header)
        self.monitor_box.insert("end", "-" * 130 + "\n")
        for d in decisions:
            color_tag = "sell" if d.action == "SELL" else "hold"
            line = (f"{d.symbol:<12} {d.horizon:<8} {d.action:<5} "
                    f"{d.buy_price:>9.2f} {d.live_price:>9.2f} "
                    f"{d.min_target_price:>9.2f} {d.max_target_price:>9.2f} "
                    f"{d.pnl_pct:>+7.2f}  {d.reason}\n")
            self.monitor_box.insert("end", line, color_tag)
        self.monitor_box.tag_config("sell", foreground="#ff6b6b")
        self.monitor_box.tag_config("hold", foreground="#69db7c")
        self.monitor_box.configure(state="disabled")

    def _render_decisions_error(self, err: str):
        self.monitor_box.configure(state="normal")
        self.monitor_box.insert("end", f"\nERROR: {err}\n")
        self.monitor_box.configure(state="disabled")

    def _auto_sell(self):
        decisions = getattr(self, "_cached_decisions", None)
        if not decisions:
            messagebox.showinfo("Nothing to sell",
                "Run 'Refresh decisions' first so the engine knows what to do.")
            return
        msgs = auto_execute(self.portfolio, decisions)
        if not msgs:
            messagebox.showinfo("No SELL flags",
                "All positions are in HOLD state right now.")
        else:
            messagebox.showinfo("Auto-sell complete", "\n".join(msgs))
        self._refresh_dashboard()
        self._refresh_monitor()

    # ============================================================
    # Agent tab (autonomous)
    # ============================================================
    def _build_agent(self, root):
        top = ctk.CTkFrame(root)
        top.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(top, text="Market Monitor & Agent",
                     font=ctk.CTkFont(size=16, weight="bold")
                     ).pack(side="left", padx=8)

        ctk.CTkLabel(top, text="Horizon:").pack(side="left", padx=(20, 4))
        self.agent_horizon = ctk.StringVar(value=config.AGENT_DEFAULT_HORIZON)
        ctk.CTkSegmentedButton(top, values=["swing", "intraday"],
                               variable=self.agent_horizon).pack(side="left", padx=4)

        self.run_cycle_btn = ctk.CTkButton(
            top, text="Run buy scan", fg_color="#1565c0",
            hover_color="#0d47a1", command=self._run_agent_threaded)
        self.run_cycle_btn.pack(side="left", padx=16)

        ctk.CTkButton(top, text="Train all (first run)",
                      command=self._train_all_threaded).pack(side="left", padx=4)
        ctk.CTkButton(top, text="Retrain stale models",
                      command=self._retrain_stale_threaded).pack(side="left", padx=4)

        # ── Universe selector ────────────────────────────────────────
        univ_frame = ctk.CTkFrame(root)
        univ_frame.pack(fill="x", padx=12, pady=(4, 0))

        ctk.CTkLabel(univ_frame,
                     text="Trading Universe",
                     font=ctk.CTkFont(weight="bold")
                     ).pack(side="left", padx=(8, 12))

        self._univ_var = ctk.StringVar(value=config.ACTIVE_UNIVERSE)
        for val, label in [
            ("nifty50", "Nifty 50 stocks"),
            ("funds",   "ETF / Funds"),
            ("both",    "Stocks + ETFs"),
        ]:
            ctk.CTkRadioButton(
                univ_frame, text=label,
                variable=self._univ_var, value=val,
                command=self._on_universe_change,
            ).pack(side="left", padx=6)

        ctk.CTkButton(
            univ_frame, text="ℹ Fund list", width=90,
            fg_color="transparent", border_width=1,
            command=self._show_fund_list,
        ).pack(side="left", padx=(12, 4))

        self._univ_status_lbl = ctk.CTkLabel(
            univ_frame, text="", text_color="#9aa4b2",
            font=ctk.CTkFont(size=11))
        self._univ_status_lbl.pack(side="left", padx=8)
        self._refresh_universe_label()

        # Auto-monitor row (always active while the app is open)
        auto = ctk.CTkFrame(root)
        auto.pack(fill="x", padx=12, pady=(4, 8))
        ctk.CTkLabel(auto, text="Auto-monitor:",
                     font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(8, 6))
        ctk.CTkLabel(auto, text="every").pack(side="left", padx=(4, 2))
        self.monitor_interval_entry = ctk.CTkEntry(auto, width=55)
        self.monitor_interval_entry.insert(0, str(config.AGENT_AUTOLOOP_MINUTES))
        self.monitor_interval_entry.pack(side="left", padx=2)
        ctk.CTkLabel(auto, text="min  |").pack(side="left", padx=(2, 6))
        ctk.CTkButton(auto, text="Check now", width=90,
                      command=self._manual_check_now).pack(side="left", padx=4)
        self.monitor_last_lbl = ctk.CTkLabel(
            auto, text="Last check: —", text_color="#9aa4b2")
        self.monitor_last_lbl.pack(side="left", padx=10)
        self.monitor_next_lbl = ctk.CTkLabel(
            auto, text="", text_color="#9aa4b2")
        self.monitor_next_lbl.pack(side="left", padx=4)

        self.cache_status_lbl = ctk.CTkLabel(
            auto, text="○ NSE cache: starting...", text_color="#9aa4b2")
        self.cache_status_lbl.pack(side="right", padx=12)

        # Progress
        self.agent_progress = ctk.CTkProgressBar(root)
        self.agent_progress.pack(fill="x", padx=14, pady=(0, 4))
        self.agent_progress.set(0)
        self.agent_status = ctk.CTkLabel(root, text="Idle.",
                                         text_color="#9aa4b2")
        self.agent_status.pack(anchor="w", padx=14)

        # Goal banner
        self.agent_goal_lbl = ctk.CTkLabel(
            root, text="", font=ctk.CTkFont(size=13, weight="bold"))
        self.agent_goal_lbl.pack(anchor="w", padx=14, pady=(6, 0))

        # Output log
        self.agent_box = ctk.CTkTextbox(
            root, font=ctk.CTkFont(family="Consolas", size=12))
        self.agent_box.pack(fill="both", expand=True, padx=12, pady=12)
        self._agent_log(
            "Auto-monitor is ON — open positions are evaluated every N minutes "
            "while the app is running. Decisions appear in Portfolio & Monitor.\n\n"
            "FIRST TIME: click 'Train all' once (can take 20-40 min). Models "
            "are saved and reused after that.\n"
            "Use 'Run buy scan' to score all Nifty 50 stocks and buy top picks.\n")
        self._update_agent_goal_banner()

    def _agent_log(self, text: str):
        self.agent_box.configure(state="normal")
        self.agent_box.insert("end", text)
        self.agent_box.see("end")
        self.agent_box.configure(state="disabled")

    def _agent_progress_cb(self, msg: str, done: int, total: int):
        def upd():
            self.agent_status.configure(text=f"{msg}  ({done}/{total})")
            self.agent_progress.set((done / total) if total else 0)
        self.after(0, upd)

    def _update_agent_goal_banner(self):
        positions = self.portfolio.get_positions()
        lp = {p.symbol: p.buy_price for p in positions}
        horizon = self.agent_horizon.get()
        g = self.portfolio.period_goal_status(horizon, lp)
        color = "#69db7c" if g["reached"] else "#e0e0e0"
        state = "REACHED - no new buys" if g["reached"] else "in progress"
        self.agent_goal_lbl.configure(
            text=(f"{g['label']} goal: Rs.{g['progress']:+,.0f} / "
                  f"Rs.{g['goal_amount']:,.0f} ({g['pct_of_goal']:.0f}%) - {state}"),
            text_color=color)

    # ----- universe helpers -----

    def _refresh_universe_label(self):
        u = config.ACTIVE_UNIVERSE
        universe = config.get_active_universe()
        trained  = len(universe) - len(untrained_symbols(
            self.agent_horizon.get() if hasattr(self, "agent_horizon")
            else config.AGENT_DEFAULT_HORIZON, universe))
        label = {"nifty50": "Nifty 50 stocks",
                 "funds":   "ETF/Funds", "both": "Stocks + ETFs"}.get(u, u)
        self._univ_status_lbl.configure(
            text=f"({label} — {len(universe)} symbols, "
                 f"{trained} models trained)")

    def _on_universe_change(self):
        new_val = self._univ_var.get()
        old_val = config.ACTIVE_UNIVERSE
        if new_val == old_val:
            return

        # Warn: switching universe means old models belong to different instruments
        horizon  = getattr(self, "agent_horizon",
                           ctk.StringVar(value=config.AGENT_DEFAULT_HORIZON)
                           ).get()
        new_universe = (NIFTY_50 if new_val == "nifty50"
                        else ALL_FUNDS if new_val == "funds"
                        else NIFTY_50 + ALL_FUNDS)
        missing = untrained_symbols(horizon, new_universe)

        config.ACTIVE_UNIVERSE = new_val

        if missing:
            messagebox.showwarning(
                "⚠ Retrain required",
                f"You switched to '{new_val}'.\n\n"
                f"{len(missing)} of {len(new_universe)} symbols have no trained model "
                f"for the '{horizon}' horizon.\n\n"
                "ACTION REQUIRED:\n"
                "  1. Click  'Train all'  to build models for the new universe.\n"
                "  2. Only then run the buy scan.\n\n"
                "Running a scan before training will skip untrained symbols "
                "or use stale models from the old universe.")
        else:
            messagebox.showinfo(
                "Universe changed",
                f"Switched to '{new_val}'.\n"
                f"All {len(new_universe)} symbols already have trained models. "
                "You can run the buy scan immediately.")

        self._refresh_universe_label()
        # Restart price cache for the new symbol set
        price_cache.stop()
        self.after(500, self._start_price_cache)

    def _show_fund_list(self):
        """Pop up a window listing all ETF categories and their symbols."""
        win = ctk.CTkToplevel(self)
        win.title("ETF / Fund Universe")
        win.geometry("620x540")
        win.grab_set()

        ctk.CTkLabel(win,
                     text="Available ETF Categories",
                     font=ctk.CTkFont(size=15, weight="bold")
                     ).pack(pady=(16, 4), padx=16, anchor="w")
        ctk.CTkLabel(win,
                     text="ETFs are baskets of stocks — far safer than individual stocks.\n"
                          "Select 'ETF / Funds' in the universe picker to trade these.",
                     text_color="#9aa4b2", font=ctk.CTkFont(size=11)
                     ).pack(padx=16, anchor="w", pady=(0, 8))

        box = ctk.CTkTextbox(win, font=ctk.CTkFont(family="Consolas", size=11))
        box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        box.configure(state="normal")

        for cat, symbols in FUND_CATEGORIES.items():
            box.insert("end", f"\n{'─'*60}\n")
            box.insert("end", f"  {cat}\n")
            box.insert("end", f"{'─'*60}\n")
            for sym in symbols:
                box.insert("end", f"  {sym}\n")

        box.configure(state="disabled")
        ctk.CTkButton(win, text="Close", command=win.destroy).pack(pady=(0, 12))

    def _train_all_threaded(self):
        horizon  = self.agent_horizon.get()
        universe = config.get_active_universe()
        missing  = untrained_symbols(horizon, universe)

        if not missing:
            messagebox.showinfo(
                "Already trained",
                f"All {len(universe)} models for '{horizon}' "
                f"({config.ACTIVE_UNIVERSE}) already exist.\n\n"
                "Nothing to do. Run the buy scan when ready.")
            return
        if not messagebox.askyesno(
                "Train all",
                f"Train {len(missing)} missing '{horizon}' models "
                f"for universe: {config.ACTIVE_UNIVERSE}?\n\n"
                f"• {len(universe) - len(missing)} already trained (will be skipped)\n"
                f"• {len(missing)} need training (~30–60 s each on CPU)\n\n"
                "You only need to do this once per universe. Models are saved "
                "permanently and reused across sessions."):
            return
        self.run_cycle_btn.configure(state="disabled")
        self._agent_log(
            f"\n--- Training {len(missing)} models "
            f"({horizon}, universe: {config.ACTIVE_UNIVERSE}) ---\n")
        threading.Thread(target=self._train_all, args=(horizon, universe),
                         daemon=True).start()

    def _train_all(self, horizon, universe=None):
        try:
            res = train_universe(horizon=horizon, universe=universe,
                                 only_missing=True,
                                 progress=self._agent_progress_cb)
            msg = (f"\nTraining done. Trained {len(res['trained'])}, "
                   f"skipped {len(res['skipped'])}, "
                   f"failed {len(res['failed'])}.\n")
            if res["failed"]:
                msg += "Failed: " + ", ".join(
                    f"{s}" for s, _ in res["failed"][:10]) + "\n"
            self.after(0, lambda: self._agent_log(msg))
        except Exception as e:
            self.after(0, lambda: self._agent_log(f"\nTraining error: {e}\n"))
        finally:
            self.after(0, lambda: self.run_cycle_btn.configure(state="normal"))
            self.after(0, lambda: self.agent_progress.set(0))
            self.after(0, lambda: self.agent_status.configure(text="Idle."))

    def _retrain_stale_threaded(self):
        self._agent_log("\n--- Checking for stale models to retrain ---\n")
        threading.Thread(target=self._retrain_stale, daemon=True).start()

    def _retrain_stale(self):
        try:
            res = retrain_stale(progress=self._agent_progress_cb)
            if res["checked"] == 0:
                self.after(0, lambda: self._agent_log(
                    "No models flagged as inaccurate yet (need resolved "
                    "predictions first).\n"))
            else:
                msg = (f"Retrained {len(res['retrained'])} of {res['checked']} "
                       f"flagged models (MAPE > {config.RETRAIN_ERROR_THRESHOLD_PCT}%)."
                       f" New walk-forward models saved for all.")
                self.after(0, lambda m=msg: self._agent_log(m + "\n"))
        except Exception as e:
            self.after(0, lambda: self._agent_log(f"Retrain error: {e}\n"))
        finally:
            self.after(0, lambda: self.agent_progress.set(0))
            self.after(0, lambda: self.agent_status.configure(text="Idle."))

    def _run_agent_threaded(self):
        """Manual single cycle (triggered by the button)."""
        if self.cycle_busy:
            messagebox.showinfo("Busy", "A cycle is already running.")
            return
        if self.mode.get() == "real":
            messagebox.showwarning("Paper only",
                "The agent runs in paper mode only. Real trading is disabled.")
            return

        horizon  = self.agent_horizon.get()
        universe = config.get_active_universe()
        missing  = untrained_symbols(horizon, universe)

        # ── Fail-safe 1: no models at all ────────────────────────────
        if len(missing) == len(universe):
            messagebox.showerror(
                "❌ Step 1 missing — train models first",
                f"No models trained for the current universe "
                f"({config.ACTIVE_UNIVERSE}, {len(universe)} symbols).\n\n"
                "REQUIRED SEQUENCE:\n"
                "  Step 1 → Click 'Train all'  (20–40 min, done once)\n"
                "  Step 2 → Run buy scan\n\n"
                "Without trained models the agent has nothing to predict with.")
            return

        # ── Fail-safe 2: some models missing (universe changed?) ─────
        if len(missing) > len(universe) * 0.3:   # >30% untrained
            if not messagebox.askyesno(
                "⚠ Caution — many symbols untrained",
                f"{len(missing)} of {len(universe)} symbols in the current universe "
                f"({config.ACTIVE_UNIVERSE}) have no trained model.\n\n"
                "Did you recently switch universes without retraining?\n\n"
                "• Click  'No'  → go train first ('Train all')\n"
                "• Click  'Yes' → continue anyway (untrained symbols will be skipped)\n\n"
                "For best results, train all symbols before scanning."):
                return

        # ── Fail-safe 3: price cache not running ─────────────────────
        if not price_cache.is_running():
            messagebox.showwarning(
                "⚠ Price cache offline",
                "The live price cache is not running.\n\n"
                "Prices may be stale (up to 15 min delay from Yahoo Finance).\n"
                "Check your internet connection. The cache starts automatically "
                "on launch — restarting the app should fix this.\n\n"
                "The scan will continue with best-available prices.")

        self._start_cycle(horizon)

    def _start_cycle(self, horizon: str):
        """Shared cycle launcher with a busy-guard (used by manual + auto)."""
        if self.cycle_busy:
            return
        self.cycle_busy = True
        self.run_cycle_btn.configure(state="disabled")
        self._agent_log(f"\n=== Running agent cycle ({horizon}) ===\n")
        threading.Thread(target=self._run_agent, args=(horizon,),
                         daemon=True).start()

    def _run_agent(self, horizon):
        try:
            report = self.agent.run_cycle(
                horizon=horizon, progress=self._agent_progress_cb,
                mode=self.mode.get())
            self.after(0, lambda: self._render_agent_report(report))
        except Exception as e:
            self.after(0, lambda: self._agent_log(f"Agent error: {e}\n"))
        finally:
            self.after(0, self._cycle_finished)

    def _cycle_finished(self):
        self.cycle_busy = False
        self.run_cycle_btn.configure(state="normal")
        self.agent_progress.set(0)
        self.agent_status.configure(text="Idle.")

    # ----- auto-loop -----

    def _start_auto_monitor(self):
        """Start the continuous position monitor — runs as long as the app is open."""
        self._schedule_next_monitor(delay_ms=15_000)  # first check 15 s after open

    def _schedule_next_monitor(self, delay_ms: int | None = None):
        if self._monitor_after_id is not None:
            try:
                self.after_cancel(self._monitor_after_id)
            except Exception:
                pass
        if delay_ms is None:
            try:
                minutes = float(self.monitor_interval_entry.get())
                if minutes <= 0:
                    minutes = config.AGENT_AUTOLOOP_MINUTES
            except Exception:
                minutes = config.AGENT_AUTOLOOP_MINUTES
            delay_ms = int(minutes * 60_000)
        next_min = round(delay_ms / 60_000, 1)
        self._monitor_after_id = self.after(delay_ms, self._monitor_tick)
        try:
            self.monitor_next_lbl.configure(
                text=f"Next check in {next_min:g} min")
        except Exception:
            pass

    def _monitor_tick(self):
        self._monitor_after_id = None
        if self.cycle_busy:
            self._schedule_next_monitor()
            return
        positions = self.portfolio.get_positions()
        if positions:
            self.cycle_busy = True
            import datetime as _dt
            now_str = _dt.datetime.now().strftime("%H:%M")
            try:
                self.monitor_last_lbl.configure(text=f"Last check: {now_str}")
                self.agent_status.configure(
                    text="Auto-monitor: evaluating positions...")
            except Exception:
                pass
            threading.Thread(target=self._do_auto_monitor, daemon=True).start()
        self._schedule_next_monitor()

    def _do_auto_monitor(self):
        try:
            decisions = evaluate_all(self.portfolio, do_fresh_prediction=True)
            self.after(0, lambda: self._on_auto_monitor_done(decisions))
        except Exception as e:
            self.after(0, lambda: self.agent_status.configure(
                text=f"Monitor error: {e}"))
        finally:
            self.after(0, lambda: setattr(self, "cycle_busy", False))
            self.after(0, lambda: self.agent_status.configure(text="Idle."))

    def _on_auto_monitor_done(self, decisions):
        self._render_decisions(decisions, clear=True)
        self._refresh_dashboard()
        sells = [d for d in decisions if d.action == "SELL"]
        if sells:
            self._agent_log(
                f"[Auto-monitor] SELL recommended for: "
                + ", ".join(d.symbol for d in sells)
                + "\n  -> Go to Portfolio & Monitor tab to review and execute.\n")

    def _manual_check_now(self):
        """Trigger an immediate position check, resetting the timer."""
        if self._monitor_after_id is not None:
            try:
                self.after_cancel(self._monitor_after_id)
            except Exception:
                pass
            self._monitor_after_id = None
        self._monitor_tick()

    def _start_price_cache(self):
        if config.PRICE_CACHE_INTERVAL_SECS <= 0:
            return
        price_cache.on_update = self._on_cache_update
        # No symbol list passed — cache imports NIFTY_50 directly from nifty50.py,
        # the same source used by the trainer, so they always stay in sync.
        price_cache.start(interval=config.PRICE_CACHE_INTERVAL_SECS)

    def _on_cache_update(self, updated: int, total: int):
        """Called by the cache thread after each scrape cycle."""
        def upd():
            try:
                st = price_cache.status()
                ts = st["last_run"].strftime("%H:%M:%S") if st["last_run"] else "—"
                dot = "●" if updated > 0 else "○"
                self.cache_status_lbl.configure(
                    text=f"{dot} NSE cache: {updated}/{total}  {ts}",
                    text_color="#69db7c" if updated > 0 else "#ff6b6b")
            except Exception:
                pass
        self.after(0, upd)

    def _on_close(self):
        price_cache.stop()
        if self._monitor_after_id is not None:
            try:
                self.after_cancel(self._monitor_after_id)
            except Exception:
                pass
        self.destroy()

    def _render_agent_report(self, report):
        g = report.goal_status
        lines = [f"\nResult: {report.summary()}"]
        if g:
            lines.append(
                f"{g['label']} goal: Rs.{g['progress']:+,.0f} / "
                f"Rs.{g['goal_amount']:,.0f} "
                f"({'REACHED' if g['reached'] else 'in progress'})")
        if report.resolved_predictions:
            lines.append(f"Resolved {report.resolved_predictions} past predictions.")

        if report.sells:
            lines.append("\nSELLS:")
            lines += [f"  {m}" for m in report.sells]
        if report.buys:
            lines.append("\nBUYS:")
            lines += [f"  {m}" for m in report.buys]
        if not report.buys and not report.goal_reached:
            lines.append("\nNo new buys this cycle (nothing cleared the bar).")

        # Top 8 ranked candidates with their score breakdown
        if report.ranked:
            lines.append("\nTop candidates:")
            for sc in report.ranked[:8]:
                lines.append(f"  {sc.symbol:<12} {sc.explain()}")

        if report.messages:
            lines.append("\nNotes:")
            lines += [f"  {m}" for m in report.messages]

        self._agent_log("\n".join(lines) + "\n")
        self._update_agent_goal_banner()
        self._refresh_dashboard()
        self._refresh_ledger()

    # ============================================================
    # Ledger tab
    # ============================================================
    def _build_ledger(self, root):
        top = ctk.CTkFrame(root)
        top.pack(fill="x", padx=12, pady=12)

        ctk.CTkButton(top, text="Refresh",
                      command=self._refresh_ledger).pack(side="left", padx=6)
        ctk.CTkButton(top, text="Export CSV",
                      command=self._export_csv).pack(side="left", padx=6)
        ctk.CTkButton(top, text="Generate report (.md)",
                      command=self._export_report).pack(side="left", padx=6)
        ctk.CTkButton(top, text="Clear Ledger",
                      fg_color="#b71c1c", hover_color="#7f0000",
                      command=self._clear_ledger).pack(side="left", padx=6)

        self.ledger_summary = ctk.CTkLabel(
            root, text="", justify="left",
            font=ctk.CTkFont(family="Consolas", size=12))
        self.ledger_summary.pack(anchor="w", padx=16, pady=(0, 4))

        self.ledger_box = ctk.CTkTextbox(
            root, font=ctk.CTkFont(family="Consolas", size=12))
        self.ledger_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._refresh_ledger()

    def _refresh_ledger(self):
        import datetime as dt

        def fmt_ts(ts: str) -> str:
            try:
                return dt.datetime.fromisoformat(ts).strftime("%d-%b-%Y %H:%M")
            except Exception:
                return ts[:19]

        led = Ledger(self.portfolio.db_path)
        s = led.stats()
        pf = ("inf" if s.profit_factor == float("inf")
              else f"{s.profit_factor:.2f}")
        self.ledger_summary.configure(text=(
            f"Trades: {s.n_trades} ({s.n_buys} buy / {s.n_sells} sell)   "
            f"Win rate: {s.win_rate:.1f}%   Net P&L: Rs.{s.net_pnl:+,.2f}   "
            f"Profit factor: {pf}\n"
            f"Avg win: Rs.{s.avg_win:+,.2f}   Avg loss: Rs.{s.avg_loss:+,.2f}   "
            f"Best: Rs.{s.best:+,.2f}   Worst: Rs.{s.worst:+,.2f}"))

        self.ledger_box.configure(state="normal")
        self.ledger_box.delete("1.0", "end")
        header = (f"{'#':>4} {'Date & Time':<18} {'Side':<5} {'Symbol':<12} "
                  f"{'Qty':>5} {'Price':>10} {'P&L':>11}  Reason\n")
        self.ledger_box.insert("end", header)
        self.ledger_box.insert("end", "-" * 120 + "\n")
        for r in reversed(led.all_trades()):
            tag = "sell" if r.side == "SELL" else "buy"
            self.ledger_box.insert("end",
                f"{r.id:>4} {fmt_ts(r.timestamp):<18} {r.side:<5} {r.symbol:<12} "
                f"{r.quantity:>5} {r.price:>10.2f} {r.pnl:>+11.2f}  {r.reason}\n",
                tag)
        self.ledger_box.tag_config("sell", foreground="#ffb86b")
        self.ledger_box.tag_config("buy", foreground="#8be9fd")
        self.ledger_box.configure(state="disabled")

    def _export_csv(self):
        path = os.path.join(EXPORT_DIR, "trade_ledger.csv")
        try:
            Ledger(self.portfolio.db_path).export_csv(path)
            messagebox.showinfo("Exported", f"Ledger saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _export_report(self):
        path = os.path.join(EXPORT_DIR, "trading_report.md")
        try:
            Ledger(self.portfolio.db_path).export_report(path)
            messagebox.showinfo("Report saved", f"Report saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Report failed", str(e))

    def _clear_ledger(self):
        if messagebox.askyesno(
                "Clear Ledger",
                "Delete all trade records from the ledger?\n"
                "Open positions and wallet balance are NOT affected.\n"
                "This cannot be undone."):
            self.portfolio.clear_trades()
            self._refresh_ledger()
            self._refresh_dashboard()
            messagebox.showinfo("Cleared", "Trade ledger cleared.")

    # ============================================================
    # Settings tab
    # ============================================================
    def _build_settings(self, root):
        # Scrollable container so every section is always reachable
        scroll = ctk.CTkScrollableFrame(root)
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        # ---- Risk / profit settings ----
        f = ctk.CTkFrame(scroll)
        f.pack(fill="x", padx=16, pady=(16, 8))

        # Parameter definitions: (label, default_value, info_text)
        _INFO = {
            "Stop-loss %": (
                "WHAT:  Sell immediately if price falls this % below your buy price.\n\n"
                "WHY:   Stops a small loss from becoming a big one. Without this, a "
                "position that goes down 10% may never recover in your holding period.\n\n"
                "RECOMMENDED:\n"
                "  • Individual stocks: 3–5%  (stocks can be volatile)\n"
                "  • ETFs / Nifty funds: 2–3%  (ETFs are smoother)\n\n"
                "CAUTION: Setting too tight (< 1%) = getting stopped out on normal "
                "daily noise. Too loose (> 7%) = taking large losses before exit."
            ),
            "Trailing stop %": (
                "WHAT:  The stop-loss that follows the price UP as you profit.\n\n"
                "HOW:   Buy Rs.100 → price rises to Rs.120 → stop moves to Rs.116.40 "
                "(120 × (1 − 3%)). If price then falls to Rs.116.40 → auto-sell.\n\n"
                "WHY:   A fixed stop-loss lets you give back ALL your gains before "
                "selling. A trailing stop LOCKS IN profits as the price rises.\n\n"
                "RECOMMENDED: Same as stop-loss % or slightly tighter (e.g. 2.5%)."
            ),
            "Intraday MIN profit %": (
                "WHAT:  Below this P&L%, the position is always held (intraday).\n\n"
                "WHY:   Prevents selling too early on a stock that just started moving. "
                "Think of MIN as your 'good enough' target — you're happy if you exit "
                "anywhere above this.\n\n"
                "RECOMMENDED: 1.0–1.5% for intraday (4-hour hold).\n\n"
                "Between MIN and MAX: the LSTM decides — positive forecast = hold, "
                "negative = sell to lock the gain."
            ),
            "Intraday MAX profit %": (
                "WHAT:  Auto-sell when P&L reaches this % (intraday greed cap).\n\n"
                "WHY:   Markets reverse. A stock up 3% intraday can easily give it all "
                "back in the next hour. Lock in the gain before that happens.\n\n"
                "RECOMMENDED: 2.5–4% for intraday.\n\n"
                "Must be higher than Intraday MIN. If set too high (> 5% intraday) "
                "you'll rarely hit it — the price reverses before getting there."
            ),
            "Swing MIN profit %": (
                "WHAT:  Below this P&L%, the position is held for swing trades (multi-day).\n\n"
                "WHY:   A swing trade needs room to breathe across several days. Selling "
                "at 1% profit after 3 days barely covers brokerage and doesn't justify "
                "the time your capital was tied up.\n\n"
                "RECOMMENDED:\n"
                "  • Individual stocks: 2.5–4%\n"
                "  • ETFs: 1.5–2.5%  (ETFs move more slowly)\n\n"
                "Keep MIN < MAX. A typical swing aim is +3% MIN, +6% MAX."
            ),
            "Swing MAX profit %": (
                "WHAT:  Auto-sell at this P&L% for swing trades.\n\n"
                "WHY:   Swing trades held past their target become gambles. Booking "
                "profit at the MAX cap is the disciplined choice.\n\n"
                "RECOMMENDED:\n"
                "  • Individual stocks: 5–8%\n"
                "  • ETFs: 3–5%  (ETFs trend more slowly)\n\n"
                "Must be higher than Swing MIN."
            ),
            "Min buy edge %": (
                "WHAT:  The minimum % gain the LSTM must predict before the agent "
                "will consider buying.\n\n"
                "WHY:   If the model predicts only +0.3%, brokerage + slippage + "
                "model error easily wipe out the gain. Only buy when the predicted "
                "edge is meaningfully positive.\n\n"
                "RECOMMENDED:\n"
                "  • Intraday: ≥ 1.0%\n"
                "  • Swing: ≥ 1.5%\n\n"
                "Note: swing and intraday thresholds are in config.py as "
                "SWING_MIN_LSTM_EDGE_PCT and INTRADAY_MIN_LSTM_EDGE_PCT. "
                "This setting is the legacy display value."
            ),
            "Min hold edge %": (
                "WHAT:  In the profit zone [MIN, MAX], if the fresh LSTM forecast "
                "drops below this %, sell and lock the gain.\n\n"
                "WHY:   A positive position with a freshly-negative forecast is about "
                "to reverse. This setting determines how bearish the model must "
                "turn before you exit.\n\n"
                "RECOMMENDED: 0.0–0.5%  (0 = sell if ANY negative forecast; "
                "0.5 = only sell if forecast is meaningfully negative)."
            ),
            "Max invested % (cap)": (
                "WHAT:  Never invest more than this % of your total equity at once. "
                "The rest stays as cash reserve.\n\n"
                "WHY:   If 100% of your capital is in stocks and the market crashes, "
                "you have no cash to buy the dip. A 25% cash reserve also covers "
                "margin calls, brokerage, and unexpected expenses.\n\n"
                "RECOMMENDED:\n"
                "  • Conservative: 60–65%\n"
                "  • Moderate: 70–75% (default)\n"
                "  • Aggressive: 80–85% (not recommended for beginners)\n\n"
                "Lower = safer. Never set above 90%."
            ),
            "Max position %": (
                "WHAT:  Maximum % of equity in any single stock or ETF.\n\n"
                "WHY:   If one stock crashes and it's 40% of your portfolio, "
                "you lose 40% of your capital in one trade. Spreading across "
                "stocks limits damage from any single bad call.\n\n"
                "RECOMMENDED:\n"
                "  • Individual stocks: 8–12%  (high single-stock risk)\n"
                "  • ETFs: 15–20%  (ETFs are diversified internally)\n\n"
                "Lower = more diversified = safer."
            ),
            "Allocation scale (x edge)": (
                "WHAT:  Multiplier that converts predicted edge into position size.\n"
                "Formula: target_pct = predicted_edge% × this_scale\n\n"
                "EXAMPLE with scale = 4:\n"
                "  • Predicted edge +1% → 4% of equity allocated\n"
                "  • Predicted edge +2% → 8% of equity allocated\n"
                "  • Predicted edge +2.5% → 10% (hits max position cap)\n\n"
                "WHY:   Higher-confidence predictions get bigger positions — "
                "but still capped by 'Max position %'.\n\n"
                "RECOMMENDED: 3–5. Lower (2) = very conservative sizing. "
                "Higher (6+) = aggressive, more concentrated positions."
            ),
            "Min allocation %": (
                "WHAT:  If the calculated position size is below this % of equity, "
                "skip the trade entirely.\n\n"
                "WHY:   Very small positions (e.g. 0.5% of Rs.1 lakh = Rs.500) "
                "barely move your portfolio in profit but still consume brokerage "
                "and attention. Not worth the overhead.\n\n"
                "RECOMMENDED: 2–3%. Below 2% is generally not worth trading."
            ),
        }

        rows = [
            ("Stop-loss %",               f"{config.STOP_LOSS_PCT*100:.1f}"),
            ("Intraday MIN profit %",      f"{config.INTRADAY_MIN_PROFIT_PCT*100:.1f}"),
            ("Intraday MAX profit %",      f"{config.INTRADAY_MAX_PROFIT_PCT*100:.1f}"),
            ("Swing MIN profit %",         f"{config.SWING_MIN_PROFIT_PCT*100:.1f}"),
            ("Swing MAX profit %",         f"{config.SWING_MAX_PROFIT_PCT*100:.1f}"),
            ("Min buy edge %",             f"{config.MIN_PREDICTION_EDGE*100:.1f}"),
            ("Min hold edge %",            f"{config.MIN_HOLD_CONFIDENCE*100:.1f}"),
            ("Max invested % (cap)",       f"{config.MAX_INVESTED_PCT*100:.1f}"),
            ("Max position %",             f"{config.MAX_POSITION_PCT*100:.1f}"),
            ("Allocation scale (x edge)",  f"{config.ALLOCATION_SCALE:.1f}"),
            ("Min allocation %",           f"{config.MIN_ALLOCATION_PCT*100:.1f}"),
        ]
        self.settings_entries = {}
        for i, (label, val) in enumerate(rows):
            ctk.CTkLabel(f, text=label).grid(row=i, column=0,
                                             sticky="w", padx=8, pady=6)
            e = ctk.CTkEntry(f, width=120)
            e.insert(0, val)
            e.grid(row=i, column=1, padx=8, pady=6)
            self.settings_entries[label] = e
            # ℹ info button
            info_text = _INFO.get(label, "No description available.")
            ctk.CTkButton(
                f, text="ℹ", width=26, height=26,
                fg_color="transparent", border_width=1,
                font=ctk.CTkFont(size=12),
                command=lambda t=label, m=info_text: messagebox.showinfo(
                    f"About: {t}", m)
            ).grid(row=i, column=2, padx=(2, 8), pady=6)

        ctk.CTkLabel(f, text="Tip: MIN = lock-in floor, MAX = greed cap. "
                             "P&L between them is decided by a fresh prediction.",
                     text_color="#9aa4b2", font=ctk.CTkFont(size=11)
                     ).grid(row=len(rows), column=0, columnspan=3,
                            sticky="w", padx=8, pady=(8, 0))
        ctk.CTkButton(f, text="Apply settings",
                      command=self._apply_settings
                      ).grid(row=len(rows)+1, column=0, columnspan=3,
                             pady=12, sticky="ew")

        # ---- Total capital ----
        cap = ctk.CTkFrame(scroll)
        cap.pack(fill="x", padx=16, pady=8)
        ctk.CTkLabel(cap, text="Total capital available to invest",
                     font=ctk.CTkFont(weight="bold")
                     ).pack(anchor="w", padx=8, pady=(8, 0))
        cap_row = ctk.CTkFrame(cap, fg_color="transparent")
        cap_row.pack(fill="x", padx=8, pady=8)
        ctk.CTkLabel(cap_row, text="Rs.").pack(side="left", padx=(0, 4))
        self.capital_entry = ctk.CTkEntry(cap_row, width=160)
        self.capital_entry.insert(0, f"{config.TOTAL_INVESTMENT_AMOUNT:.0f}")
        self.capital_entry.pack(side="left", padx=4)
        ctk.CTkButton(cap_row, text="Set total capital",
                      command=self._set_capital).pack(side="left", padx=8)
        ctk.CTkLabel(cap, text="Note: setting capital resets the wallet "
                               "(wipes positions & trades) and refunds it with "
                               "this amount.",
                     text_color="#9aa4b2", font=ctk.CTkFont(size=11)
                     ).pack(anchor="w", padx=8, pady=(0, 8))

        # ---- Danger zone ----
        danger = ctk.CTkFrame(scroll)
        danger.pack(fill="x", padx=16, pady=8)
        ctk.CTkLabel(danger, text="Danger zone",
                     font=ctk.CTkFont(weight="bold")
                     ).pack(anchor="w", padx=8, pady=(8, 0))
        ctk.CTkButton(danger, text="Reset paper wallet",
                      fg_color="#b71c1c", hover_color="#7f0000",
                      command=self._reset_wallet
                      ).pack(anchor="w", padx=8, pady=8)

        # ---- Saved Models ----
        mdl = ctk.CTkFrame(scroll)
        mdl.pack(fill="x", padx=16, pady=(8, 20))

        mdl_header = ctk.CTkFrame(mdl, fg_color="transparent")
        mdl_header.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(mdl_header, text="Saved Models",
                     font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(mdl_header, text="Refresh list", width=100,
                      command=self._refresh_model_list).pack(side="right", padx=4)
        ctk.CTkButton(mdl_header, text="Clear all models",
                      fg_color="#b71c1c", hover_color="#7f0000", width=130,
                      command=self._clear_all_models).pack(side="right", padx=4)

        self.model_list_box = ctk.CTkTextbox(
            mdl, font=ctk.CTkFont(family="Consolas", size=11), height=260)
        self.model_list_box.pack(fill="x", padx=8, pady=(0, 8))
        self._refresh_model_list()

    def _apply_settings(self):
        try:
            config.STOP_LOSS_PCT             = float(self.settings_entries["Stop-loss %"].get())/100
            config.INTRADAY_MIN_PROFIT_PCT   = float(self.settings_entries["Intraday MIN profit %"].get())/100
            config.INTRADAY_MAX_PROFIT_PCT   = float(self.settings_entries["Intraday MAX profit %"].get())/100
            config.SWING_MIN_PROFIT_PCT      = float(self.settings_entries["Swing MIN profit %"].get())/100
            config.SWING_MAX_PROFIT_PCT      = float(self.settings_entries["Swing MAX profit %"].get())/100
            config.MIN_PREDICTION_EDGE       = float(self.settings_entries["Min buy edge %"].get())/100
            config.MIN_HOLD_CONFIDENCE       = float(self.settings_entries["Min hold edge %"].get())/100
            config.MAX_INVESTED_PCT          = float(self.settings_entries["Max invested % (cap)"].get())/100
            config.MAX_POSITION_PCT          = float(self.settings_entries["Max position %"].get())/100
            config.ALLOCATION_SCALE          = float(self.settings_entries["Allocation scale (x edge)"].get())
            config.MIN_ALLOCATION_PCT        = float(self.settings_entries["Min allocation %"].get())/100

            # Sanity checks
            for hor, (mn, mx) in [
                ("intraday", (config.INTRADAY_MIN_PROFIT_PCT,
                              config.INTRADAY_MAX_PROFIT_PCT)),
                ("swing",    (config.SWING_MIN_PROFIT_PCT,
                              config.SWING_MAX_PROFIT_PCT)),
            ]:
                if mn >= mx:
                    raise ValueError(
                        f"{hor.title()} MIN ({mn*100:.1f}%) must be less than "
                        f"MAX ({mx*100:.1f}%).")

            messagebox.showinfo("Saved", "Settings applied for this session.")
            self._refresh_dashboard()
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e))

    def _set_capital(self):
        try:
            amount = float(self.capital_entry.get())
            if amount <= 0:
                raise ValueError("Capital must be positive.")
        except ValueError as e:
            messagebox.showerror("Invalid amount", str(e))
            return
        if not messagebox.askyesno(
                "Set total capital",
                f"Set total capital to Rs. {amount:,.0f}?\n"
                "This wipes all positions & trades and refunds the wallet."):
            return
        self.portfolio.set_total_capital(amount)
        self._refresh_dashboard()
        self._refresh_monitor()
        self._refresh_ledger()
        self._update_agent_goal_banner()
        messagebox.showinfo("Capital set",
                            f"Total capital is now Rs. {amount:,.0f}.")

    def _reset_wallet(self):
        if messagebox.askyesno(
                "Reset wallet",
                f"Wipe all trades & positions, restart with Rs. "
                f"{config.TOTAL_INVESTMENT_AMOUNT:,.0f}?"):
            self.portfolio.reset(cash=config.TOTAL_INVESTMENT_AMOUNT)
            self._refresh_dashboard()
            self._refresh_monitor()
            self._refresh_ledger()
            messagebox.showinfo("Reset", "Paper wallet reset.")

    def _refresh_model_list(self):
        import json
        import datetime as dt

        box = self.model_list_box
        box.configure(state="normal")
        box.delete("1.0", "end")

        model_dir = config.MODEL_DIR
        if not os.path.isdir(model_dir):
            box.insert("end", "No models directory found — train a model first.\n")
            box.configure(state="disabled")
            return

        pt_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pt"))
        if not pt_files:
            box.insert("end", "No trained models yet. Use 'Train all' in the Agent tab.\n")
            box.configure(state="disabled")
            return

        header = (f"{'Symbol':<20} {'Horizon':<10} {'Dir Acc':>8} "
                  f"{'Val Loss':>10} {'Size':>8}  Trained\n")
        box.insert("end", header)
        box.insert("end", "─" * 82 + "\n")

        total_size = 0
        for fname in pt_files:
            fpath = os.path.join(model_dir, fname)
            size_bytes = os.path.getsize(fpath)
            total_size += size_bytes
            mtime = dt.datetime.fromtimestamp(
                os.path.getmtime(fpath)).strftime("%d-%b-%Y %H:%M")

            # Parse symbol + horizon from filename e.g. RELIANCE_NS_swing.pt
            stem = fname[:-3]
            if stem.endswith("_swing"):
                horizon = "swing"
                raw_sym = stem[:-6]
            elif stem.endswith("_intraday"):
                horizon = "intraday"
                raw_sym = stem[:-9]
            else:
                horizon = "?"
                raw_sym = stem
            # Restore dots: _NS -> .NS, _BO -> .BO, and -> &
            symbol = (raw_sym
                      .replace("_NS", ".NS")
                      .replace("_BO", ".BO")
                      .replace("and", "&"))

            # Pull accuracy stats from companion .json meta file
            meta_path = os.path.join(model_dir, fname.replace(".pt", ".json"))
            dir_acc_str = "—"
            val_loss_str = "—"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                    if "wf_dir_acc" in meta:
                        dir_acc_str = f"{meta['wf_dir_acc']*100:.1f}%"
                    if "val_loss" in meta:
                        val_loss_str = f"{meta['val_loss']:.5f}"
                except Exception:
                    pass

            line = (f"{symbol:<20} {horizon:<10} {dir_acc_str:>8} "
                    f"{val_loss_str:>10} {size_bytes/1024:>6.0f} KB  {mtime}\n")
            box.insert("end", line)

        box.insert("end", "─" * 82 + "\n")
        box.insert("end",
                   f"{len(pt_files)} models  |  "
                   f"Total size: {total_size/1024/1024:.1f} MB\n")
        box.configure(state="disabled")

    def _clear_all_models(self):
        model_dir = config.MODEL_DIR
        if not os.path.isdir(model_dir):
            messagebox.showinfo("No models", "No model directory found.")
            return

        files = [f for f in os.listdir(model_dir)
                 if f.endswith(".pt") or f.endswith(".json")]
        n_models = sum(1 for f in files if f.endswith(".pt"))
        if n_models == 0:
            messagebox.showinfo("No models", "No trained models to delete.")
            return

        if not messagebox.askyesno(
                "Clear all models",
                f"Delete all {n_models} trained models (weights + metadata)?\n\n"
                "The next prediction will retrain from scratch — this can take "
                "20-40 min per stock.\n\nThis cannot be undone."):
            return

        deleted, failed = 0, 0
        for fname in files:
            try:
                os.remove(os.path.join(model_dir, fname))
                deleted += 1
            except Exception:
                failed += 1

        self._refresh_model_list()
        msg = f"Deleted {deleted} files."
        if failed:
            msg += f"\n{failed} files could not be deleted (in use?)."
        messagebox.showinfo("Models cleared", msg)

    # ============================================================
    # Mode switch
    # ============================================================
    def _on_mode_change(self, choice):
        if choice == "real":
            messagebox.showwarning(
                "Real-money mode not configured",
                "Real-money trading is intentionally NOT implemented yet.\n\n"
                "Use paper trading until you trust the predictions.\n"
                "When ready, implement broker_interface.RealBrokerAdapter "
                "with your Zerodha/Upstox API credentials.")
            self.mode.set("paper")


def launch():
    app = StockTraderApp()
    app.mainloop()
