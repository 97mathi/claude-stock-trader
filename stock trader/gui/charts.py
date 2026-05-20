"""
Matplotlib chart helpers — embedded inside CustomTkinter frames.
"""

from __future__ import annotations
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def make_price_chart(parent, history: pd.DataFrame,
                     predicted_path: list[float] | None = None,
                     title: str = ""):
    """
    Return a tk widget showing past closes + (optional) forecast path.
    `parent` is a CTkFrame or similar.
    """
    fig = Figure(figsize=(7, 3.6), dpi=100, facecolor="#1e1e1e")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#1e1e1e")

    if history is not None and not history.empty:
        closes = history["Close"].tail(120)
        ax.plot(range(len(closes)), closes.values, color="#4cc9f0",
                linewidth=1.5, label="History")

        if predicted_path:
            start = len(closes) - 1
            xs = list(range(start, start + len(predicted_path) + 1))
            ys = [float(closes.values[-1])] + list(predicted_path)
            ax.plot(xs, ys, color="#f72585", linewidth=2.0,
                    linestyle="--", label="Forecast")

    ax.set_title(title, color="#ffffff", fontsize=11)
    ax.tick_params(colors="#bbbbbb")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(True, color="#333333", linewidth=0.5)
    ax.legend(facecolor="#1e1e1e", edgecolor="#444444",
              labelcolor="#dddddd", fontsize=9)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    return canvas.get_tk_widget()
