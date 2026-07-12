"""Build a slide-friendly PNG latency figure from the saved benchmark CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent
    latency = pd.read_csv(root / "output" / "ps5_q1_latency_runs.csv")

    import matplotlib.pyplot as plt

    values = latency["latency_us"]
    median = values.median()
    p95 = values.quantile(0.95)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.hist(values, bins=60, color="#CC0000", alpha=0.78)
    ax.axvline(median, color="black", linestyle="--", linewidth=1.5, label=f"median {median:.1f} μs")
    ax.axvline(p95, color="#555555", linestyle=":", linewidth=2.0, label=f"p95 {p95:.1f} μs")

    ax.set_title("Latency Distribution for One Full Quote-Grid Update", fontsize=16, pad=14)
    ax.set_xlabel("Latency per quote-grid computation (microseconds)", fontsize=12)
    ax.set_ylabel("Number of benchmark runs", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    # A few rare OS/runtime outliers stretch the x-axis. Keeping the full range
    # would make the main 100--200 μs mass unreadable on slides, so we zoom to
    # the 99th percentile plus a little margin and report tail latency via p95.
    ax.set_xlim(values.min() * 0.98, values.quantile(0.99) * 1.15)

    fig.tight_layout()
    fig.savefig(root / "figures" / "quote_grid_latency_distribution.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
