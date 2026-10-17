"""Exploratory analysis needed before building the windows.

Answers three questions for each signal:
  1. How long are the natural gaps? (supports the choice of GAP_LEN)
  2. How many 96-observation windows are eligible for masking? (Risk 2)
  3. How are eligible windows distributed across participants?

Outputs:
  results/eda/summary.csv          one row per signal
  results/eda/gap_lengths_<s>.csv  natural gap length distribution
  results/eda/gap_lengths.png      histogram of natural gap lengths

Usage:
    python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import GAP_LEN, RESULTS_DIR, SIGNALS, WINDOW_LEN
from src.load import load_raw
from src.windows import eligible_starts, split_into_windows


def gap_lengths(x: np.ndarray) -> list:
    """Lengths of every run of consecutive NaNs in a sequence."""
    runs, current = [], 0
    for missing in np.isnan(x):
        if missing:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def main():
    out = RESULTS_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)
    df = load_raw()

    rows = []
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(14, 3.5))
    for ax, (col, name) in zip(axes, SIGNALS.items()):
        gaps = [g for x in df[col] for g in gap_lengths(x)]
        gaps = pd.Series(gaps, dtype=int)
        gaps.value_counts().sort_index().rename("count").to_csv(
            out / f"gap_lengths_{name}.csv", index_label="gap_length")

        n_windows = n_complete = n_eligible_nan = 0
        per_user = {}
        for uid, x in zip(df.user_id, df[col]):
            for w in split_into_windows(x, WINDOW_LEN):
                n_windows += 1
                complete = not np.isnan(w).any()
                n_complete += complete
                n_eligible_nan += len(eligible_starts(w, GAP_LEN)) > 0
                per_user[uid] = per_user.get(uid, 0) + complete

        per_user = pd.Series(per_user)
        rows.append({
            "signal": name,
            "pct_missing": 100 * df[col].map(lambda a: np.isnan(a).sum()).sum()
                           / df[col].map(len).sum(),
            "n_natural_gaps": len(gaps),
            "gap_median": gaps.median(),
            "pct_gaps_le_12": 100 * (gaps <= GAP_LEN).mean(),
            "n_windows": n_windows,
            "n_complete_windows": n_complete,
            "n_eligible_allowing_nan": n_eligible_nan,
            "users_with_0_complete": int((per_user == 0).sum()),
            "min_complete_per_user": int(per_user.min()),
            "median_complete_per_user": float(per_user.median()),
        })

        clipped = gaps.clip(upper=60)
        ax.hist(clipped, bins=range(1, 62), color="#4C72B0")
        ax.axvline(GAP_LEN + 0.5, color="#C44E52", ls="--", label=f"gap = {GAP_LEN}")
        ax.set_title(f"{name}: natural gap lengths (60+ clipped)")
        ax.set_xlabel("consecutive missing observations")
        ax.set_yscale("log")
        ax.legend()
    axes[0].set_ylabel("number of gaps (log)")
    fig.tight_layout()
    fig.savefig(out / "gap_lengths.png", dpi=150)

    summary = pd.DataFrame(rows).round(2)
    summary.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.T.to_string(header=False))


if __name__ == "__main__":
    main()
