"""95% bootstrap confidence intervals for the participant-level results.

Requires: python -m src.evaluate (reads results/cv/participants.csv).

Participants are resampled with replacement (10,000 bootstrap samples, seed 42),
so the interval reflects the variation between participants, the unit of analysis.

Outputs results/cv/uncertainty.csv with, per signal and subset:
  - the mean participant MAE of each method with its 95% CI
  - the mean paired difference A - B for each pair of methods with its 95% CI

Usage:
    python -m src.uncertainty
"""
from itertools import combinations

import numpy as np
import pandas as pd

from src.config import SEED
from src.cv import CV_DIR

N_BOOT = 10_000


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    per_user = pd.read_csv(CV_DIR / "participants.csv")
    rows = []
    for (signal, subset), df in per_user.groupby(["signal", "subset"], sort=False):
        mae = df.pivot(index="user_id", columns="method", values="mae").dropna()
        rng = np.random.default_rng(SEED)
        for m in mae.columns:
            lo, hi = bootstrap_ci(mae[m].to_numpy(), rng)
            rows.append({"signal": signal, "subset": subset, "quantity": f"MAE {m}",
                         "mean": mae[m].mean(), "ci_low": lo, "ci_high": hi})
        for a, b in combinations(["bert", "kalman", "linear"], 2):
            if a not in mae or b not in mae:
                continue
            diff = (mae[a] - mae[b]).to_numpy()
            lo, hi = bootstrap_ci(diff, rng)
            rows.append({"signal": signal, "subset": subset, "quantity": f"{a} - {b}",
                         "mean": diff.mean(), "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows).round(3)
    out.to_csv(CV_DIR / "uncertainty.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
