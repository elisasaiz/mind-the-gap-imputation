"""Evaluate every method that has saved predictions.

Errors are computed only at the artificially masked positions. For each
participant, MAE and RMSE are computed over all masked points of all their
windows; the table then summarises these participant-level values (mean, median).

Two subsets are reported:
  all       every eligible window (natural NaNs allowed outside the gap)
  complete  only windows without natural missing values (sensitivity analysis)

Outputs:
  results/metrics_<split>_participants.csv   one row per method x signal x subset x participant
  results/metrics_<split>_summary.csv        mean / median across participants

Usage:
    python -m src.evaluate                  # validation set (default)
    python -m src.evaluate --split test
"""
import argparse

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR, SIGNALS

PRED_DIR = RESULTS_DIR / "predictions"


def participant_errors(x_true, x_pred, mask, user_id) -> pd.DataFrame:
    err = (x_pred - x_true)[mask].reshape(len(x_true), -1)  # (N, GAP_LEN)
    df = pd.DataFrame({"user_id": np.repeat(user_id, err.shape[1]),
                       "abs": np.abs(err).ravel(), "sq": (err ** 2).ravel()})
    out = df.groupby("user_id").agg(mae=("abs", "mean"), mse=("sq", "mean"),
                                    n_points=("abs", "size"))
    out["rmse"] = np.sqrt(out.pop("mse"))
    out["n_windows"] = out.n_points // err.shape[1]
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    rows = []
    for name in SIGNALS.values():
        data = np.load(PROCESSED_DIR / f"{name}_{args.split}.npz")
        x_true, mask, user_id = data["x_true"], data["mask"], data["user_id"]
        complete = ~np.isnan(x_true).any(axis=1)
        subsets = {"all": np.ones(len(x_true), bool), "complete": complete}

        for f in sorted(PRED_DIR.glob(f"*_{name}_{args.split}.npz")):
            method = f.name.removesuffix(f"_{name}_{args.split}.npz")
            x_pred = np.load(f)["x_pred"]
            if np.isnan(x_pred[mask]).any():
                raise ValueError(f"{f.name}: NaN predictions at masked positions")
            for subset, keep in subsets.items():
                if keep.sum() == 0:
                    continue
                pe = participant_errors(x_true[keep], x_pred[keep],
                                        mask[keep], user_id[keep])
                pe.insert(0, "subset", subset)
                pe.insert(0, "signal", name)
                pe.insert(0, "method", method)
                rows.append(pe)

    if not rows:
        raise SystemExit("No predictions found. Run `python -m src.baselines` first.")
    per_user = pd.concat(rows, ignore_index=True)
    per_user.to_csv(RESULTS_DIR / f"metrics_{args.split}_participants.csv", index=False)

    summary = (per_user.groupby(["signal", "subset", "method"])
               .agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"),
                    rmse_mean=("rmse", "mean"), rmse_median=("rmse", "median"),
                    n_participants=("user_id", "nunique"),
                    n_windows=("n_windows", "sum"))
               .round(3).reset_index())
    summary.to_csv(RESULTS_DIR / f"metrics_{args.split}_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
