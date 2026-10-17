"""Baseline imputation methods: linear interpolation and Kalman smoothing.

Each method receives only `x_input` (the window with the artificial gap and any
natural missing values set to NaN) and returns a full (N, 96) array of
predictions. Predictions are saved so that evaluate.py treats every method,
including BERT later, in exactly the same way.

Kalman smoothing:
  A structural state-space model (statsmodels UnobservedComponents) is fitted by
  maximum likelihood to the observed values of each window; the masked values are
  never seen. The smoothed state at the missing positions is the imputation.
  The model specification (local level vs local linear trend) is chosen on the
  validation set and then fixed for the test set.

Outputs:
  results/predictions/<method>_<signal>_<split>.npz   (key: x_pred)

Usage:
    python -m src.baselines                 # validation set (default)
    python -m src.baselines --split test    # final evaluation only
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from src.config import PROCESSED_DIR, RESULTS_DIR, SIGNALS

PRED_DIR = RESULTS_DIR / "predictions"

# Specification used on the test set; set after comparing them on validation.
KALMAN_SPEC = "llevel"
KALMAN_SPECS = ["llevel", "lltrend"]


def linear_interpolation(x_input: np.ndarray) -> np.ndarray:
    """Straight line between the observed values on each side of every gap."""
    df = pd.DataFrame(x_input.T)  # one column per window
    return df.interpolate(method="linear", limit_area="inside").to_numpy().T


def kalman_window(x: np.ndarray, spec: str) -> np.ndarray:
    """Kalman-smoothed reconstruction of one window (NaNs are handled natively)."""
    model = UnobservedComponents(x, level=spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warnings on short windows
        res = model.fit(disp=False)
    return res.smoothed_state[0]  # level component


def kalman_smoothing(x_input: np.ndarray, spec: str = KALMAN_SPEC) -> np.ndarray:
    return np.stack([kalman_window(x, spec) for x in x_input])


def save(method: str, signal: str, split: str, x_pred: np.ndarray):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED_DIR / f"{method}_{signal}_{split}.npz", x_pred=x_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    # On validation, try every Kalman specification; on test, only the chosen one.
    specs = KALMAN_SPECS if args.split == "val" else [KALMAN_SPEC]

    for name in SIGNALS.values():
        data = np.load(PROCESSED_DIR / f"{name}_{args.split}.npz")
        x_input = data["x_input"]
        save("linear", name, args.split, linear_interpolation(x_input))
        for spec in specs:
            method = f"kalman_{spec}" if args.split == "val" else "kalman"
            save(method, name, args.split, kalman_smoothing(x_input, spec))
        print(f"{name:7s} done ({len(x_input)} windows)")


if __name__ == "__main__":
    main()
