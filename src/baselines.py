"""Baseline imputation methods: linear interpolation and Kalman smoothing.

Both methods work window by window and learn nothing across participants, so
their predictions can be computed once for all windows. The only choice is the
Kalman specification (local level vs local linear trend), which evaluate.py
selects separately in every fold, using the inner validation participants only.

Kalman smoothing:
  A structural state-space model (statsmodels UnobservedComponents) is fitted by
  maximum likelihood to the observed values of each window; the masked values are
  never seen. The smoothed level at the missing positions is the imputation.

Outputs:
  results/cv/predictions/linear_<signal>.npz
  results/cv/predictions/kalman_llevel_<signal>.npz
  results/cv/predictions/kalman_lltrend_<signal>.npz     (key: x_pred, all windows)

Usage:
    python -m src.baselines
"""
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from src.config import SIGNALS
from src.cv import PRED_DIR, load_signal

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


def kalman_smoothing(x_input: np.ndarray, spec: str) -> np.ndarray:
    return np.stack([kalman_window(x, spec) for x in x_input])


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for name in SIGNALS.values():
        x_input = load_signal(name)["x_input"]
        t0 = time.time()
        np.savez_compressed(PRED_DIR / f"linear_{name}.npz", x_pred=linear_interpolation(x_input))
        for spec in KALMAN_SPECS:
            np.savez_compressed(PRED_DIR / f"kalman_{spec}_{name}.npz",
                                x_pred=kalman_smoothing(x_input, spec))
        print(f"{name:7s} done ({len(x_input)} windows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
