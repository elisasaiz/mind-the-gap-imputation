"""Helpers shared by the cross-validation scripts."""
import json

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR

CV_DIR = RESULTS_DIR / "cv"
PRED_DIR = CV_DIR / "predictions"


def load_signal(signal: str) -> dict:
    with np.load(PROCESSED_DIR / f"{signal}_all.npz") as d:
        return {k: d[k] for k in d.files}


def load_folds() -> dict:
    with open(PROCESSED_DIR / "folds.json") as f:
        return json.load(f)


def rows_of(data: dict, users) -> np.ndarray:
    """Indices of the windows that belong to the given participants."""
    return np.flatnonzero(np.isin(data["user_id"], users))


def participant_errors(x_true, x_pred, mask, user_id) -> pd.DataFrame:
    """MAE / RMSE per participant over the artificially masked positions."""
    n_gap = int(mask[0].sum())
    err = (x_pred - x_true)[mask].reshape(len(x_true), n_gap)
    df = pd.DataFrame({"user_id": np.repeat(user_id, n_gap),
                       "abs": np.abs(err).ravel(), "sq": (err ** 2).ravel()})
    out = df.groupby("user_id").agg(mae=("abs", "mean"), mse=("sq", "mean"),
                                    n_points=("abs", "size"))
    out["rmse"] = np.sqrt(out.pop("mse"))
    out["n_windows"] = out.n_points // n_gap
    return out.reset_index()


def participant_mae(x_true, x_pred, mask, user_id) -> float:
    """Selection criterion: MAE averaged over participants (as in the evaluation)."""
    return float(participant_errors(x_true, x_pred, mask, user_id).mae.mean())
