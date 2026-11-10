"""Build the fixed experimental dataset: windows, masks and cross-validation folds.

Every method (linear, Kalman, BERT) reads these files, so all of them are
evaluated on exactly the same windows, gap positions and folds.

Outputs (data/processed/):
  <signal>_all.npz   all eligible windows of all participants:
      x_true   (N, 96) original values (NaN where naturally missing)
      x_input  (N, 96) same as x_true but with the artificial gap set to NaN
      mask     (N, 96) True on the artificially masked positions
      user_id, day, window_idx, gap_start  (N,)
  folds.json         for each fold: test / inner-train / inner-val participants

Cross-validation design:
  - outer loop: participants are shuffled (fixed seed) and split into N_FOLDS
    groups; each group is the test set of one fold
  - inner split: in each fold, INNER_VAL_PARTICIPANTS of the remaining
    (training) participants form the inner validation set, used only for
    hyperparameter selection and early stopping; the rest is inner training
  - the same folds are used for every signal and every method

Usage:
    python -m src.make_dataset
"""
import json

import numpy as np
import pandas as pd

from src.config import (ALLOW_NATURAL_NAN, GAP_LEN, INNER_VAL_PARTICIPANTS, N_FOLDS,
                        PROCESSED_DIR, SEED, SIGNALS, WINDOW_LEN)
from src.load import load_raw
from src.windows import eligible_starts, split_into_windows


def make_folds(user_ids, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    users = np.array(sorted(set(int(u) for u in user_ids)))
    rng.shuffle(users)
    folds = {}
    for k, test in enumerate(np.array_split(users, N_FOLDS)):
        train = np.setdiff1d(users, test)
        inner_rng = np.random.default_rng(seed + 1 + k)
        val = np.sort(inner_rng.choice(train, size=INNER_VAL_PARTICIPANTS, replace=False))
        folds[str(k)] = {
            "test": sorted(int(u) for u in test),
            "inner_train": sorted(int(u) for u in np.setdiff1d(train, val)),
            "inner_val": sorted(int(u) for u in val),
        }
    return folds


def build_signal(df: pd.DataFrame, col: str, rng: np.random.Generator) -> dict:
    records = []
    for uid, day, x in zip(df.user_id, df.day, df[col]):
        for k, w in enumerate(split_into_windows(x, WINDOW_LEN)):
            if not ALLOW_NATURAL_NAN and np.isnan(w).any():
                continue
            starts = eligible_starts(w, GAP_LEN)
            if len(starts) == 0:
                continue
            s = int(rng.choice(starts))
            mask = np.zeros(WINDOW_LEN, dtype=bool)
            mask[s:s + GAP_LEN] = True
            x_input = w.copy()
            x_input[mask] = np.nan
            records.append((uid, day, k, s, w, x_input, mask))
    uid, day, k, s, w, x_input, mask = zip(*records)
    return {"x_true": np.stack(w), "x_input": np.stack(x_input), "mask": np.stack(mask),
            "user_id": np.array(uid), "day": np.array(day),
            "window_idx": np.array(k), "gap_start": np.array(s)}


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()

    folds = make_folds(df.user_id)
    with open(PROCESSED_DIR / "folds.json", "w") as f:
        json.dump(folds, f, indent=1)
    for k, fold in folds.items():
        print(f"fold {k}: test={len(fold['test'])}  inner_train={len(fold['inner_train'])}  "
              f"inner_val={len(fold['inner_val'])} participants")

    for col, name in SIGNALS.items():
        rng = np.random.default_rng(SEED)  # same seed per signal -> reproducible
        data = build_signal(df, col, rng)
        np.savez_compressed(PROCESSED_DIR / f"{name}_all.npz", **data)
        test_counts = [int(np.isin(data["user_id"], folds[k]["test"]).sum()) for k in folds]
        print(f"{name:7s} windows: {len(data['x_true'])}  (test windows per fold: {test_counts})")


if __name__ == "__main__":
    main()
