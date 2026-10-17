"""Build the fixed experimental dataset: participant split, windows, masks.

Every method (linear, Kalman, BERT) reads these files, so all of them are
evaluated on exactly the same windows and gap positions.

Outputs (data/processed/):
  split.csv                       user_id -> train / val / test
  <signal>_<split>.npz            for each signal and split:
      x_true   (N, 96) original values (NaN where naturally missing)
      x_input  (N, 96) same as x_true but with the artificial gap set to NaN
      mask     (N, 96) True on the artificially masked positions
      user_id, day, window_idx, gap_start  (N,)

Usage:
    python -m src.make_dataset
"""
import numpy as np
import pandas as pd

from src.config import (ALLOW_NATURAL_NAN, GAP_LEN, PROCESSED_DIR, SEED,
                        SIGNALS, SPLIT_FRACTIONS, WINDOW_LEN)
from src.load import load_raw
from src.windows import eligible_starts, split_into_windows


def participant_split(user_ids, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = np.array(sorted(set(user_ids)))
    rng.shuffle(users)
    n = len(users)
    n_train = round(SPLIT_FRACTIONS["train"] * n)
    n_val = round(SPLIT_FRACTIONS["val"] * n)
    split = (["train"] * n_train + ["val"] * n_val
             + ["test"] * (n - n_train - n_val))
    return pd.DataFrame({"user_id": users, "split": split}).sort_values("user_id")


def build_signal(df: pd.DataFrame, col: str, rng: np.random.Generator) -> pd.DataFrame:
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
    return pd.DataFrame(records, columns=["user_id", "day", "window_idx",
                                          "gap_start", "x_true", "x_input", "mask"])


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    split = participant_split(df.user_id)
    split.to_csv(PROCESSED_DIR / "split.csv", index=False)
    split_of = dict(zip(split.user_id, split.split))
    print("participants per split:", split.split.value_counts().to_dict())

    for col, name in SIGNALS.items():
        rng = np.random.default_rng(SEED)  # same seed per signal -> reproducible
        windows = build_signal(df, col, rng)
        windows["split"] = windows.user_id.map(split_of)
        counts = {}
        for part, sub in windows.groupby("split"):
            np.savez_compressed(
                PROCESSED_DIR / f"{name}_{part}.npz",
                x_true=np.stack(sub.x_true.values),
                x_input=np.stack(sub.x_input.values),
                mask=np.stack(sub["mask"].values),
                user_id=sub.user_id.values,
                day=sub.day.values,
                window_idx=sub.window_idx.values,
                gap_start=sub.gap_start.values,
            )
            counts[part] = len(sub)
        print(f"{name:7s} windows: {counts}")


if __name__ == "__main__":
    main()
