"""Load the CSI Balance Challenge training files into one DataFrame.

Each CSV is one participant (user id in the filename), each row one day.
The three time series are stored as strings like "[55, 51, None, ...]"
and are converted to float numpy arrays where every unavailable value is NaN:
  - heart rate:  "None"
  - respiration: negative codes (-1, -2)
  - stress:      negative codes (-1, -2). 0 is a valid stress value.

Usage:
    python -m src.load
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DIR, SIGNALS


def parse_series(value) -> np.ndarray:
    """Convert a string like "[55, None, -1]" into a float array with NaNs."""
    if not isinstance(value, str):
        return np.array([], dtype=float)
    values = []
    for token in value.strip("[]").split(","):
        token = token.strip()
        if token == "":
            continue
        if token == "None":
            values.append(np.nan)
        else:
            x = float(token)
            values.append(np.nan if x < 0 else x)  # -1 / -2 are missing codes
    return np.array(values, dtype=float)


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Merge all participant files. Returns one row per participant-day."""
    files = sorted(Path(raw_dir).rglob("dataset_user_*_train.csv"))
    files = [f for f in files if "__MACOSX" not in f.parts]
    if not files:
        raise FileNotFoundError(f"No training CSVs found under {raw_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, sep=";", index_col=0)
        df["user_id"] = int(re.search(r"user_(\d+)", f.name).group(1))
        df["group"] = f.parent.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    for col in SIGNALS:
        df[col] = df[col].map(parse_series)

    keep = ["user_id", "group", "day", "label"] + list(SIGNALS)
    return df[keep].sort_values(["user_id", "day"]).reset_index(drop=True)


if __name__ == "__main__":
    data = load_raw()
    print(f"{len(data)} participant-days, {data.user_id.nunique()} participants")
    for col, name in SIGNALS.items():
        lengths = data[col].map(len)
        n_nan = data[col].map(lambda a: np.isnan(a).sum()).sum()
        total = lengths.sum()
        print(f"{name:7s} most common length={lengths.mode()[0]}, "
              f"missing={n_nan}/{total} ({100 * n_nan / total:.1f}%)")
