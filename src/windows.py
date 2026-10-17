"""Window extraction and mask placement rules (shared by EDA and dataset build)."""
import numpy as np


def split_into_windows(x: np.ndarray, window_len: int) -> list:
    """Non-overlapping windows from the start of the day; the tail is dropped."""
    n = len(x) // window_len
    return [x[i * window_len:(i + 1) * window_len] for i in range(n)]


def eligible_starts(w: np.ndarray, gap_len: int) -> np.ndarray:
    """Start positions s where the gap w[s:s+gap_len] can be placed.

    Rules (from the research plan):
      - the gap is not at the window edges (s >= 1, s + gap_len <= len(w) - 1)
      - the gap covers originally observed values only
      - the value immediately before and after the gap is observed
    """
    observed = ~np.isnan(w)
    starts = []
    for s in range(1, len(w) - gap_len):
        if observed[s - 1:s + gap_len + 1].all():
            starts.append(s)
    return np.array(starts, dtype=int)
