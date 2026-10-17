"""Experiment settings shared by all scripts.

Change values here, never inside the individual scripts, so that every
method reads the same configuration.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "csi-balance"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SEED = 42

# Signals studied (column name in the CSV -> short name used in files)
SIGNALS = {
    "hr_time_series": "hr",
    "resp_time_series": "resp",
    "stress_time_series": "stress",
}

# Window and gap
WINDOW_LEN = 96   # observations per window (Jacobsen & Tropmann-Frick, 2024)
GAP_LEN = 12      # contiguous masked observations (12.5% of the window)

# If False, only windows with no natural missing values are kept.
# If True, windows may contain natural NaNs outside the artificial gap,
# as long as the gap and its two neighbours are observed.
ALLOW_NATURAL_NAN = True

# Participant-level split (fractions of participants)
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
