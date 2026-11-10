"""Cross-validated evaluation of all methods.

For every signal and every fold:
  - linear interpolation: no choice to make
  - Kalman smoothing: the specification (local level / local linear trend) with the
    lowest participant-level MAE on the fold's INNER-VALIDATION participants is
    selected, then applied to the fold's test participants
  - BERT: among the finished runs of the fold (one per learning rate), the one with
    the lowest inner-validation MAE is selected; its test predictions are used
The test participants of a fold are never used for any choice. Every participant is
a test participant in exactly one fold, so in the end each method has one test
error per participant (45 participants).

Errors are computed only at the artificially masked positions, per participant.
Two subsets are reported:
  all       every eligible window (natural NaNs allowed outside the gap)
  complete  only windows without natural missing values (sensitivity analysis)

If BERT is not finished for every fold yet, the table is computed only on the
participants of the folds where ALL methods are available, so the comparison is
always on the same participants.

Outputs (results/cv/):
  participants.csv   one row per signal x subset x method x participant
  summary.csv        mean / median / std across participants
  pairwise.csv       per pair of methods: participants where A beats B, mean difference,
                     paired Wilcoxon signed-rank p-value (descriptive, see report)
  selections.csv     setting selected in each fold (Kalman spec, BERT learning rate)

Usage:
    python -m src.evaluate
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import SIGNALS
from src.cv import CV_DIR, PRED_DIR, load_folds, load_signal, participant_errors, participant_mae, rows_of

KALMAN_SPECS = ["llevel", "lltrend"]


def test_predictions(name: str, data: dict, folds: dict):
    """Assemble out-of-fold test predictions for every method.

    Returns {method: x_pred with NaN rows for folds that are not available},
    and a list of selection records.
    """
    n = len(data["x_true"])
    preds, selections = {}, []

    preds["linear"] = np.load(PRED_DIR / f"linear_{name}.npz")["x_pred"]

    kalman = {s: np.load(PRED_DIR / f"kalman_{s}_{name}.npz")["x_pred"] for s in KALMAN_SPECS}
    preds["kalman"] = np.full((n, data["x_true"].shape[1]), np.nan)
    for k, fold in folds.items():
        va, te = rows_of(data, fold["inner_val"]), rows_of(data, fold["test"])
        scores = {s: participant_mae(data["x_true"][va], p[va], data["mask"][va],
                                     data["user_id"][va]) for s, p in kalman.items()}
        best = min(scores, key=scores.get)
        preds["kalman"][te] = kalman[best][te]
        selections.append({"signal": name, "fold": k, "method": "kalman",
                           "selected": best, "inner_val_mae": round(scores[best], 3)})

    preds["bert"] = np.full((n, data["x_true"].shape[1]), np.nan)
    for k in folds:
        runs = []
        for run_file in sorted((CV_DIR / "bert" / name / f"fold{k}").glob("lr*/run.json")):
            with open(run_file) as f:
                runs.append((json.load(f), run_file.parent))
        if not runs:
            continue
        run, run_dir = min(runs, key=lambda r: r[0]["inner_val_mae"])
        with np.load(run_dir / "predictions.npz") as d:
            preds["bert"][d["test_rows"]] = d["test_pred"]
        selections.append({"signal": name, "fold": k, "method": "bert",
                           "selected": f"lr={run['lr']:g} ({len(runs)} runs)",
                           "inner_val_mae": round(run["inner_val_mae"], 3)})
    return preds, selections


def main():
    folds = load_folds()
    rows, pair_rows, all_selections = [], [], []

    for name in SIGNALS.values():
        data = load_signal(name)
        preds, selections = test_predictions(name, data, folds)
        all_selections += selections

        # keep only participants for which every method has test predictions
        available = {m: ~np.isnan(p[data["mask"]].reshape(len(p), -1)).any(axis=1)
                     for m, p in preds.items()}
        common = np.logical_and.reduce(list(available.values()))
        if not available["bert"].any():
            preds.pop("bert")
            common = np.logical_and(available["linear"], available["kalman"])
            print(f"{name}: no BERT runs yet, evaluating linear and Kalman only")
        elif not available["bert"].all():
            done = sorted({k for k, f in folds.items()
                           if available["bert"][rows_of(data, f["test"])].all()})
            print(f"{name}: BERT finished for folds {done} only; "
                  f"all methods evaluated on those folds' participants")

        complete = ~np.isnan(data["x_true"]).any(axis=1)
        for subset, keep in {"all": common, "complete": common & complete}.items():
            per_method = {}
            for method, p in preds.items():
                pe = participant_errors(data["x_true"][keep], p[keep], data["mask"][keep],
                                        data["user_id"][keep])
                per_method[method] = pe.set_index("user_id")
                rows.append(pe.assign(signal=name, subset=subset, method=method))
            for a, b in combinations(per_method, 2):
                ua = per_method[a].mae
                ub = per_method[b].mae.reindex(ua.index)
                diff = ua - ub
                try:
                    p_value = wilcoxon(ua, ub).pvalue
                except ValueError:
                    p_value = np.nan
                pair_rows.append({"signal": name, "subset": subset, "A": a, "B": b,
                                  "participants": len(diff),
                                  "A_better": int((diff < 0).sum()),
                                  "B_better": int((diff > 0).sum()),
                                  "mean_mae_difference_A_minus_B": round(diff.mean(), 3),
                                  "wilcoxon_p": round(p_value, 4)})

    per_user = pd.concat(rows, ignore_index=True)
    per_user = per_user[["signal", "subset", "method", "user_id", "mae", "rmse",
                         "n_windows", "n_points"]]
    per_user.to_csv(CV_DIR / "participants.csv", index=False)

    summary = (per_user.groupby(["signal", "subset", "method"], sort=False)
               .agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"), mae_std=("mae", "std"),
                    rmse_mean=("rmse", "mean"), rmse_median=("rmse", "median"),
                    n_participants=("user_id", "nunique"), n_windows=("n_windows", "sum"))
               .round(3).reset_index())
    summary.to_csv(CV_DIR / "summary.csv", index=False)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(CV_DIR / "pairwise.csv", index=False)
    pd.DataFrame(all_selections).to_csv(CV_DIR / "selections.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n=== Test errors (participant level, out-of-fold) ===")
        print(summary.to_string(index=False))
        print("\n=== Pairwise comparison of participant MAE ===")
        print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
