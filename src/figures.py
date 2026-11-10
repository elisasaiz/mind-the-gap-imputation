"""Figures for the report, built from the cross-validated test predictions.

Requires: python -m src.baselines, BERT runs in results/cv/bert/ (all folds).
The same out-of-fold predictions and selection rules as evaluate.py are used.

Outputs (results/figures/, each as PNG for slides and PDF for LaTeX):
  fig1_participant_mae      test MAE of every participant for the three methods,
                            one panel per signal; grey lines join the same participant
  fig2_paired_difference    per participant: MAE(BERT) - MAE(Kalman), sorted;
                            below zero = BERT better
  fig3_example_windows      one example window per signal: observed context, true
                            values in the gap and the three reconstructions

Example windows are not hand-picked: among the test windows without natural
missing values, the window whose true gap values have the 80th-percentile
standard deviation (a gap with visible structure) is shown for each signal.

Usage:
    python -m src.figures
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SIGNALS
from src.cv import load_folds, load_signal, participant_errors
from src.evaluate import test_predictions

FIG_DIR = RESULTS_DIR / "figures"
WIDTH = 6.5  # inches = text width of an A4 report with ~2.5 cm margins: no rescaling in LaTeX

# Categorical slots 1-3 of the reference palette (validated all-pairs, light mode).
# Each method also has its own marker, so identity never relies on colour alone.
METHODS = {
    "linear": {"label": "Linear", "long": "Linear interpolation", "color": "#2a78d6", "marker": "o"},
    "kalman": {"label": "Kalman", "long": "Kalman smoothing", "color": "#eb6834", "marker": "s"},
    "bert": {"label": "BERT", "long": "BERT (LoRA)", "color": "#1baf7a", "marker": "^"},
}
TITLES = {"hr": "Heart rate", "resp": "Respiration", "stress": "Stress"}
UNITS = {"hr": "bpm", "resp": "breaths/min", "stress": "stress score"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.edgecolor": "#b5b4ae", "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {FIG_DIR / name}.png / .pdf")


def collect():
    folds = load_folds()
    out = {}
    for name in SIGNALS.values():
        data = load_signal(name)
        preds, _ = test_predictions(name, data, folds)
        if "bert" not in preds or np.isnan(preds["bert"][data["mask"]]).any():
            raise SystemExit(f"{name}: BERT predictions missing for some folds; "
                             "copy all runs into results/cv/bert/ first")
        errors = {m: participant_errors(data["x_true"], p, data["mask"], data["user_id"])
                  .set_index("user_id").mae for m, p in preds.items()}
        out[name] = {"data": data, "preds": preds, "mae": pd.DataFrame(errors)}
    return out


def fig_participant_mae(res):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9))
    rng = np.random.default_rng(0)
    for ax, (name, r) in zip(axes, res.items()):
        mae = r["mae"][list(METHODS)]
        xs = np.arange(len(METHODS))
        jitter = rng.uniform(-0.08, 0.08, size=len(mae))
        for k, (_, row) in enumerate(mae.iterrows()):  # one grey line per participant
            ax.plot(xs + jitter[k], row.values, color=GRID, lw=0.7, zorder=1)
        for i, (m, s) in enumerate(METHODS.items()):
            ax.scatter(np.full(len(mae), i) + jitter, mae[m], s=18, color=s["color"],
                       marker=s["marker"], edgecolor=SURFACE, linewidth=0.5, zorder=2)
            med = mae[m].median()
            ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2.2, zorder=3,
                    solid_capstyle="round")
        meds = mae.median()
        ax.set_xticks(xs, [f"{METHODS[m]['label']}\n{meds[m]:.2f}" for m in METHODS])
        ax.tick_params(axis="x", length=0, pad=4)
        ax.set_xlim(-0.5, len(METHODS) - 0.5)
        ax.set_title(TITLES[name], loc="left", color=INK)
        ax.set_ylabel(f"MAE ({UNITS[name]})")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig1_participant_mae")


def fig_paired_difference(res):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7))
    for ax, (name, r) in zip(axes, res.items()):
        diff = (r["mae"]["bert"] - r["mae"]["kalman"]).sort_values()
        better = diff < 0
        x = np.arange(len(diff))
        ax.bar(x[better], diff[better], width=0.85, color=METHODS["bert"]["color"])
        ax.bar(x[~better], diff[~better], width=0.85, color=METHODS["kalman"]["color"])
        ax.axhline(0, color=INK_2, lw=0.9)
        ax.set_xticks([])
        ax.set_xlabel("participants (sorted)")
        ax.set_title(TITLES[name], loc="left", color=INK)
        ax.set_ylabel(f"Δ MAE ({UNITS[name]})")
        # bottom right: with real results the most negative bars are on the left
        ax.text(0.97, 0.04, f"BERT better: {int(better.sum())}/{len(diff)}\n"
                f"mean Δ = {diff.mean():+.2f}", transform=ax.transAxes,
                va="bottom", ha="right", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.85))
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.9))
    fig.legend(handles=[Patch(color=METHODS["bert"]["color"], label="BERT lower error"),
                        Patch(color=METHODS["kalman"]["color"], label="Kalman lower error")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig2_paired_difference")


def pick_example(data):
    complete = ~np.isnan(data["x_true"]).any(axis=1)
    idx = np.flatnonzero(complete)
    gap_std = np.array([data["x_true"][i][data["mask"][i]].std() for i in idx])
    target = np.quantile(gap_std, 0.8)
    return int(idx[np.argmin(np.abs(gap_std - target))])


def fig_example_windows(res):
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 7.2))
    for ax, (name, r) in zip(axes, res.items()):
        data, preds = r["data"], r["preds"]
        i = pick_example(data)
        x, mask = data["x_true"][i], data["mask"][i]
        t = np.arange(len(x))
        gap = np.flatnonzero(mask)
        lo, hi = gap[0] - 1, gap[-1] + 1  # observed neighbours, to join the lines
        ax.axvspan(gap[0] - 0.5, gap[-1] + 0.5, color="#efeee9", zorder=0)
        ax.plot(t, np.where(mask, np.nan, x), color=INK_2, lw=1.3)
        ax.plot(t[lo:hi + 1], x[lo:hi + 1], color=INK, lw=1.6, ls=(0, (3, 2)))
        lines = []
        for m, s in METHODS.items():
            p = preds[m][i]
            ax.plot(np.r_[lo, gap, hi], np.r_[x[lo], p[gap], x[hi]], color=s["color"],
                    lw=2.2, marker=s["marker"], markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=0.6, markevery=list(range(1, len(gap) + 1)))
            lines.append(f"{s['label']:<7}{np.abs(p[gap] - x[gap]).mean():5.2f}")
        ax.text(1.02, 0.5, "MAE in gap\n" + "\n".join(lines), transform=ax.transAxes,
                va="center", fontsize=9.5, color=INK, family="monospace")
        ax.set_xlim(max(0, gap[0] - 22), min(len(x) - 1, gap[-1] + 22))
        ax.set_title(f"{TITLES[name]}", loc="left", color=INK)
        ax.set_ylabel(UNITS[name])
    axes[-1].set_xlabel("time step in the 96-step window  (shaded: 12-step artificial gap)")
    handles = [Line2D([], [], color=INK_2, lw=1.3, label="observed"),
               Line2D([], [], color=INK, lw=1.6, ls=(0, (3, 2)), label="true (hidden)")]
    handles += [Line2D([], [], color=s["color"], lw=2.2, marker=s["marker"], markersize=5,
                       label=s["long"]) for s in METHODS.values()]
    fig.tight_layout(h_pad=1.2, rect=(0, 0, 1, 0.94))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.45, 1.0))
    save(fig, "fig3_example_windows")


def main():
    res = collect()
    fig_participant_mae(res)
    fig_paired_difference(res)
    fig_example_windows(res)


if __name__ == "__main__":
    main()
