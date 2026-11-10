"""Train the BERT imputer with participant-level k-fold cross-validation.

For every fold and every hyperparameter setting in the grid (--lrs):
  - the model is trained on the INNER-TRAINING participants of the fold
  - early stopping uses the INNER-VALIDATION participants of the fold
  - predictions are saved for the inner-validation windows (used by evaluate.py to
    select the best setting of the fold) and for the TEST windows of the fold
The test participants of a fold are never used for training, early stopping or
selection. Selection of the best setting per fold happens in evaluate.py, with the
same rule used for the Kalman specification (lowest participant-level MAE on the
inner validation participants).

Training recipe (fixed, the same in every fold):
  - training windows get a NEW random 12-value gap every epoch (same placement
    rules as the fixed masks), so the model sees many gap positions
  - natural missing values are flagged in the input and never used as targets
  - loss: MSE on the artificially masked positions only, in normalised scale
  - per-window normalisation with the mean / std of the observed input values
    (--norm window), predictions transformed back to the original scale
  - LoRA on query / key / value of a frozen pretrained BERT; separate learning rates
    for the LoRA adapters (--lr, the tuned hyperparameter) and for the input
    projection and regression head (--head-lr)
  - linear warm-up in the first epoch, cosine decay over --epochs, gradient clipping
  - --min-epochs and --patience prevent early stopping on the initial plateau where
    the model only predicts the window mean
  - mixed precision (float16) on CUDA GPUs (--no-amp to disable)

The script can be interrupted and restarted: finished (fold, setting) runs are
skipped, so on Colab it simply continues where it stopped.

Outputs, per run, in results/cv/bert/<signal>/fold<k>/lr<lr>/:
  predictions.npz   val_rows, val_pred, test_rows, test_pred
  history.csv       loss and inner-validation MAE per epoch
  run.json          settings, best epoch, inner-validation MAE

Usage:
  python -m src.train_bert --signal hr --folds 0 --subset 300 --epochs 2 --tiny-random  # offline test
  python -m src.train_bert --signal hr                     # all folds, full grid (GPU)
  python -m src.train_bert --signal hr --folds 0,1         # only some folds
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch

from src.bert_model import BertImputer, count_parameters
from src.config import GAP_LEN, SEED
from src.cv import CV_DIR, load_folds, load_signal, participant_mae, rows_of
from src.windows import eligible_starts


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Normaliser:
    """Per-window or global standardisation, always based on observed inputs only."""

    def __init__(self, mode: str, train_windows: np.ndarray):
        self.mode = mode
        self.g_mean = float(np.nanmean(train_windows))
        self.g_std = float(np.nanstd(train_windows))

    def stats(self, x_input: np.ndarray):
        if self.mode == "global":
            n = len(x_input)
            return np.full((n, 1), self.g_mean), np.full((n, 1), self.g_std)
        mu = np.nanmean(x_input, axis=1, keepdims=True)
        sd = np.nanstd(x_input, axis=1, keepdims=True)
        sd = np.maximum(sd, 0.1 * self.g_std)  # flat windows: avoid dividing by ~0
        return mu, sd


def to_model_inputs(x_input: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    """NaN-containing windows -> (normalised values with 0 at NaN, missing flag)."""
    missing = np.isnan(x_input)
    values = np.where(missing, 0.0, (x_input - mu) / sd)
    return values.astype(np.float32), missing.astype(np.float32)


@torch.no_grad()
def predict(model, x_input, norm: Normaliser, device, batch_size=256, amp=False) -> np.ndarray:
    model.eval()
    mu, sd = norm.stats(x_input)
    values, missing = to_model_inputs(x_input, mu, sd)
    out = []
    for i in range(0, len(values), batch_size):
        v = torch.from_numpy(values[i:i + batch_size]).to(device)
        m = torch.from_numpy(missing[i:i + batch_size]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out.append(model(v, m).float().cpu().numpy())
    return np.concatenate(out) * sd + mu


def train_one(args, data, fold, lr, device, amp, out_dir):
    torch.manual_seed(SEED + int(fold["id"]))
    rng = np.random.default_rng(SEED + int(fold["id"]))

    tr = rows_of(data, fold["inner_train"])
    va = rows_of(data, fold["inner_val"])
    te = rows_of(data, fold["test"])
    if args.subset:
        tr = rng.permutation(tr)[:args.subset]
    x_train = data["x_true"][tr]
    norm = Normaliser(args.norm, x_train)
    starts = [eligible_starts(w, GAP_LEN) for w in x_train]

    model = BertImputer(args.model_name, args.lora_r, args.lora_alpha,
                        lora_targets=args.lora_targets.split(","),
                        tiny_random=args.tiny_random).to(device)
    trainable, total = count_parameters(model)
    lora_params = [q for n, q in model.named_parameters() if q.requires_grad and "lora_" in n]
    new_params = list(model.proj.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": lora_params, "lr": lr},
                             {"params": new_params, "lr": args.head_lr}])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    steps_per_epoch = int(np.ceil(len(x_train) / args.batch_size))
    total_steps, warmup = args.epochs * steps_per_epoch, steps_per_epoch

    def lr_factor(step):  # linear warm-up, then cosine decay to 0
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    def val_score():
        pred = predict(model, data["x_input"][va], norm, device, amp=amp)
        return participant_mae(data["x_true"][va], pred, data["mask"][va], data["user_id"][va])

    best_mae, best_epoch, best_state, history = np.inf, -1, None, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        x_in = x_train.copy()
        target_mask = np.zeros_like(x_train, dtype=bool)
        for i, s in enumerate(starts):  # new random gap for every training window
            s = int(rng.choice(s))
            target_mask[i, s:s + GAP_LEN] = True
        x_in[target_mask] = np.nan
        mu, sd = norm.stats(x_in)  # statistics of the observed input only
        values, missing = to_model_inputs(x_in, mu, sd)
        targets = np.nan_to_num((x_train - mu) / sd).astype(np.float32)

        model.train()
        order = rng.permutation(len(x_train))
        losses = []
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            v = torch.from_numpy(values[idx]).to(device)
            m = torch.from_numpy(missing[idx]).to(device)
            y = torch.from_numpy(targets[idx]).to(device)
            tm = torch.from_numpy(target_mask[idx]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                pred = model(v, m)
            loss = ((pred.float() - y) ** 2)[tm].mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g["params"]],
                                           args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        val_mae = val_score()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_mae": val_mae, "seconds": round(time.time() - t0, 1)})
        print(f"    epoch {epoch:3d}  train_loss={np.mean(losses):.4f}  "
              f"inner_val_MAE={val_mae:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch
            best_state = copy.deepcopy(model.trainable_state_dict())
        elif epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(f"    early stopping: no improvement for {args.patience} epochs")
            break

    model.load_state_dict(best_state, strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        val_rows=va, val_pred=predict(model, data["x_input"][va], norm, device, amp=amp),
        test_rows=te, test_pred=predict(model, data["x_input"][te], norm, device, amp=amp))
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    run = {**vars(args), "fold": fold["id"], "lr": lr, "best_epoch": best_epoch,
           "inner_val_mae": best_mae, "trainable_parameters": trainable,
           "total_parameters": total, "n_train_windows": int(len(tr))}
    with open(out_dir / "run.json", "w") as f:  # written last: marks the run as finished
        json.dump(run, f, indent=2)
    return best_mae, best_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signal", required=True, choices=["hr", "resp", "stress"])
    p.add_argument("--folds", default="all", help='e.g. "0,1" (default: all folds)')
    p.add_argument("--lrs", default="1e-4,2e-4",
                   help="grid of LoRA learning rates, selected per fold on inner validation")
    p.add_argument("--model-name", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--min-epochs", type=int, default=10, help="no early stopping before this epoch")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-lr", type=float, default=5e-4, help="peak lr of projection and head")
    p.add_argument("--clip", type=float, default=1.0, help="max gradient norm")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-targets", default="query,key,value")
    p.add_argument("--norm", default="window", choices=["window", "global"])
    p.add_argument("--subset", type=int, default=0, help="use only N training windows (debug)")
    p.add_argument("--device", default="auto")
    p.add_argument("--tiny-random", action="store_true", help="offline test without pretrained BERT")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    p.add_argument("--out-name", default="bert", help="results subfolder (use another for tests)")
    args = p.parse_args()

    device = pick_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    data = load_signal(args.signal)
    folds = load_folds()
    fold_ids = list(folds) if args.folds == "all" else args.folds.split(",")
    lrs = [float(x) for x in args.lrs.split(",")]
    print(f"signal={args.signal}  device={device} (mixed precision: {amp})  "
          f"folds={fold_ids}  lr grid={lrs}")

    for k in fold_ids:
        fold = {**folds[k], "id": k}
        for lr in lrs:
            out_dir = CV_DIR / args.out_name / args.signal / f"fold{k}" / f"lr{lr:g}"
            if (out_dir / "run.json").exists():
                print(f"fold {k}  lr={lr:g}: already done, skipping")
                continue
            print(f"fold {k}  lr={lr:g}: training on {len(fold['inner_train'])} participants, "
                  f"early stopping on {len(fold['inner_val'])}", flush=True)
            t0 = time.time()
            mae, ep = train_one(args, data, fold, lr, device, amp, out_dir)
            print(f"fold {k}  lr={lr:g}: best epoch {ep}, inner-val MAE {mae:.3f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
