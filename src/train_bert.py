"""Train the BERT imputer for one signal and save its predictions.

Training:
  - training windows get a NEW random 12-value gap every epoch (same placement
    rules as the fixed masks), so the model sees many gap positions
  - natural missing values are flagged in the input and never used as targets
  - loss: MSE on the artificially masked positions only, in normalised scale
  - normalisation (--norm):
      window  each window is standardised with the mean / std of its own
              OBSERVED input values (the gap is excluded), and predictions are
              transformed back. Removes level differences between participants
              and days (instance normalisation, as in RevIN-style time-series models).
      global  mean / std of all observed training values of that signal
  - two learning rates: --lr for the LoRA adapters, --head-lr for the input
    projection and regression head, which start from scratch
  - the model first spends several epochs predicting roughly the window mean
    before it learns to use the values next to the gap; --min-epochs and a long
    --patience keep early stopping from ending training on that plateau
  - stable optimisation: linear warm-up during the first epoch, cosine decay of the
    learning rate over --epochs, and gradient clipping (--clip). Without these,
    a high learning rate made training collapse back to predicting the window mean
  - mixed precision (float16) on CUDA GPUs to speed up training (--no-amp to disable)
  - early stopping on validation MAE (original scale), computed on the FIXED
    validation masks from make_dataset.py

Outputs:
  results/bert/<signal>/adapter.pt        trained weights (LoRA + proj + head)
  results/bert/<signal>/history.csv       loss and validation MAE per epoch
  results/bert/<signal>/config.json       all settings, for reproducibility
  results/predictions/bert_<signal>_val.npz   (and _test.npz with --predict-test)

Usage:
  python -m src.train_bert --signal hr --subset 200 --epochs 2 --tiny-random   # offline smoke test
  python -m src.train_bert --signal hr                                         # real training (GPU)
  python -m src.train_bert --signal hr --predict-test                          # final evaluation only
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from src.bert_model import BertImputer, count_parameters
from src.config import GAP_LEN, PROCESSED_DIR, RESULTS_DIR, SEED
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signal", required=True, choices=["hr", "resp", "stress"])
    p.add_argument("--model-name", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--min-epochs", type=int, default=10, help="no early stopping before this epoch")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4, help="peak learning rate of the LoRA adapters")
    p.add_argument("--head-lr", type=float, default=5e-4, help="peak learning rate of projection and head")
    p.add_argument("--clip", type=float, default=1.0, help="max gradient norm")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-targets", default="query,key,value",
                   help="comma-separated BERT modules that receive LoRA adapters")
    p.add_argument("--norm", default="window", choices=["window", "global"])
    p.add_argument("--subset", type=int, default=0, help="use only N training windows (debug)")
    p.add_argument("--device", default="auto")
    p.add_argument("--tiny-random", action="store_true", help="offline test without pretrained BERT")
    p.add_argument("--predict-test", action="store_true", help="also predict the test set")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    args = p.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = pick_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    out_dir = RESULTS_DIR / "bert" / args.signal
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    train = np.load(PROCESSED_DIR / f"{args.signal}_train.npz")
    val = np.load(PROCESSED_DIR / f"{args.signal}_val.npz")
    x_train = train["x_true"]
    if args.subset:
        x_train = x_train[rng.permutation(len(x_train))[:args.subset]]
    norm = Normaliser(args.norm, x_train)
    starts = [eligible_starts(w, GAP_LEN) for w in x_train]

    # ---- model ----
    model = BertImputer(args.model_name, args.lora_r, args.lora_alpha,
                        lora_targets=args.lora_targets.split(","),
                        tiny_random=args.tiny_random).to(device)
    trainable, total = count_parameters(model)
    print(f"device={device} (mixed precision: {amp})  train windows={len(x_train)}  val windows={len(val['x_true'])}")
    print(f"trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    lora_params = [q for n, q in model.named_parameters() if q.requires_grad and "lora_" in n]
    new_params = list(model.proj.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lr},
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

    val_mask = val["mask"]
    best_mae, best_epoch, history = np.inf, -1, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # new random gap for every training window
        x_in = x_train.copy()
        target_mask = np.zeros_like(x_train, dtype=bool)
        for i, s in enumerate(starts):
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

        val_pred = predict(model, val["x_input"], norm, device, amp=amp)
        val_mae = float(np.abs(val_pred - val["x_true"])[val_mask].mean())
        history.append({"epoch": epoch, "train_loss": np.mean(losses), "val_mae": val_mae,
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch:3d}  train_loss={np.mean(losses):.4f}  val_MAE={val_mae:.3f}  "
              f"({time.time() - t0:.0f}s)")

        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch
            torch.save(model.trainable_state_dict(), out_dir / "adapter.pt")
        elif epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(f"early stopping: no improvement for {args.patience} epochs")
            break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(args), "global_mean": norm.g_mean, "global_std": norm.g_std,
                   "best_epoch": best_epoch, "best_val_mae": best_mae,
                   "trainable_parameters": trainable, "total_parameters": total}, f, indent=2)

    # ---- predictions with the best checkpoint ----
    model.load_state_dict(torch.load(out_dir / "adapter.pt", map_location=device), strict=False)
    pred_dir = RESULTS_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    splits = ["val", "test"] if args.predict_test else ["val"]
    for split in splits:
        data = np.load(PROCESSED_DIR / f"{args.signal}_{split}.npz")
        x_pred = predict(model, data["x_input"], norm, device, amp=amp)
        np.savez_compressed(pred_dir / f"bert_{args.signal}_{split}.npz", x_pred=x_pred)
    print(f"best epoch {best_epoch}, val MAE {best_mae:.3f}; predictions saved for {splits}")


if __name__ == "__main__":
    main()
