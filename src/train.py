"""
train.py — Shared training/evaluation loop for the CNN and LSTM classifiers.

One function, `run_experiment`, is used by every research question; the RQs
differ only in which view (input / output / input+output) and which split
(stratified random vs held-out domain) they pass in. Keeping a single code path
means RQ1-RQ4 numbers are directly comparable rather than accidentally
reflecting different training regimes.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import (LABELS, build_vocab, encode, iterate_batches)
from models import build_model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, X, lengths, y, batch_size, device, num_classes):
    """Return accuracy, macro-F1, per-class P/R/F1, confusion matrix, preds."""
    model.eval()
    preds = []
    total_loss, n = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, lb, yb in iterate_batches(X, lengths, y, batch_size):
            xb_t = torch.from_numpy(xb).to(device)
            lb_t = torch.from_numpy(lb).to(device)
            yb_t = torch.from_numpy(yb).to(device)
            logits = model(xb_t, lb_t)
            total_loss += criterion(logits, yb_t).item() * len(yb)
            n += len(yb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)

    acc = float((preds == y).mean())
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y, preds):
        cm[t, p] += 1

    per_class = {}
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[LABELS[c]] = {"precision": prec, "recall": rec,
                                "f1": f1, "support": int(cm[c, :].sum())}
        f1s.append(f1)

    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "loss": total_loss / max(n, 1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "predictions": preds.tolist(),
    }


def run_experiment(records, train_idx, val_idx, test_idx, *, model_name, view,
                   tag, max_len=256, embed_dim=128, batch_size=64, epochs=6,
                   lr=1e-3, weight_decay=1e-5, patience=2, seed=13,
                   max_vocab=30000, device=None, verbose=True,
                   token_filter=None, results_dir="results", model_kwargs=None):
    """Train one model on one view and return a result dict.

    `token_filter` is an optional callable applied to the raw text before
    tokenisation; RQ4's ablations use it to strip formatting or normalise
    length so we can measure how much each signal contributes.
    """
    # Resume support: a completed run leaves its result JSON on disk, so a
    # restarted session (or a re-run of the driver) skips work already done
    # rather than repeating hours of training.
    cached = Path(results_dir) / f"{tag}.json"
    if cached.exists():
        try:
            with open(cached) as fh:
                prev = json.load(fh)
            if verbose:
                print(f"    [{tag}] cached: acc={prev['test']['accuracy']:.4f} "
                      f"macro_f1={prev['test']['macro_f1']:.4f} (skipping)",
                      flush=True)
            return prev
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt/partial file: fall through and retrain

    device = device or torch.device("cpu")
    set_seed(seed)

    work = records
    if token_filter is not None:
        work = []
        for r in records:
            r2 = dict(r)
            r2["input"] = token_filter(r["input"])
            r2["output"] = token_filter(r["output"])
            work.append(r2)

    stoi, itos = build_vocab(work, train_idx, view,
                             max_size=max_vocab, min_freq=2)
    Xtr, Ltr, ytr = encode(work, train_idx, view, stoi, max_len)
    Xva, Lva, yva = encode(work, val_idx, view, stoi, max_len)
    Xte, Lte, yte = encode(work, test_idx, view, stoi, max_len)

    model = build_model(model_name, len(itos), len(LABELS),
                        embed_dim=embed_dim, **(model_kwargs or {})).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state, best_epoch, stale = -1.0, None, -1, 0
    history = []
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, lb, yb in iterate_batches(Xtr, Ltr, ytr, batch_size,
                                          shuffle=True, seed=seed + epoch):
            xb_t = torch.from_numpy(xb).to(device)
            lb_t = torch.from_numpy(lb).to(device)
            yb_t = torch.from_numpy(yb).to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb_t, lb_t), yb_t)
            loss.backward()
            # LSTMs on long sequences are prone to exploding gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running += loss.item() * len(yb)
            seen += len(yb)

        val = evaluate(model, Xva, Lva, yva, batch_size, device, len(LABELS))
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                        "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                        "val_macro_f1": val["macro_f1"]})
        if verbose:
            print(f"    [{tag}] epoch {epoch}/{epochs} "
                  f"train_loss={running / max(seen, 1):.4f} "
                  f"val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f}",
                  flush=True)

        # Select on macro-F1 rather than accuracy: classes are balanced here,
        # but the held-out-domain runs in RQ3 can skew badly per class.
        if val["macro_f1"] > best_val:
            best_val, best_epoch, stale = val["macro_f1"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"    [{tag}] early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, Xte, Lte, yte, batch_size, device, len(LABELS))
    elapsed = time.time() - started

    result = {
        "tag": tag,
        "model": model_name,
        "view": view,
        "seed": seed,
        "config": {"max_len": max_len, "embed_dim": embed_dim,
                   "batch_size": batch_size, "epochs": epochs, "lr": lr,
                   "weight_decay": weight_decay, "vocab_size": len(itos),
                   "n_train": len(train_idx), "n_val": len(val_idx),
                   "n_test": len(test_idx)},
        "best_epoch": best_epoch,
        "history": history,
        "test": {k: v for k, v in test.items() if k != "predictions"},
        "test_labels": yte.tolist(),
        "test_predictions": test["predictions"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_seconds": elapsed,
    }

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(results_dir) / f"{tag}.json", "w") as fh:
        json.dump(result, fh, indent=2)

    if verbose:
        print(f"    [{tag}] TEST acc={test['accuracy']:.4f} "
              f"macro_f1={test['macro_f1']:.4f} ({elapsed:.0f}s)", flush=True)
    return result
