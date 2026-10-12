"""
run_experiments.py — Driver for all four research questions.

Usage:
    python src/run_experiments.py --rq 1
    python src/run_experiments.py --rq 2
    python src/run_experiments.py --rq 3
    python src/run_experiments.py --rq 4
    python src/run_experiments.py --rq all

RQ1 is a strict subset of RQ2 (the output-only condition), so running RQ2
produces RQ1's numbers as well; `--rq 1` exists so the required result can be
reproduced on its own.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import (LABELS, DOMAINS, load_corpus, make_splits, domain_splits,
                  grouped_splits, field_text, tokenize)
from train import run_experiment

RESULTS = Path("results")
VIEWS = ["input", "output", "input+output"]
MODELS = ["cnn", "lstm"]


def set_results_dir(path):
    global RESULTS
    RESULTS = Path(path)
    RESULTS.mkdir(parents=True, exist_ok=True)


# The LSTM is far more expensive per token than the CNN on CPU, so each
# architecture gets its own sequence budget. Both see the same data; only the
# truncation length and hidden width differ, and both are reported.
ARCH = {
    "cnn":  {"max_len": 256, "batch_size": 64, "model_kwargs": {}},
    "lstm": {"max_len": 128, "batch_size": 64, "model_kwargs": {"hidden_dim": 96}},
}


def arch_cfg(model_name, args):
    cfg = dict(ARCH[model_name])
    if args.max_len:
        cfg["max_len"] = args.max_len
    return cfg


def cached(name):
    """Return a previously computed artefact, so restarts skip finished work."""
    p = RESULTS / name
    if p.exists():
        try:
            with open(p) as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            return None
    return None


def save(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / name, "w") as fh:
        json.dump(obj, fh, indent=2)


# ---------------------------------------------------------------- RQ1 & RQ2

def rq1_rq2(records, args, views=VIEWS, grouped=False, prefix="rq2"):
    """Train every (architecture, view) pair on one shared split.

    `grouped=True` splits by prompt group instead of at random, so all
    responses to a given question land in the same fold (see
    data.grouped_splits for why this matters for the input-only condition).
    """
    if grouped:
        train_idx, val_idx, test_idx = grouped_splits(records, seed=args.seed)
    else:
        train_idx, val_idx, test_idx = make_splits(records, seed=args.seed)
    print(f"split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    summary = []
    for view in views:
        for model_name in MODELS:
            tag = f"{prefix}_{model_name}_{view.replace('+', '-')}"
            cfg = arch_cfg(model_name, args)
            print(f"\n>>> {model_name.upper()} / view={view} "
                  f"(max_len={cfg['max_len']})")
            res = run_experiment(
                records, train_idx, val_idx, test_idx,
                model_name=model_name, view=view, tag=tag,
                max_len=cfg["max_len"], epochs=args.epochs,
                batch_size=cfg["batch_size"], seed=args.seed,
                model_kwargs=cfg["model_kwargs"],
                device=torch.device("cpu"), results_dir=str(RESULTS))
            summary.append({
                "model": model_name, "view": view,
                "accuracy": res["test"]["accuracy"],
                "macro_f1": res["test"]["macro_f1"],
                "per_class": res["test"]["per_class"],
                "confusion_matrix": res["test"]["confusion_matrix"],
                "n_params": res["n_params"],
                "best_epoch": res["best_epoch"],
                "train_seconds": res["train_seconds"],
            })
            save(f"{prefix}_summary.json"
                 if prefix != "rq2" else "rq1_rq2_summary.json", summary)
    return summary


# ---------------------------------------------------------------- RQ3

def rq3(records, args):
    """Hold out each domain in turn; train on the rest, test on the held-out one.

    Rotating all four domains (rather than picking one) shows whether any
    apparent generalisation is a property of the fingerprint or an artefact of
    one convenient train/test pairing.
    """
    summary = []
    for held_out in DOMAINS:
        tr, va, te = domain_splits(records, held_out, seed=args.seed)
        print(f"\n### held-out domain: {held_out} "
              f"(train={len(tr)} val={len(va)} test={len(te)})")
        for model_name in MODELS:
            tag = f"rq3_{model_name}_holdout-{held_out}"
            cfg = arch_cfg(model_name, args)
            res = run_experiment(
                records, tr, va, te,
                model_name=model_name, view="output", tag=tag,
                max_len=cfg["max_len"], epochs=args.epochs,
                batch_size=cfg["batch_size"], seed=args.seed,
                model_kwargs=cfg["model_kwargs"],
                device=torch.device("cpu"), results_dir=str(RESULTS))
            summary.append({
                "held_out": held_out, "model": model_name,
                "accuracy": res["test"]["accuracy"],
                "macro_f1": res["test"]["macro_f1"],
                "per_class": res["test"]["per_class"],
                "confusion_matrix": res["test"]["confusion_matrix"],
            })
            save("rq3_summary.json", summary)
    return summary


# ---------------------------------------------------------------- RQ4

def rq4_ablations(records, args):
    """Re-train under text transforms that each remove one family of signals."""
    from features import ABLATIONS

    train_idx, val_idx, test_idx = make_splits(records, seed=args.seed)
    summary = []
    for name, fn in ABLATIONS.items():
        print(f"\n### ablation: {name}")
        for model_name in MODELS:
            tag = f"rq4_{model_name}_{name}"
            cfg = arch_cfg(model_name, args)
            res = run_experiment(
                records, train_idx, val_idx, test_idx,
                model_name=model_name, view="output", tag=tag,
                max_len=cfg["max_len"], epochs=args.epochs,
                batch_size=cfg["batch_size"], seed=args.seed,
                model_kwargs=cfg["model_kwargs"], token_filter=fn,
                device=torch.device("cpu"), results_dir=str(RESULTS))
            summary.append({
                "ablation": name, "model": model_name,
                "accuracy": res["test"]["accuracy"],
                "macro_f1": res["test"]["macro_f1"],
            })
            save("rq4_ablation_summary.json", summary)
    return summary


def rq4_stylometrics(records, args):
    """Describe the generators with interpretable features, and test how far
    those features alone go (logistic regression + random-forest importances)."""
    prev = cached("rq4_stylometrics.json")
    if prev is not None:
        print(f"  [rq4_stylometrics.json] cached, skipping")
        return prev

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from features import feature_matrix, FEATURE_NAMES

    train_idx, val_idx, test_idx = make_splits(records, seed=args.seed)
    Xtr, names = feature_matrix(records, train_idx)
    Xte, _ = feature_matrix(records, test_idx)
    ytr = np.array([LABELS.index(records[i]["llm"]) for i in train_idx])
    yte = np.array([LABELS.index(records[i]["llm"]) for i in test_idx])

    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(scaler.transform(Xtr), ytr)
    lr_acc = float(lr.score(scaler.transform(Xte), yte))

    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed,
                                n_jobs=-1, min_samples_leaf=2)
    rf.fit(Xtr, ytr)
    rf_acc = float(rf.score(Xte, yte))
    importances = sorted(zip(names, rf.feature_importances_.tolist()),
                         key=lambda kv: -kv[1])

    # Per-generator means of each feature, for the descriptive table.
    profiles = {}
    for ci, label in enumerate(LABELS):
        rows = Xtr[ytr == ci]
        profiles[label] = {n: float(v) for n, v in zip(names, rows.mean(axis=0))}

    out = {
        "stylometric_logreg_accuracy": lr_acc,
        "stylometric_randomforest_accuracy": rf_acc,
        "top_features": importances[:25],
        "per_generator_means": profiles,
        "n_features": len(names),
    }
    save("rq4_stylometrics.json", out)
    print(f"\nstylometric-only accuracy: logreg={lr_acc:.4f} rf={rf_acc:.4f}")
    print("top features:")
    for n, v in importances[:12]:
        print(f"   {n:<24} {v:.4f}")
    return out


def rq4_prompt_leakage(records, args):
    """RQ2(iii) follow-up: is input-only accuracy real signal or template leakage?

    M4's prompts were written per (domain, generator) by the corpus authors, so
    an input-only classifier may be recognising prompt templates rather than
    anything about the model. We quantify that directly: how many distinct
    prompt templates exist per generator, and how separable are they?
    """
    prev = cached("rq4_prompt_leakage.json")
    if prev is not None:
        print(f"  [rq4_prompt_leakage.json] cached, skipping")
        return prev

    from collections import Counter, defaultdict

    def skeleton(text, k=12):
        """First k tokens as a crude template signature."""
        return " ".join(tokenize(text)[:k])

    per_gen = defaultdict(Counter)
    for r in records:
        per_gen[r["llm"]][skeleton(r["input"])] += 1

    all_sk = Counter()
    for g, c in per_gen.items():
        all_sk.update(c.keys())

    # A template is "generator-exclusive" if it appears for exactly one model.
    exclusive = {g: sum(1 for s in c if all_sk[s] == 1) for g, c in per_gen.items()}
    coverage = {}
    for g, c in per_gen.items():
        total = sum(c.values())
        excl = sum(n for s, n in c.items() if all_sk[s] == 1)
        coverage[g] = {"n_examples": total,
                       "n_distinct_templates": len(c),
                       "n_exclusive_templates": exclusive[g],
                       "frac_examples_with_exclusive_template": excl / total}

    out = {"note": ("Fraction of examples whose first-12-token prompt skeleton "
                    "occurs for only one generator. High values mean input-only "
                    "accuracy reflects corpus construction, not model identity."),
           "per_generator": coverage}
    save("rq4_prompt_leakage.json", out)
    print("\nprompt-template exclusivity by generator:")
    for g, v in coverage.items():
        print(f"   {g:<9} {v['frac_examples_with_exclusive_template']:.3f} "
              f"({v['n_distinct_templates']} distinct templates)")
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq", default="all",
                    choices=["1", "2", "3", "4", "grouped", "all"])
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--results", default="results",
                    help="output directory (use a separate one per corpus)")
    ap.add_argument("--max-len", type=int, default=0, dest="max_len",
                    help="override the per-architecture default (0 = use it)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    torch.set_num_threads(2)
    set_results_dir(args.results)
    records = load_corpus(args.corpus)
    print(f"loaded {len(records):,} records from {args.corpus}")
    print(f"generators ({len(LABELS)}): {LABELS}")
    print(f"domains ({len(DOMAINS)}): {DOMAINS}")
    print(f"results -> {RESULTS}")

    if args.rq == "1":
        rq1_rq2(records, args, views=["output"])
    elif args.rq in ("2", "all"):
        rq1_rq2(records, args)
    if args.rq in ("3", "all"):
        if len(DOMAINS) < 2:
            print("\n[skip] RQ3 needs >=2 labelled domains; this corpus has "
                  f"{len(DOMAINS)} ({DOMAINS}). Run RQ3 on the M4 corpus.")
        else:
            rq3(records, args)
    if args.rq in ("4", "all"):
        rq4_stylometrics(records, args)
        rq4_prompt_leakage(records, args)
        rq4_ablations(records, args)
    if args.rq in ("grouped", "all"):
        # Methodological control: identical models and views, but split by
        # prompt group so no question spans train and test.
        print("\n=== grouped-split control (no question spans folds) ===")
        rq1_rq2(records, args, grouped=True, prefix="rq2group")


if __name__ == "__main__":
    main()
