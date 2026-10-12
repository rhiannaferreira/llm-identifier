"""
make_figures.py — Turn the saved result JSON into the report's figures.

Produces, in results/figures/:
    fig1_rq2_views.png          accuracy by view and architecture
    fig2_confusion.png          confusion matrices for the best RQ1 models
    fig3_rq3_crossdomain.png    held-out-domain accuracy
    fig4_rq4_ablations.png      accuracy under each signal ablation
    fig5_rq4_features.png       top stylometric features + generator profiles
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import LABELS, DOMAINS

RESULTS = Path("results")
FIGDIR = RESULTS / "figures"

# Colour-blind-safe qualitative palette, consistent across every figure.
PALETTE = ["#4269D0", "#EFB118", "#FF725C", "#6CC5B0", "#A463F2"]
MODEL_COLOUR = {"cnn": "#4269D0", "lstm": "#FF725C"}
def chance():
    """Random-guess accuracy; LABELS is filled at runtime."""
    return 1.0 / max(len(LABELS), 1)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def _infer_labels():
    """Recover LABELS/DOMAINS from saved results when no corpus is passed."""
    import data as _d
    for fname in ("rq1_rq2_summary.json", "rq4_stylometrics.json"):
        obj = load(fname)
        if not obj:
            continue
        if fname == "rq1_rq2_summary.json":
            pc = obj[0].get("per_class", {})
            if pc:
                _d.LABELS[:] = list(pc.keys()); break
        else:
            keys = list(obj.get("per_generator_means", {}))
            if keys:
                _d.LABELS[:] = keys; break
    rq3s = load("rq3_summary.json")
    if rq3s and not _d.LABELS:
        # The M4 results hold only RQ3, so take the generators from there.
        _d.LABELS[:] = list(rq3s[0].get("per_class", {}))
    if rq3s:
        seen = []
        for r in rq3s:
            if r["held_out"] not in seen:
                seen.append(r["held_out"])
        _d.DOMAINS[:] = seen


def load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _chance_line(ax):
    c = chance()
    ax.axhline(c, ls="--", lw=1, color="#888")
    ax.text(0.995, c + 0.015, f"chance ({c:.0%})",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="#666")


def fig_rq2_views(summary):
    views = ["input", "output", "input+output"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.36
    x = np.arange(len(views))
    for k, model in enumerate(["cnn", "lstm"]):
        # Only plot conditions that actually ran; a missing run must not be
        # drawn as a 0.0 bar, which would read as "the model scored zero".
        xs, vals = [], []
        for xi, v in zip(x, views):
            rec = next((r for r in summary if r["model"] == model and r["view"] == v), None)
            if rec:
                xs.append(xi + (k - 0.5) * width)
                vals.append(rec["accuracy"])
        if not xs:
            continue
        bars = ax.bar(xs, vals, width, label=model.upper(), color=MODEL_COLOUR[model])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=7.5)
    _chance_line(ax)
    ax.set_xticks(x); ax.set_xticklabels(["Input only", "Output only", "Input + Output"])
    ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("RQ2 — Does the prompt help identify the generator?", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig1_rq2_views.png"); plt.close(fig)


def fig_confusion(summary):
    recs = [r for r in summary if r["view"] == "output"]
    fig, axes = plt.subplots(1, len(recs), figsize=(4.6 * len(recs), 4.1))
    if len(recs) == 1:
        axes = [axes]
    for ax, rec in zip(axes, recs):
        cm = np.array(rec["confusion_matrix"], dtype=float)
        cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=45, ha="right"); ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{rec['model'].upper()} — output only "
                     f"(acc {rec['accuracy']:.3f})", loc="left", fontweight="bold")
        ax.grid(False)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{cm_n[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if cm_n[i, j] > 0.5 else "#333")
        fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised rate")
    fig.suptitle("RQ1 — Which generators get confused with which?",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIGDIR / "fig2_confusion.png"); plt.close(fig)


def fig_rq3(summary, baseline):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(DOMAINS)); width = 0.36
    for k, model in enumerate(["cnn", "lstm"]):
        xs, vals = [], []
        for xi, d in zip(x, DOMAINS):
            rec = next((r for r in summary if r["model"] == model and r["held_out"] == d), None)
            if rec:
                xs.append(xi + (k - 0.5) * width)
                vals.append(rec["accuracy"])
        if not xs:
            continue
        bars = ax.bar(xs, vals, width, label=f"{model.upper()} (held-out)",
                      color=MODEL_COLOUR[model])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=7.5)
    # In-domain reference line from the RQ1 output-only run.
    if baseline:
        for model in ["cnn", "lstm"]:
            rec = next((r for r in baseline if r["model"] == model and r["view"] == "output"), None)
            if rec:
                ax.axhline(rec["accuracy"], ls=":", lw=1.2, color=MODEL_COLOUR[model])
                ax.text(len(DOMAINS) - 0.45, rec["accuracy"] + 0.012,
                        f"{model.upper()} in-domain {rec['accuracy']:.3f}",
                        fontsize=7, color=MODEL_COLOUR[model], ha="right")
    _chance_line(ax)
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in DOMAINS])
    ax.set_xlabel("Held-out domain (trained on the other three)")
    ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("RQ3 — Do fingerprints survive a domain shift?", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig3_rq3_crossdomain.png"); plt.close(fig)


def fig_rq4_ablations(summary):
    order = ["full", "no_formatting", "no_punctuation", "length_capped_100w",
             "shuffled_words"]
    pretty = {"full": "Full text", "no_formatting": "No markdown\nformatting",
              "no_punctuation": "No\npunctuation",
              "length_capped_100w": "Length capped\n(100 words)",
              "shuffled_words": "Word order\nshuffled"}
    present = [o for o in order if any(r["ablation"] == o for r in summary)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(present)); width = 0.36
    for k, model in enumerate(["cnn", "lstm"]):
        xs, vals = [], []
        for xi, a in zip(x, present):
            rec = next((r for r in summary if r["model"] == model and r["ablation"] == a), None)
            if rec:
                xs.append(xi + (k - 0.5) * width)
                vals.append(rec["accuracy"])
        if not xs:
            continue
        bars = ax.bar(xs, vals, width, label=model.upper(),
                      color=MODEL_COLOUR[model])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=7.5)
    _chance_line(ax)
    ax.set_xticks(x); ax.set_xticklabels([pretty[a] for a in present], fontsize=8)
    ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("RQ4 — Which signals is the classifier actually using?",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig4_rq4_ablations.png"); plt.close(fig)


def fig_rq4_features(sty):
    top = sty["top_features"][:14][::-1]
    names = [t[0] for t in top]; vals = [t[1] for t in top]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    ax1.barh(np.arange(len(names)), vals, color="#4269D0")
    ax1.set_yticks(np.arange(len(names))); ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Random-forest importance")
    ax1.set_title("Most discriminative surface features", loc="left", fontweight="bold")

    # Profile a few headline features per generator, z-scored for comparability.
    feats = ["n_words", "mean_sentence_len", "root_ttr", "bullet_lines",
             "comma_per_100w"]
    prof = sty["per_generator_means"]
    mat = np.array([[prof[l][f] for f in feats] for l in LABELS], dtype=float)
    mu, sd = mat.mean(axis=0), mat.std(axis=0)
    z = (mat - mu) / np.where(sd == 0, 1, sd)
    im = ax2.imshow(z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax2.set_xticks(range(len(feats)))
    ax2.set_xticklabels(["Response\nlength", "Sentence\nlength", "Lexical\ndiversity",
                         "Bullet\nlines", "Commas\nper 100w"], fontsize=8)
    ax2.set_yticks(range(len(LABELS))); ax2.set_yticklabels(LABELS)
    ax2.grid(False)
    for i in range(len(LABELS)):
        for j in range(len(feats)):
            ax2.text(j, i, f"{z[i, j]:+.1f}", ha="center", va="center", fontsize=7.5,
                     color="white" if abs(z[i, j]) > 1.3 else "#333")
    ax2.set_title("Generator style profiles (z-scored across generators)",
                  loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax2, fraction=0.046, label="z-score")
    fig.tight_layout(); fig.savefig(FIGDIR / "fig5_rq4_features.png"); plt.close(fig)


def main():
    global RESULTS, FIGDIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--corpus", default="",
                    help="corpus jsonl, used only to recover label/domain names")
    args = ap.parse_args()
    RESULTS = Path(args.results)
    FIGDIR = RESULTS / "figures"
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # Figures label their axes with the generator and domain names, which are
    # corpus-specific. Recover them from the corpus if given, else from results.
    if args.corpus:
        from data import load_corpus
        load_corpus(args.corpus)
    else:
        _infer_labels()

    rq12 = load("rq1_rq2_summary.json")
    rq3s = load("rq3_summary.json")
    rq4a = load("rq4_ablation_summary.json")
    sty = load("rq4_stylometrics.json")

    if rq12:
        fig_rq2_views(rq12); fig_confusion(rq12)
    if rq3s:
        fig_rq3(rq3s, rq12)
    if rq4a:
        fig_rq4_ablations(rq4a)
    if sty:
        fig_rq4_features(sty)
    print(f"figures -> {FIGDIR}")


if __name__ == "__main__":
    main()
