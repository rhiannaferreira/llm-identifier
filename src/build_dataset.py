"""
build_dataset.py — Curate a balanced (LLM_name, LLM_input, LLM_output) corpus
from the M4 machine-generated-text corpus.

Source: M4 (Wang et al., 2024), https://github.com/mbzuai-nlp/M4
Each source file is `{domain}_{generator}.jsonl`, with fields:
    prompt       -> the instruction given to the generator   (LLM_input)
    machine_text -> the generator's response                  (LLM_output)
    human_text   -> the original human-written text           (unused here)

We keep the five generators present across all five English domains, and sample
an equal number of examples per (generator, domain) cell so that classes are
balanced both overall and within every domain. The per-domain `domain` field is
retained so RQ3 can hold out whole domains.

Output: data/corpus.jsonl with fields
    llm, domain, input, output
"""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# Canonical label -> the filename spelling used in each domain's file.
# M4 is inconsistent about capitalisation and the dolly/dolly2 suffix, so we map
# explicitly rather than globbing.
FILE_MAP = {
    "wikipedia": {
        "chatGPT": "wikipedia_chatgpt.jsonl",
        "davinci": "wikipedia_davinci.jsonl",
        "cohere": "wikipedia_cohere.jsonl",
        "dolly": "wikipedia_dolly.jsonl",
        "bloomz": "wikipedia_bloomz.jsonl",
    },
    "wikihow": {
        "chatGPT": "wikihow_chatGPT.jsonl",
        "davinci": "wikihow_davinci.jsonl",
        "cohere": "wikihow_cohere.jsonl",
        "dolly": "wikihow_dolly2.jsonl",
        "bloomz": "wikihow_bloomz.jsonl",
    },
    "reddit": {
        "chatGPT": "reddit_chatGPT.jsonl",
        "davinci": "reddit_davinci.jsonl",
        "cohere": "reddit_cohere.jsonl",
        "dolly": "reddit_dolly.jsonl",
        "bloomz": "reddit_bloomz.jsonl",
    },
    "arxiv": {
        "chatGPT": "arxiv_chatGPT.jsonl",
        "davinci": "arxiv_davinci.jsonl",
        "cohere": "arxiv_cohere.jsonl",
        "dolly": "arxiv_dolly.jsonl",
        "bloomz": "arxiv_bloomz.jsonl",
    },
}

# NOTE ON peerread: M4's fifth English domain is deliberately excluded. Its
# `prompt` and `machine_text` fields hold *lists* of reviews rather than single
# strings, and the bloomz file uses an entirely different schema (per-generator
# `*_reviews` columns on a shared row). Flattening it would mean inventing a
# one-prompt-to-one-response mapping that the source does not actually specify,
# and the domain only has 586 rows per generator. Four domains is enough to
# rotate held-out domains for RQ3, so we keep the curation honest instead.

LABELS = ["chatGPT", "davinci", "cohere", "dolly", "bloomz"]

DEFAULT_CAPS = {
    "wikipedia": 1000,
    "wikihow": 1000,
    "reddit": 1000,
    "arxiv": 1000,
}

MIN_OUTPUT_CHARS = 200
MIN_OUTPUT_WORDS = 30


def normalise(text):
    """Light normalisation only.

    Deliberately conservative: we do NOT strip markdown, casing, punctuation or
    whitespace structure, because those are exactly the stylistic signals RQ4
    asks us to study. We only fix unicode form and collapse runs of blank lines
    so that the token budget is not eaten by padding-like whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def usable(record):
    """Drop truncated, empty or degenerate generations."""
    out = record["output"]
    if len(out) < MIN_OUTPUT_CHARS:
        return False
    if len(out.split()) < MIN_OUTPUT_WORDS:
        return False
    if not record["input"]:
        return False
    # Some generators emit a single character or token repeated; drop those.
    if len(set(out.split())) < 10:
        return False
    return True


def load_cell(path, llm, domain, cap, rng):
    """Read one {domain}_{generator}.jsonl and return up to `cap` clean records."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = {
                "llm": llm,
                "domain": domain,
                "input": normalise(row.get("prompt", "")),
                "output": normalise(row.get("machine_text", "")),
            }
            if usable(rec):
                records.append(rec)
    rng.shuffle(records)
    return records[:cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/tmp/m4/data")
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    source = Path(args.source)
    all_records = []
    shortfall = []

    for domain, files in FILE_MAP.items():
        cap = DEFAULT_CAPS[domain]
        cell_records = {}
        for llm, fname in files.items():
            path = source / fname
            if not path.exists():
                raise FileNotFoundError(f"missing source file: {path}")
            cell_records[llm] = load_cell(path, llm, domain, cap, rng)

        # Enforce exact balance within the domain: every generator contributes
        # the same count, set by the smallest surviving cell.
        n = min(len(v) for v in cell_records.values())
        if n < cap:
            shortfall.append((domain, cap, n))
        for llm in LABELS:
            all_records.extend(cell_records[llm][:n])

    rng.shuffle(all_records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_llm = Counter(r["llm"] for r in all_records)
    by_domain = Counter(r["domain"] for r in all_records)
    cross = defaultdict(Counter)
    for r in all_records:
        cross[r["domain"]][r["llm"]] += 1

    print(f"wrote {len(all_records):,} records -> {out_path}")
    print("\nper generator:")
    for k in LABELS:
        print(f"  {k:<9} {by_llm[k]:>6,}")
    print("\nper domain (count per generator in parentheses):")
    for d in FILE_MAP:
        per = sorted(set(cross[d].values()))
        print(f"  {d:<10} {by_domain[d]:>6,}  ({per} each)")
    if shortfall:
        print("\ncells that fell short of the cap after filtering:")
        for d, cap, n in shortfall:
            print(f"  {d}: cap {cap} -> {n} per generator")


if __name__ == "__main__":
    main()
