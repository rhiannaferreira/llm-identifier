"""
build_txd22.py — Curate the TXD-22 benchmark into the project's corpus schema.

Source: TXD-22: A Large-Scale Benchmark Dataset for Multi-Class AI-Generated
and Mixed-Source Text Identification. Mendeley Data, V1 (July 2026).
DOI: 10.17632/prcjcggtjf.1
Columns: Question (LLM_input), Answer (LLM_output), Source (class label).

Two data-quality issues in the released CSV are handled explicitly:

1. MIXED ENCODING. The file is mostly UTF-8 but contains ~85k stray
   Windows-1252 bytes (smart quotes, en/em dashes, ellipses). Decoding the
   whole file as latin-1 "works" but silently turns every genuine UTF-8
   multi-byte character into mojibake. Since punctuation style is one of the
   signals RQ4 studies, we repair byte-exactly instead: decode as UTF-8 and
   fall back to CP1252 only for the individual bytes that fail.

2. MIXED-AUTHORSHIP AND HUMAN CLASSES. TXD-22 has 22 classes: 17 AI systems,
   human-written text, and four human/AI hybrid categories. The assignment asks
   which *LLM* produced a response, so we keep only pure single-model AI
   classes. The hybrids are retained in the raw file and simply not selected;
   the exclusion is reported rather than hidden.

Output: data/corpus_txd22.jsonl with fields llm, domain, input, output.
(`domain` is set to "qa" — TXD-22 ships no task labels, which is precisely why
RQ3 is run on M4 instead.)
"""

import argparse
import codecs
import csv
import io
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter

csv.field_size_limit(sys.maxsize)


# --- Encoding repair ---------------------------------------------------------

def _cp1252_fallback(err):
    """Decode individual invalid UTF-8 bytes as CP1252 instead of dropping them."""
    chunk = err.object[err.start:err.end]
    try:
        return chunk.decode("cp1252"), err.end
    except UnicodeDecodeError:
        return chunk.decode("latin-1"), err.end


codecs.register_error("cp1252_fallback", _cp1252_fallback)


def read_text(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return raw.decode("utf-8", errors="cp1252_fallback")


# --- Label handling ----------------------------------------------------------

# Classes that are a single AI system. Anything containing a hybrid marker
# ("human", "+", "refined", "mixed") is excluded from the attribution task.
HYBRID_MARKERS = ("human", "+", "refined", "mixed", "hybrid", "/")


def is_pure_ai(label):
    low = label.strip().lower()
    return not any(m in low for m in HYBRID_MARKERS)


def normalise(text):
    """Light normalisation only — markdown, casing and layout are left intact
    because they are the stylistic signals we want to study."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


MIN_WORDS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",
                    default="/mnt/user-data/uploads/TXD-22_A_Large-Scale_Benchmark_Dataset.csv")
    ap.add_argument("--out", default="data/corpus_txd22.jsonl")
    ap.add_argument("--classes", type=int, default=5,
                    help="how many AI classes to keep (largest first)")
    ap.add_argument("--per-class", type=int, default=0, dest="per_class",
                    help="cap per class; 0 = use the smallest surviving class")
    ap.add_argument("--keep", default="",
                    help="comma-separated explicit class names to keep")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    text = read_text(args.source)
    reader = csv.DictReader(io.StringIO(text))

    by_label = {}
    raw_counts = Counter()
    dropped_short = 0
    for row in reader:
        label = (row.get("Source") or "").strip()
        if not label:
            continue
        raw_counts[label] += 1
        if not is_pure_ai(label):
            continue
        q = normalise(row.get("Question", ""))
        a = normalise(row.get("Answer", ""))
        if not q or len(a.split()) < MIN_WORDS:
            dropped_short += 1
            continue
        by_label.setdefault(label, []).append({"input": q, "output": a})

    print(f"raw classes in file: {len(raw_counts)}")
    hybrid = [l for l in raw_counts if not is_pure_ai(l)]
    print(f"excluded hybrid/human classes ({len(hybrid)}): {sorted(hybrid)}")
    print(f"dropped {dropped_short:,} rows as empty/too short (<{MIN_WORDS} words)")

    if args.keep:
        wanted = [c.strip() for c in args.keep.split(",")]
        missing = [c for c in wanted if c not in by_label]
        if missing:
            raise SystemExit(f"requested classes not found: {missing}\n"
                             f"available: {sorted(by_label)}")
    else:
        wanted = sorted(by_label, key=lambda l: -len(by_label[l]))[:args.classes]

    cap = args.per_class or min(len(by_label[c]) for c in wanted)
    records = []
    for label in wanted:
        rows = by_label[label][:]
        rng.shuffle(rows)
        for r in rows[:cap]:
            records.append({"llm": label, "domain": "qa",
                            "input": r["input"], "output": r["output"]})
    rng.shuffle(records)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(records):,} records -> {args.out}")
    print(f"kept {len(wanted)} classes at {cap:,} examples each:")
    for c in wanted:
        print(f"  {c:<28} (available {len(by_label[c]):,})")


if __name__ == "__main__":
    main()
