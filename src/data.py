"""
data.py — Tokenisation, vocabulary, splits and tensor batching.

Design note: the tokeniser is intentionally *structure-preserving*. Standard NLP
cleaning (lowercasing, stripping punctuation and markdown) would destroy the
very signals RQ4 asks about — bullet characters, heading marks, quote style,
capitalisation habits. So we keep punctuation and markdown as separate tokens
and let the model decide what matters.
"""

import json
import re
from collections import Counter

import numpy as np

# LABELS and DOMAINS are populated from whichever corpus is loaded (M4 or
# TXD-22). They are mutated IN PLACE rather than rebound so that modules which
# did `from data import LABELS` keep seeing the current values.
LABELS = []
LABEL2ID = {}
DOMAINS = []


def configure(records):
    """Derive the label and domain inventory from the loaded corpus."""
    LABELS[:] = sorted({r["llm"] for r in records})
    LABEL2ID.clear()
    LABEL2ID.update({l: i for i, l in enumerate(LABELS)})
    DOMAINS[:] = sorted({r["domain"] for r in records})
    return LABELS, DOMAINS

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

# Words, contractions, numbers, newlines, and every other non-space character
# (punctuation, bullets, markdown marks) as its own token.
TOKEN_RE = re.compile(r"\n|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z\d]")


def tokenize(text):
    """Tokenise, mapping newlines to an explicit token so layout is learnable."""
    return ["<nl>" if t == "\n" else t.lower() if t.isalpha() else t
            for t in TOKEN_RE.findall(text)]


def load_corpus(path="data/corpus.jsonl"):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    configure(records)
    return records


def make_splits(records, seed=13, val_frac=0.1, test_frac=0.1):
    """Stratified split on (llm, domain) so every cell is proportionally present."""
    rng = np.random.RandomState(seed)
    buckets = {}
    for i, r in enumerate(records):
        buckets.setdefault((r["llm"], r["domain"]), []).append(i)

    train, val, test = [], [], []
    for key in sorted(buckets):
        idx = np.array(buckets[key])
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test.extend(idx[:n_test].tolist())
        val.extend(idx[n_test:n_test + n_val].tolist())
        train.extend(idx[n_test + n_val:].tolist())

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def grouped_splits(records, seed=13, val_frac=0.1, test_frac=0.1):
    """Split by PROMPT GROUP, so all responses to one question share a fold.

    Why this matters: in TXD-22 every question is answered by all five models.
    Under a plain random split, a test question's four siblings sit in the
    training set carrying the four *other* labels. An input-only classifier
    then learns "this question text goes with those four classes" and
    systematically predicts anything but the correct one -- which is exactly
    why the prompt-only CNN scores below the 0.200 chance rate. Grouping by question removes the artefact and makes
    the input-only number interpretable.
    """
    rng = np.random.RandomState(seed)
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault(r["input"], []).append(i)

    keys = sorted(groups)
    rng.shuffle(keys)
    n = len(keys)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test, val, train = [], [], []
    for k in keys[:n_test]:
        test.extend(groups[k])
    for k in keys[n_test:n_test + n_val]:
        val.extend(groups[k])
    for k in keys[n_test + n_val:]:
        train.extend(groups[k])

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def domain_splits(records, held_out, seed=13, val_frac=0.1):
    """RQ3: train on every domain except `held_out`, test on `held_out` alone.

    The validation set is carved from the *training* domains, so model selection
    never sees the held-out domain. That is the whole point of the experiment.
    """
    rng = np.random.RandomState(seed)
    in_idx = [i for i, r in enumerate(records) if r["domain"] != held_out]
    out_idx = [i for i, r in enumerate(records) if r["domain"] == held_out]

    by_cell = {}
    for i in in_idx:
        by_cell.setdefault((records[i]["llm"], records[i]["domain"]), []).append(i)

    train, val = [], []
    for key in sorted(by_cell):
        idx = np.array(by_cell[key])
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val.extend(idx[:n_val].tolist())
        train.extend(idx[n_val:].tolist())

    rng.shuffle(train); rng.shuffle(val)
    return train, val, out_idx


def field_text(record, view):
    """Assemble the text for a given RQ2 view."""
    if view == "input":
        return record["input"]
    if view == "output":
        return record["output"]
    if view == "input+output":
        # An explicit separator token keeps the two segments distinguishable.
        return record["input"] + " <sep> " + record["output"]
    raise ValueError(f"unknown view: {view}")


def build_vocab(records, indices, view, max_size=30000, min_freq=2):
    """Vocabulary is built on TRAINING indices only — never on val/test."""
    counter = Counter()
    for i in indices:
        counter.update(tokenize(field_text(records[i], view)))
    itos = [PAD, UNK, "<sep>"]
    for tok, freq in counter.most_common():
        if freq < min_freq or tok in (PAD, UNK, "<sep>"):
            continue
        itos.append(tok)
        if len(itos) >= max_size:
            break
    return {t: i for i, t in enumerate(itos)}, itos


def encode(records, indices, view, stoi, max_len):
    """Return (X, lengths, y) as padded int arrays, truncating to max_len."""
    X = np.full((len(indices), max_len), PAD_ID, dtype=np.int64)
    lengths = np.zeros(len(indices), dtype=np.int64)
    y = np.zeros(len(indices), dtype=np.int64)
    for row, i in enumerate(indices):
        toks = tokenize(field_text(records[i], view))[:max_len]
        ids = [stoi.get(t, UNK_ID) for t in toks]
        if not ids:
            ids = [UNK_ID]
        X[row, :len(ids)] = ids
        lengths[row] = len(ids)
        y[row] = LABEL2ID[records[i]["llm"]]
    return X, lengths, y


def iterate_batches(X, lengths, y, batch_size, shuffle=False, seed=0):
    order = np.arange(len(y))
    if shuffle:
        np.random.RandomState(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        sel = order[start:start + batch_size]
        yield X[sel], lengths[sel], y[sel]
