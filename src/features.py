"""
features.py — RQ4: what actually distinguishes the generators?

Two complementary angles:

1. `stylometrics` computes interpretable surface features (length, lexical
   diversity, formatting, punctuation habits) so we can describe *how* the
   generators differ in plain language, and test how far those features alone
   carry a classifier.

2. `ABLATIONS` are text transforms fed back through the neural models. If
   stripping a signal collapses accuracy, the models were leaning on it. This
   is the causal counterpart to the descriptive statistics above.
"""

import re

import numpy as np

from data import tokenize

# --- Ablation transforms -----------------------------------------------------
# Each takes raw text and returns text with one family of signals removed.

MD_BULLET = re.compile(r"(?m)^[ \t]*([*\-+•]|\d+[.)])[ \t]+")
MD_HEADING = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]*")
MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`{1,3})")


def strip_formatting(text):
    """Remove markdown scaffolding: bullets, headings, emphasis, blank lines.

    Targets *layout* habits (does the model like bulleted lists and bold
    headers?) while leaving wording intact.
    """
    text = MD_HEADING.sub("", text)
    text = MD_BULLET.sub("", text)
    text = MD_EMPHASIS.sub("", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def strip_punctuation(text):
    """Remove punctuation, keeping words, digits and spacing."""
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def truncate_words(n):
    """Clip every document to the same word count, neutralising length.

    Response length is usually the single strongest giveaway, and a model can
    exploit it without learning anything stylistic. Equalising it tests whether
    a fingerprint survives.
    """
    def _f(text):
        return " ".join(text.split()[:n])
    _f.__name__ = f"truncate_{n}"
    return _f


def shuffle_words(seed=0):
    """Shuffle word order, destroying syntax but preserving vocabulary.

    Separates "which words does this model pick" from "how does it arrange
    them". A CNN reading n-grams should suffer more than a bag-of-words view.
    """
    rng = np.random.RandomState(seed)

    def _f(text):
        words = text.split()
        rng.shuffle(words)
        return " ".join(words)
    return _f


ABLATIONS = {
    "full": None,
    "no_formatting": strip_formatting,
    "no_punctuation": strip_punctuation,
    "length_capped_100w": truncate_words(100),
    "shuffled_words": shuffle_words(0),
}


# --- Descriptive stylometrics ------------------------------------------------

FUNCTION_WORDS = [
    "the", "a", "an", "and", "but", "or", "if", "because", "as", "of", "to",
    "in", "for", "with", "that", "this", "it", "is", "are", "was", "were",
    "you", "your", "we", "i", "they", "however", "therefore", "also",
    "additionally", "overall", "furthermore", "moreover",
]


def stylometrics(text):
    """Interpretable surface features for one document."""
    toks = tokenize(text)
    words = [t for t in toks if t.isalpha()]
    n_words = len(words)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    lines = text.split("\n")

    if n_words == 0:
        return {k: 0.0 for k in FEATURE_NAMES}

    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    freqs = np.array(list(counts.values()), dtype=float)
    probs = freqs / freqs.sum()

    feats = {
        "n_words": float(n_words),
        "n_chars": float(len(text)),
        "n_sentences": float(len(sentences)),
        "mean_sentence_len": float(n_words / max(len(sentences), 1)),
        "mean_word_len": float(np.mean([len(w) for w in words])),
        # Type-token ratio is length-sensitive, so we also report the
        # length-robust root TTR for fairer cross-model comparison.
        "ttr": float(len(counts) / n_words),
        "root_ttr": float(len(counts) / np.sqrt(n_words)),
        "entropy": float(-(probs * np.log2(probs)).sum()),
        "hapax_ratio": float((freqs == 1).sum() / len(freqs)),
        # Formatting / layout
        "n_newlines": float(text.count("\n")),
        "blank_line_ratio": float(sum(1 for l in lines if not l.strip()) / max(len(lines), 1)),
        "bullet_lines": float(len(MD_BULLET.findall(text))),
        "heading_lines": float(len(MD_HEADING.findall(text))),
        "emphasis_marks": float(len(MD_EMPHASIS.findall(text))),
        # Punctuation habits
        "comma_per_100w": float(text.count(",") / n_words * 100),
        "semicolon_per_100w": float(text.count(";") / n_words * 100),
        "colon_per_100w": float(text.count(":") / n_words * 100),
        "question_per_100w": float(text.count("?") / n_words * 100),
        "exclaim_per_100w": float(text.count("!") / n_words * 100),
        "quote_per_100w": float((text.count('"') + text.count("'")) / n_words * 100),
        "dash_per_100w": float((text.count("—") + text.count(" - ")) / n_words * 100),
        "uppercase_ratio": float(sum(1 for c in text if c.isupper()) / max(len(text), 1)),
        "digit_ratio": float(sum(1 for c in text if c.isdigit()) / max(len(text), 1)),
    }
    for fw in FUNCTION_WORDS:
        feats[f"fw_{fw}"] = float(counts.get(fw, 0) / n_words * 100)
    return feats


_probe = {
    "n_words": 0, "n_chars": 0, "n_sentences": 0, "mean_sentence_len": 0,
    "mean_word_len": 0, "ttr": 0, "root_ttr": 0, "entropy": 0,
    "hapax_ratio": 0, "n_newlines": 0, "blank_line_ratio": 0,
    "bullet_lines": 0, "heading_lines": 0, "emphasis_marks": 0,
    "comma_per_100w": 0, "semicolon_per_100w": 0, "colon_per_100w": 0,
    "question_per_100w": 0, "exclaim_per_100w": 0, "quote_per_100w": 0,
    "dash_per_100w": 0, "uppercase_ratio": 0, "digit_ratio": 0,
}
FEATURE_NAMES = list(_probe) + [f"fw_{w}" for w in FUNCTION_WORDS]


def feature_matrix(records, indices, field="output"):
    """Stack stylometric features for a set of records."""
    rows = [stylometrics(records[i][field]) for i in indices]
    X = np.array([[r[name] for name in FEATURE_NAMES] for r in rows],
                 dtype=np.float64)
    return X, FEATURE_NAMES
