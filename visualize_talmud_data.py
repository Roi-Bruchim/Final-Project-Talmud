#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# ---------- Load data (includes structure if available) ----------
def load_and_label(folder: str, label: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "text_transformed" not in df.columns:
            continue

        cols = ["text_transformed"]
        if "merged_lexicon" in df.columns:
            cols.append("merged_lexicon")

        tmp = df[cols].dropna(how="all").copy()
        if "text_transformed" not in tmp.columns:
            continue

        tmp["text_transformed"] = tmp["text_transformed"].astype(str)
        if "merged_lexicon" in tmp.columns:
            tmp["merged_lexicon"] = tmp["merged_lexicon"].astype(str)
        else:
            tmp["merged_lexicon"] = np.nan

        tmp["label"] = label
        tmp["source_file"] = os.path.basename(path)  # keep real file name
        rows.append(tmp)

    if not rows:
        raise ValueError(f"No CSVs with 'text_transformed' under: {folder}")
    return pd.concat(rows, ignore_index=True)

# ---------- Structure normalization & filtering ----------
# split "conjunction | verb peal | noun sg. emphatic" -> tokens
_SPLIT_PIPE = re.compile(r"\s*\|\s*")

# keep only coarse POS-like categories (case-insensitive)
ALLOWED_PREFIXES = {
    "noun", "verb", "conjunction", "preposition", "adverb",
    "pronoun", "proper_name", "numeral", "participle", "particle",
    "interjection"
}

def is_relevant_structure_token(tok: str) -> bool:
    """Return True if token belongs to a coarse structural category."""
    if not tok:
        return False
    t = tok.lower()
    # discard obvious code-y tags
    if re.fullmatch(r"[a-z]?\d+[a-z]?(\s.*)?", t):  # e.g., a01, p01, v2
        return False
    # allow only tokens whose first chunk (before '_' or space) is in whitelist
    head = re.split(r"[_\s]", t, maxsplit=1)[0]
    return head in ALLOWED_PREFIXES

def normalize_structure_keep_relevant(s: str) -> str:
    """
    Normalize merged_lexicon to space-separated tokens (underscored multi-words),
    then keep only relevant coarse categories (NOUN/VERB/...).
    """
    if not isinstance(s, str):
        return ""
    if not s or s.strip().lower() == "no data":
        return ""
    parts = [p.strip() for p in _SPLIT_PIPE.split(s) if p.strip()]
    parts = [re.sub(r"\s+", "_", p) for p in parts]   # multi-word -> single token
    parts = [p for p in parts if is_relevant_structure_token(p)]
    return " ".join(parts)

# ---------- Top-N utility & plotting ----------
def top_terms(texts: pd.Series, n: int = 25, ngram_range=(1,1), token_pattern=r"(?u)\b\w+\b"):
    vec = CountVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        token_pattern=token_pattern,
        lowercase=False,
        min_df=1
    )
    X = vec.fit_transform(texts.fillna(""))
    sums = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    order = np.argsort(sums)[::-1][:n]
    terms = vocab[order]
    counts = sums[order]
    return list(zip(terms, counts))

def plot_top_bar(pairs, title, color="steelblue"):
    terms, counts = zip(*pairs) if pairs else ([], [])
    plt.figure(figsize=(10, 6))
    y = np.arange(len(terms))
    plt.barh(y, counts, color=color)
    plt.yticks(y, terms)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.show()

# ---------- Load ----------
bavli = load_and_label("Data/csv_Bavli", "bavli")
yeru  = load_and_label("Data/csv_Yerushalmi", "yerushalmi")
df = pd.concat([bavli, yeru], ignore_index=True)
df["text_transformed"] = df["text_transformed"].astype(str)

# Prepare normalized & filtered structure column (if any)
if "merged_lexicon" in df.columns:
    df["structure_norm"] = df["merged_lexicon"].map(normalize_structure_keep_relevant)
else:
    df["structure_norm"] = ""

print("Counts per label:\n", df["label"].value_counts())

# ---------- 1) Distribution of samples per label (Pie chart) ----------
counts = df["label"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#1f77b4", "#2ca02c"]
)
plt.title("Distribution of Samples (Bavli vs Yerushalmi)")
plt.tight_layout()
plt.show()

# ---------- 2) Top-25 WORDS per label (counts) ----------
def top_words(texts: pd.Series, n: int = 25):
    return top_terms(texts, n=n, ngram_range=(1,1), token_pattern=r"(?u)\b\w+\b")

plot_top_bar(top_words(df[df["label"]=="bavli"]["text_transformed"], n=25),
             "Top 25 Words — Bavli", color="#1f77b4")
plot_top_bar(top_words(df[df["label"]=="yerushalmi"]["text_transformed"], n=25),
             "Top 25 Words — Yerushalmi", color="#2ca02c")

# ---------- 3) Top-25 STRUCTURE TAGS per label (filtered to NOUN/VERB/...) ----------
has_structure = df["structure_norm"].str.strip().astype(bool).any()
if has_structure:
    token_pattern_struct = r"[^\s]+"  # tokens already space-separated
    plot_top_bar(
        top_terms(df[df["label"]=="bavli"]["structure_norm"], n=25,
                  ngram_range=(1,1), token_pattern=token_pattern_struct),
        "Top 25 Structure Tags (coarse) — Bavli", color="#1f77b4"
    )
    plot_top_bar(
        top_terms(df[df["label"]=="yerushalmi"]["structure_norm"], n=25,
                  ngram_range=(1,1), token_pattern=token_pattern_struct),
        "Top 25 Structure Tags (coarse) — Yerushalmi", color="#2ca02c"
    )
else:
    print("\n[INFO] No 'merged_lexicon' found or no relevant structure tokens — skipping structure plots.")

# ---------- 4) Rows per source file (real CSV names) ----------
per_file = (
    df.groupby(["label", "source_file"])
      .size()
      .reset_index(name="rows")
      .sort_values("rows", ascending=False)
)

print("\nTop files by number of rows:\n", per_file.head(10))

pivot = per_file.pivot(index="source_file", columns="label", values="rows").fillna(0)
plt.figure(figsize=(12, max(6, 0.25 * len(pivot))))
pivot.plot(kind="barh", stacked=False, figsize=(12, max(6, 0.25 * len(pivot))))
plt.gca().invert_yaxis()
plt.xlabel("Number of rows")
plt.title("Rows per source file (bavli / yerushalmi)")
plt.tight_layout()
plt.show()
