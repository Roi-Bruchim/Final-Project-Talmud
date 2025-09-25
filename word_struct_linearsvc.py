#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
word_struct_linearsvc.py
========================
מסווג **בבלי/ירושלמי לפי מילה + מבנה** (שילוב שתי התכונות) באמצעות **LinearSVC** (SVM לינארי).
- טקסט: `text_transformed` → TF‑IDF word uni/bi + TF‑IDF char 3–5
- מבנה: `merged_lexicon` → פיצול לפי `|` ואיחוד multi‑word עם `_`, TF‑IDF על טאגים (uni/bi)
- איחוד פיצ'רים עם ColumnTransformer → מסווג LinearSVC (class_weight='balanced')
- פיצול מסווג, איזון בין מחלקות, דוחות מלאים ושמירת ארטיפקטים עם joblib

דוגמה להרצה:
python word_struct_linearsvc.py \
  --bavli_dir Data/csv_Bavli --yeru_dir Data/csv_Yerushalmi \
  --out_dir models/word_struct_linearsvc --no_balance 0

קבצים נשמרים:
- model.joblib, vectorizers.joblib, label_encoder.joblib, report.txt
"""

import os
import re
import json
import argparse
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import resample
import joblib

# -------------------------- IO & Preprocess --------------------------

def load_and_label_csvs(folder_path: str, label: str) -> pd.DataFrame:
    dfs = []
    for fn in os.listdir(folder_path):
        if not fn.lower().endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(folder_path, fn))
        # נוודא שיש לפחות אחת מהעמודות שנצטרך
        has_text = "text_transformed" in df.columns
        has_struct = "merged_lexicon" in df.columns
        if not (has_text or has_struct):
            continue
        keep_cols = []
        if has_text:
            keep_cols.append("text_transformed")
        if has_struct:
            keep_cols.append("merged_lexicon")
        tmp = df[keep_cols].copy()
        tmp["label"] = label
        dfs.append(tmp)
    if not dfs:
        raise ValueError(f"No suitable CSVs found in {folder_path}")
    return pd.concat(dfs, ignore_index=True)


def normalize_lex(s: str) -> str:
    """פיצול לפי | והחלפת רווחים ב- _ כך שכל טאג מרובה מילים יהפוך לטוקן יחיד."""
    parts = [p.strip() for p in re.split(r"\s*\|\s*", str(s)) if p and p.strip()]
    tokens = [re.sub(r"\s+", "_", p) for p in parts]
    return " ".join(tokens)


def clean_and_balance(df: pd.DataFrame, seed: int, do_balance: bool) -> pd.DataFrame:
    # נורמליזציה בסיסית
    if "text_transformed" not in df.columns:
        df["text_transformed"] = ""
    if "merged_lexicon" not in df.columns:
        df["merged_lexicon"] = ""

    df = df.dropna(subset=["label"]).copy()
    df["text_transformed"] = df["text_transformed"].astype(str)
    df["merged_lexicon"] = df["merged_lexicon"].astype(str)

    # סינון "No data" ו/או ריקים
    mask_nodata = df["merged_lexicon"].str.fullmatch(r"\s*No data\s*", na=False, case=False)
    df = df[~mask_nodata]
    df = df[(df["text_transformed"].str.strip().astype(bool)) | (df["merged_lexicon"].str.strip().astype(bool))]

    if do_balance:
        minc = df["label"].value_counts().min()
        df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(minc, random_state=seed)).reset_index(drop=True)
    return df


# -------------------------- Feature Builders --------------------------

def build_column_transformer() -> ColumnTransformer:
    # טקסט: מילים (uni/bi)
    vec_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b"
    )

    # טקסט: תווים (3–5)
    vec_char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        sublinear_tf=True
    )

    # מבנה: טאגים לאחר normalize_lex (uni/bi)
    vec_struct = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^\s]+",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    ct = ColumnTransformer(
        transformers=[
            ("word_uni_bi", vec_word, "text_transformed"),
            ("char_3_5", vec_char, "text_transformed"),
            ("struct_uni_bi", vec_struct, "lex_norm"),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return ct


# -------------------------- Model Helpers --------------------------

def get_feature_names(ct: ColumnTransformer) -> List[str]:
    names = []
    for name, trans, cols in ct.transformers_:
        if name == 'remainder' or trans == 'drop':
            continue
        try:
            feats = trans.get_feature_names_out()
        except Exception:
            # scikit versions where get_feature_names_out needs input feature names
            feats = trans.get_feature_names_out(None)
        # תייג כל וקטורייזר בפרפיקס כדי שתדע מאיפה הגיע
        feats = [f"{name}__{f}" for f in feats]
        names.extend(feats)
    return names


def top_features_linear(coef: np.ndarray, feature_names: List[str], k: int = 30):
    # Binary OVR: coef shape (1, n_features). חיובי -> מחלקה חיובית, שלילי -> מחלקה שלילית
    w = coef.ravel()
    pos_idx = np.argsort(w)[-k:][::-1]
    neg_idx = np.argsort(w)[:k]
    top_pos = [(feature_names[i], float(w[i])) for i in pos_idx]
    top_neg = [(feature_names[i], float(w[i])) for i in neg_idx]
    return top_pos, top_neg


# -------------------------- Main --------------------------

def main(args):
    # --- Load ---
    bavli = load_and_label_csvs(args.bavli_dir, "bavli")
    yeru  = load_and_label_csvs(args.yeru_dir,  "yerushalmi")
    df = pd.concat([bavli, yeru], ignore_index=True)

    # --- Normalize structure field ---
    df["lex_norm"] = df.get("merged_lexicon", "").astype(str).map(normalize_lex)

    # --- Clean & balance ---
    df = clean_and_balance(df, seed=args.seed, do_balance=(args.no_balance == 0))

    # --- Labels ---
    le = LabelEncoder()
    y = le.fit_transform(df["label"])  # 0..C-1

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        df[["text_transformed", "lex_norm"]], y,
        test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # --- Features + Model ---
    ct = build_column_transformer()
    clf = LinearSVC(C=1.0, class_weight='balanced', random_state=args.seed)

    pipe = Pipeline([
        ("features", ct),
        ("clf", clf)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # --- Reports ---
    rep = classification_report(y_test, y_pred, target_names=list(le.classes_))
    cm = confusion_matrix(y_test, y_pred)

    print("=== Confusion Matrix ===")
    print(cm)
    print("\n=== Classification Report ===")
    print(rep)

    # --- Top features by class (binary friendly) ---
    try:
        ct_fitted = pipe.named_steps["features"]
        fnames = get_feature_names(ct_fitted)
        coef = pipe.named_steps["clf"].coef_
        if coef.shape[0] == 1 and len(le.classes_) == 2:
            top_pos, top_neg = top_features_linear(coef, fnames, k=40)
            cls_pos = le.classes_[1]
            cls_neg = le.classes_[0]
            print(f"\nTop 40 features for class '{cls_pos}' (positive weights):")
            print(", ".join([f"{t}" for t,_ in top_pos]))
            print(f"\nTop 40 features for class '{cls_neg}' (negative weights):")
            print(", ".join([f"{t}" for t,_ in top_neg]))
    except Exception as e:
        print(f"[Warn] Could not compute top features: {e}")

    # --- Save artifacts ---
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.out_dir, "model.joblib"))
    joblib.dump(le, os.path.join(args.out_dir, "label_encoder.joblib"))

    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("=== Confusion Matrix ===\n")
        f.write(np.array2string(cm))
        f.write("\n\n=== Classification Report ===\n")
        f.write(rep)

    # Bonus: שמור גם את רשימות ה-top-features אם נבנו
    try:
        if coef.shape[0] == 1 and len(le.classes_) == 2:
            with open(os.path.join(args.out_dir, "top_features.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "positive_class": le.classes_[1],
                    "negative_class": le.classes_[0],
                    "top_positive": top_pos,
                    "top_negative": top_neg
                }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(f"\nArtifacts saved to: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bavli_dir", type=str, default="Data/csv_Bavli")
    parser.add_argument("--yeru_dir",  type=str, default="Data/csv_Yerushalmi")
    parser.add_argument("--out_dir",   type=str, default="models/word_struct_linearsvc")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no_balance", type=int, default=0, help="1=אל תבצע איזון, 0=איזון למחלקה המינימלית")
    args = parser.parse_args()
    main(args)
