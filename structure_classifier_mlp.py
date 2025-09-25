#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_and_label_csvs(folder_path, label_value):
    dfs = []
    for fn in os.listdir(folder_path):
        if fn.lower().endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, fn))
            if "merged_lexicon" not in df.columns:
                continue
            tmp = df[["merged_lexicon"]].copy()
            tmp["label"] = label_value
            dfs.append(tmp)
    if not dfs:
        raise ValueError(f"No CSV files with 'merged_lexicon' found in {folder_path}")
    return pd.concat(dfs, ignore_index=True)

def clean_and_balance(df, seed=42):
    df = df.dropna(subset=["merged_lexicon", "label"]).copy()
    df["merged_lexicon"] = df["merged_lexicon"].astype(str)
    df = df[~df["merged_lexicon"].str.fullmatch(r"\s*No data\s*", na=False, case=False)]
    df = df[df["merged_lexicon"].str.strip().astype(bool)]
    minc = df["label"].value_counts().min()
    return df.groupby("label", group_keys=False).apply(lambda x: x.sample(minc, random_state=seed)).reset_index(drop=True)

def normalize_lex(s):
    parts = [p.strip() for p in re.split(r"\s*\|\s*", s) if p.strip()]
    return " ".join(re.sub(r"\s+", "_", p) for p in parts)

def main(args):
    bavli_df = load_and_label_csvs(args.bavli_dir, "bavli")
    yeru_df  = load_and_label_csvs(args.yeru_dir, "yerushalmi")
    df = pd.concat([bavli_df, yeru_df], ignore_index=True)
    df = clean_and_balance(df, seed=args.seed)

    df["lex_norm"] = df["merged_lexicon"].map(normalize_lex)

    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^\s]+",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X = vec.fit_transform(df["lex_norm"])

    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=args.seed,
        early_stopping=True,
        n_iter_no_change=10
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.out_dir, "mlp_model.joblib"))
    joblib.dump(vec, os.path.join(args.out_dir, "vectorizer.joblib"))
    joblib.dump(le, os.path.join(args.out_dir, "label_encoder.joblib"))
    print(f"\nModel and artifacts saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bavli_dir", type=str, default="Data/csv_Bavli")
    parser.add_argument("--yeru_dir",  type=str, default="Data/csv_Yerushalmi")
    parser.add_argument("--out_dir",   type=str, default="models/structure_mlp_sklearn")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()
    main(args)
