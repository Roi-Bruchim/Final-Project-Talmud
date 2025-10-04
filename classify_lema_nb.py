import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# --- טעינה ---
def load_and_label_csvs(folder_path, label):
    dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, filename))
            if 'text_transformed' in df.columns:
                tmp = df[['text_transformed']].dropna().copy()
                # אם קיימות עמודות נוספות – נצרף אותן כטוקנים לתוך הטקסט (מותנה בקיום)
                extra_cols = []
                for col in ['Lema', 'merged_lexicon', 'lexicon_0', 'lexicon_1', 'lexicon_2']:
                    if col in df.columns:
                        extra_cols.append(col)
                if extra_cols:
                    # נהפוך כל ערך לעמודה מקבילה עם prefix לזיהוי מקור הפיצ'ר
                    for col in extra_cols:
                        # נוודא מחרוזות
                        df[col] = df[col].astype(str).fillna("")
                    # מחרוזת מועשרת: טקסט + למות/תגיות כטוקנים מסומנים
                    enriched = df['text_transformed'].astype(str)
                    for col in extra_cols:
                        enriched = enriched + " " + (df[col].astype(str).map(lambda s: " ".join(f"{col.upper()}_{tok}" for tok in s.split())))
                    tmp['text_transformed'] = enriched

                tmp['label'] = label
                dfs.append(tmp)
    if not dfs:
        raise ValueError(f"No valid CSV files with 'text_transformed' found in {folder_path}")
    return pd.concat(dfs, ignore_index=True)

bavli_df = load_and_label_csvs("Data/csv_Bavli", "bavli")
yeru_df  = load_and_label_csvs("Data/csv_Yerushalmi", "yerushalmi")
df = pd.concat([bavli_df, yeru_df], ignore_index=True)

# --- איזון ---
min_count = df['label'].value_counts().min()
df_balanced = pd.concat([
    resample(df[df['label'] == 'bavli'],      replace=False, n_samples=min_count, random_state=42),
    resample(df[df['label'] == 'yerushalmi'], replace=False, n_samples=min_count, random_state=42)
], ignore_index=True)

X_text = df_balanced['text_transformed'].astype(str)
y = df_balanced['label']

# --- פיצול ---
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate(name, clf, X_train_text, X_test_text, y_train, y_test):
    clf.fit(X_train_text, y_train)
    y_pred = clf.predict(X_test_text)
    print(f"\n=== {name} ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

# --- וקטוריזציות בסיסיות ---
vec_words = CountVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    token_pattern=r"(?u)\b\w+\b"
)

vec_chars = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=5,
    sublinear_tf=True
)

vec_combo = FeatureUnion([
    ("words", CountVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        token_pattern=r"(?u)\b\w+\b"
    )),
    ("chars", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6), 
        min_df=5,
        sublinear_tf=True
    ))
])

# --- מודל A: NB + WORDS ---
model_A = Pipeline([
    ("vec", vec_words),
    ("sel", SelectKBest(chi2, k=200000)), 
    ("clf", MultinomialNB(alpha=0.3))
])

# --- מודל B: NB + TF-IDF chars ---
model_B = Pipeline([
    ("vec", vec_chars),
    ("sel", SelectKBest(chi2, k=250000)),
    ("clf", ComplementNB(alpha=0.5))   
])

# --- מודל C: HYBRID (words + chars) + NB ---
model_C = Pipeline([
    ("vec", vec_combo),
    ("sel", SelectKBest(chi2, k=300000)),
    ("clf", MultinomialNB(alpha=0.4))
])

# === מודל D (החזק): Linear SVM + קליברציה ===

svm_base = SGDClassifier(
    loss="hinge",            
    alpha=1e-5,              
    penalty="l2",
    max_iter=2000,
    random_state=42,
    n_jobs=-1
)

model_D = Pipeline([
    ("vec", FeatureUnion([
        ("words", TfidfVectorizer(  # משתמשים ב-TF-IDF למילים עבור SVM
            analyzer="word",
            ngram_range=(1, 3),     # מרחיבים עד טריגרמות
            min_df=3,
            max_df=0.95,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )),
        ("chars", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            min_df=5,
            sublinear_tf=True
        ))
    ])),
    ("sel", SelectKBest(chi2, k=350000)),
    ("svm", CalibratedClassifierCV(estimator=svm_base, cv=3, method="isotonic"))  
])

# --- אנצמבל אופציונלי: ממוצע הסתברויות של A,B,D ---
# שים לב: רק מודלים עם predict_proba ישתתפו (NB / SVM עם קליברציה)
ensemble = VotingClassifier(
    estimators=[
        ("nb_words", model_A),
        ("nb_chars", model_B),
        ("svm_cal", model_D),
    ],
    voting="soft",
    weights=[1, 1, 2]   # נותנים יותר משקל ל-SVM
)

# --- הרצות ---
evaluate("NB + WORDS (Count) + SelectKBest", model_A, X_train_text, X_test_text, y_train, y_test)
evaluate("NB + TF-IDF char 3–6 + SelectKBest (ComplementNB)", model_B, X_train_text, X_test_text, y_train, y_test)
evaluate("NB + HYBRID (words+chars) + SelectKBest", model_C, X_train_text, X_test_text, y_train, y_test)
evaluate("MODEL D: Linear SVM (SGD) + Calibrated + SelectKBest", model_D, X_train_text, X_test_text, y_train, y_test)
evaluate("SOFT ENSEMBLE: (NB-words + NB-chars + SVM)", ensemble, X_train_text, X_test_text, y_train, y_test)

# --- הצגת top-פיצ'רים למחלקות (ל-NB בלבד) ---
def show_top_features_nb(pipeline, name, top_k=25):
    try:
        vec = pipeline.named_steps["vec"]
        sel = pipeline.named_steps["sel"]
        clf = pipeline.named_steps["clf"]
        # שחזור שמות פיצ'רים אחרי ה-FeatureUnion
        if isinstance(vec, FeatureUnion):
            feats = []
            for subname, subvec in vec.transformer_list:
                sub_feats = np.array(subvec.get_feature_names_out())
                # מוסיפים prefix לפי המקור כדי להבין מאיפה זה הגיע
                feats.append(np.array([f"{subname}:{f}" for f in sub_feats]))
            feature_names = np.concatenate(feats)
        else:
            feature_names = np.array(vec.get_feature_names_out())

        # SelectKBest שומר subset; נחלץ את המיפוי חזרה
        mask = sel.get_support(indices=True)
        selected_names = feature_names[mask]

        log_prob = clf.feature_log_prob_
        classes = clf.classes_
        for ci, cls in enumerate(classes):
            top_idx = np.argsort(log_prob[ci])[::-1][:top_k]
            tops = selected_names[top_idx]
            print(f"\nTop {top_k} n-grams for class '{cls}' in {name}:")
            print(", ".join(tops))
    except Exception as e:
        print(f"[{name}] Skipping top-features: {e}")

show_top_features_nb(model_A, "NB+WORDS")
show_top_features_nb(model_B, "NB+CHARS")
show_top_features_nb(model_C, "NB+HYBRID")
