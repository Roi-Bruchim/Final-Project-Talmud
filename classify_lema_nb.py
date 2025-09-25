import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.pipeline import FeatureUnion
import numpy as np

"""
קובץ זה מבצע סיווג טקסטים תלמודיים (בבלי / ירושלמי) באמצעות שלושה מודלים של Naive Bayes.
הקלט: קבצי CSV עם עמודת 'text_transformed' שמכילה את הטקסט המעובד.
השלבים:
1. טעינת הנתונים משני מקורות (בבלי, ירושלמי) והוספת תוויות.
2. איזון הדאטה בין המחלקות.
3. חלוקה לסט אימון ובדיקה.
4. הפעלת שלושה מודלים:
   - NB + WORDS (CountVectorizer) – ייצוג על בסיס מילים ו־bi-grams.
   - NB + TF-IDF char 3–5 – ייצוג על בסיס רצפים של תווים.
   - NB + HYBRID – שילוב שני הווקטורים (מילים + תווים).
5. הצגת Confusion Matrix, דו"ח ביצועים (precision, recall, f1), ו־Top features לכל מחלקה.
מטרת הקוד: לזהות האם טקסט שייך לתלמוד הבבלי או הירושלמי ברמת דיוק גבוהה.
"""

# --- טעינה ---
def load_and_label_csvs(folder_path, label):
    dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, filename))
            if 'text_transformed' in df.columns:
                tmp = df[['text_transformed']].dropna().copy()
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

def run_nb(vectorizer, name, alpha=0.5):
    X_train = vectorizer.fit_transform(X_train_text)
    X_test  = vectorizer.transform(X_test_text)

    clf = MultinomialNB(alpha=alpha)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n=== {name} ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --- טופ פיצ'רים ---
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        log_prob = clf.feature_log_prob_
        classes = clf.classes_
        top_k = 25
        for ci, cls in enumerate(classes):
            top_idx = np.argsort(log_prob[ci])[::-1][:top_k]
            tops = feature_names[top_idx]
            print(f"\nTop {top_k} n-grams for class '{cls}':")
            print(", ".join(tops))
    except:
        print("Skipping top features display (not applicable for combined vectorizers).")

# --- מודל A: מילים ---
vec_words = CountVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    token_pattern=r"(?u)\b\w+\b"
)

# --- מודל B: תווים ---
vec_chars = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=5,
    sublinear_tf=True
)

# --- מודל C: היברידי ---
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
        ngram_range=(3, 5),
        min_df=5,
        sublinear_tf=True
    ))
])

# --- ריצות ---
run_nb(vec_words, "NB + WORDS (Count)", alpha=0.3)
run_nb(vec_chars, "NB + TF-IDF char 3–5", alpha=0.5)
run_nb(vec_combo, "NB + HYBRID (words + chars)", alpha=0.4)
