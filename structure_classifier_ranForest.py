import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

def load_and_label_csvs(folder_path, label):
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            df["label"] = label
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# טען נתונים
bavli_df = load_and_label_csvs("Data/csv_Bavli", "bavli")
yeru_df = load_and_label_csvs("Data/csv_Yerushalmi", "yerushalmi")
df = pd.concat([bavli_df, yeru_df], ignore_index=True)

# ניקוי
df = df[df["merged_lexicon"].notna()]
df = df[df["label"].notna()]

# איזון
min_count = min(df["label"].value_counts()["bavli"], df["label"].value_counts()["yerushalmi"])
bavli_balanced = df[df["label"] == "bavli"].sample(min_count, random_state=42)
yeru_balanced = df[df["label"] == "yerushalmi"].sample(min_count, random_state=42)
balanced_df = pd.concat([bavli_balanced, yeru_balanced], ignore_index=True)

# תכונות ותיוג
X = balanced_df["merged_lexicon"].astype(str)
y = balanced_df["label"]
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# פיצול
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)

# אימון
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# הערכה
y_pred = clf.predict(X_test)
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# דוגמאות
print("\n=== דוגמאות של מבנים מסווגים ===")
for i in range(10):
    print(f"מבנה: {X.iloc[i]} → תווית צפויה: {y.iloc[i]}")
