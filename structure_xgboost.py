import os, re, warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

warnings.filterwarnings("ignore")

def load_and_label_csvs(folder_path, label):
    dfs = []
    for fn in os.listdir(folder_path):
        if fn.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, fn))
            df["label"] = label
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# --- טען ואחד ---
df_bavli = load_and_label_csvs("Data/csv_Bavli", "bavli")
df_yeru  = load_and_label_csvs("Data/csv_Yerushalmi", "yerushalmi")
df = pd.concat([df_bavli, df_yeru], ignore_index=True)

# --- שמור עמודה + ניקוי רעש ---
df = df[["merged_lexicon", "label"]].dropna()
df = df[df["merged_lexicon"].str.strip().astype(bool)]
df = df[~df["merged_lexicon"].str.contains(r"^\s*No data\s*$", na=False, case=False)]

# --- איזון בסיסי (כמו אצלך) ---
min_count = df["label"].value_counts().min()
df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(min_count, random_state=42))

# --- Tokenization: מפרק ב־" | ", שומר multi-word כטוקן יחיד עם '_' ---
def normalize_lex(s: str) -> str:
    parts = [p.strip() for p in re.split(r"\s*\|\s*", str(s)) if p.strip()]
    tokens = [re.sub(r"\s+", "_", p) for p in parts]  # "preposition with pronominal suffix" -> "preposition_with_pronominal_suffix"
    return " ".join(tokens)

df["lex_norm"] = df["merged_lexicon"].map(normalize_lex)

# --- וקטוריזציה: TF-IDF על טאגים + bi-grams של רצפים מבניים ---
vec = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"[^\s]+",   # כל טוקן הוא רצף ללא רווח (כבר שמנו _)
    ngram_range=(1, 2),        # uni + bi-grams של הטאגים
    min_df=2
)
X = vec.fit_transform(df["lex_norm"])

# --- תוויות + split מסווג ---
le = LabelEncoder()
y = le.fit_transform(df["label"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- מודל 1: XGBoost עם early stopping ---
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

num_classes = len(le.classes_)
params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "seed": 42,
    "nthread": 4
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, "train"), (dtest, "valid")],
    early_stopping_rounds=50,
    verbose_eval=False
)

y_pred = bst.predict(dtest).argmax(axis=1)
print("=== Confusion Matrix (XGBoost) ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== Classification Report (XGBoost) ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- מודל 2 (אופציונלי להשוואה מהירה): Logistic Regression לינארי על אותו X ---
logreg = LogisticRegression(max_iter=2000, n_jobs=4)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
print("\n=== Confusion Matrix (LogReg) ===")
print(confusion_matrix(y_test, y_pred_lr))
print("\n=== Classification Report (LogReg) ===")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# --- דוגמאות של טוקנים לאחר הנרמול ---
print("\n=== דוגמאות לטאגים מנורמלים ===")
for i in range(10):
    print(df.iloc[i]["lex_norm"])
