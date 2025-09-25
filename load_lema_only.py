import os
import pandas as pd

# הגדרות נתיב
BAVLI_PATH = 'data/csv_Bavli'
YERUSHALMI_PATH = 'data/csv_Yerushalmi'


def load_lema_data():
    bavli_df_list = []
    yerushalmi_df_list = []

    # טעינת קבצי הבבלי
    for filename in os.listdir(BAVLI_PATH):
        if filename.endswith('.csv'):
            filepath = os.path.join(BAVLI_PATH, filename)
            df = pd.read_csv(filepath)
            if 'Lema' in df.columns:
                df = df[['Lema']].copy()
                df['label'] = 'bavli'
                bavli_df_list.append(df)

    # טעינת קבצי הירושלמי
    for filename in os.listdir(YERUSHALMI_PATH):
        if filename.endswith('.csv'):
            filepath = os.path.join(YERUSHALMI_PATH, filename)
            df = pd.read_csv(filepath)
            if 'Lema' in df.columns:
                df = df[['Lema']].copy()
                df['label'] = 'yerushalmi'
                yerushalmi_df_list.append(df)

    # איחוד כל הדאטא
    df_all = pd.concat(bavli_df_list + yerushalmi_df_list, ignore_index=True)

    # הסרת שורות ריקות
    df_all.dropna(inplace=True)

    return df_all

if __name__ == "__main__":
    df = load_lema_data()
    print(df.sample(10))
    print(f"\nTotal rows: {len(df)}")
