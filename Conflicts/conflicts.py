import pandas as pd
from collections import Counter

# Load data for df_zev and other DataFrames from CSV files
df_zev = pd.read_csv('df_zev_split_csv.csv', encoding='utf-8')
df_ned = pd.read_csv('df_ned_split_csv.csv', encoding='utf-8')
df_naz = pd.read_csv('df_naz_split_csv.csv', encoding='utf-8')
df_meil = pd.read_csv('df_meil_split_csv.csv', encoding='utf-8')
df_ker = pd.read_csv('df_ker_split_csv.csv', encoding='utf-8')
df_tam = pd.read_csv('df_tam_split_csv.csv', encoding='utf-8')

# List of other DataFrames with their names to compare against df_zev
other_dfs = {
    'ned': df_ned,
    'naz': df_naz,
    'meil': df_meil,
    'ker': df_ker,
    'tam': df_tam
}

def get_last_split_word(row):
    """
    Extracts the last non-empty split word from split_word columns.
    """
    split_words = [row[f'split_word_{i}'] for i in range(4) if
                   pd.notna(row[f'split_word_{i}']) and row[f'split_word_{i}'].strip()]
    return split_words[-1] if split_words else None

# Loop through each DataFrame in other_dfs and perform comparisons
for df_name, other_df in other_dfs.items():
    # Initialize a counter and dictionary to keep track of pairs and meanings
    pair_counter = Counter()
    meanings_dict = {}  # Store merged_meanings and transformed texts for each pair

    # Loop through rows in df_zev and the current other DataFrame
    for _, row1 in df_zev.iterrows():
        for _, row2 in other_df.iterrows():
            # Check if merged meanings match and Lemmas are different
            if row1['merged_meanings'] == row2['merged_meanings'] and row1['Lema'] != row2['Lema']:
                # Extract last non-empty split words
                last_word_1 = get_last_split_word(row1)
                last_word_2 = get_last_split_word(row2)

                # Ensure last words are not equal
                if last_word_1 != last_word_2:
                    # Create a sorted tuple of the text values to avoid duplicate pairs
                    pair = tuple(sorted([row1['text'], row2['text']]))
                    pair_counter[pair] += 1

                    # Store meanings and transformed texts for each pair
                    meanings_dict[pair] = {
                        'meaning': row1['merged_meanings'],
                        'transformed_text_1': row1['text_transformed'],
                        'transformed_text_2': row2['text_transformed']
                    }

    # Write the results to a separate file for each comparison
    with open(f'conflict_{df_name}.txt', 'w', encoding='utf-8-sig') as file:
        for pair, count in pair_counter.items():
            # Retrieve meaning and transformed texts for each pair
            meaning = meanings_dict[pair]['meaning']
            transformed_text_1 = meanings_dict[pair]['transformed_text_1']
            transformed_text_2 = meanings_dict[pair]['transformed_text_2']

            file.write(f"Conflict found for meaning '{meaning}': Pair {pair} counter={count}\n")
            file.write(f"Transformed text 1: '{transformed_text_1}'\n")
            file.write(f"Transformed text 2: '{transformed_text_2}'\n")
            file.write("------\n")

    print(f"Results have been written to conflict_{df_name}.txt with UTF-8 encoding")
