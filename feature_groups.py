# feature_groups.py

def get_feature_sets():
    feature_sets = {
        'all_features': [
            'Lema', 'merged_lexicon', 'merged_meanings',
            'lexicon_0', 'lexicon_1', 'lexicon_2', 'lexicon_3', 'lexicon_4'
        ],
        'only_lema': ['Lema'],
        'only_lexicon': ['lexicon_0', 'lexicon_1', 'lexicon_2', 'lexicon_3', 'lexicon_4'],
        'only_merged_lexicon': ['merged_lexicon'],
        'only_meanings': ['merged_meanings'],
        'lema_and_lexicon': ['Lema', 'merged_lexicon'],
        'no_meanings': ['Lema', 'merged_lexicon', 'lexicon_0', 'lexicon_1']
    }
    return feature_sets
