# File: main.py

from data_loader import load_data
from feature_groups import get_feature_sets



def main():
    # Load data
    df = load_data('processed_data.csv')  # או נתיב אחר לקובץ המעובד שלכם
    
    # Define feature sets
    feature_sets = get_feature_sets()
    
    # Run model on each feature set
    print("Running model evaluations...")
    results_df = run_feature_tests(df, feature_sets, label_column='Label')
    
    # Show results table
    print(results_df.sort_values(by='accuracy', ascending=False))
    
    # Plot performance
    plot_feature_scores(results_df)
    
    # Plot clusters for top 2 feature sets
    top_sets = results_df.sort_values(by='accuracy', ascending=False).head(2)['feature_set']
    for name in top_sets:
        print(f"Plotting clusters for feature set: {name}")
        plot_clusters(df, feature_sets[name], method='tsne')

if __name__ == "__main__":
    main()
