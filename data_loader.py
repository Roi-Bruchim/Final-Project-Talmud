# File: data_loader.py
import pandas as pd
import os

def load_data(filepath='processed_data.csv'):
    """
    Load the processed CSV file into a DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")

    df = pd.read_csv(filepath)
    return df