# data_init.py
# Dataset Cleaning Functions for Project Analysis

import pandas as pd

def get_cleaned_dataset1():
    data = pd.read_csv("Datasets/dataset1.csv")
    data = data.drop_duplicates()
    data['habit'] = data['habit'].fillna('fast')
    data.loc[data['habit'].str.contains(r'\d', na=False), 'habit'] = 'fast'
    return data

def get_cleaned_dataset2():
    data = pd.read_csv("Datasets/dataset2.csv")
    data = data.drop_duplicates()
    print("Columns in dataset2:", data.columns)
    return data
