# Dataset Cleaning Functions for Project Analysis

import pandas as pd


def get_cleaned_dataset1():
    data = pd.read_csv("Datasets/dataset1.csv")

    # delete duplicate rows
    data = data.drop_duplicates()

    # replace NaN values in habit column to fast
    data['habit'] = data['habit'].fillna('fast')

    # replace all values with digits in habit column to fast
    data.loc[data['habit'].str.contains(r'\d', na=False), 'habit'] = 'fast'
    return data


def get_cleaned_dataset2():
    data = pd.read_csv("Datasets/dataset2.csv")

    # delete duplicate rows
    data = data.drop_duplicates()
    return data
