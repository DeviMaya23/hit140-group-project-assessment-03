# spring_features.py
import pandas as pd

def add_new_features(df):
    # make sure there are no missing values first
    df = df.fillna(0)

    # new feature 1: rat activity per arrival
    df['rat_activity_ratio'] = df['rat_minutes'] / (df['rat_arrival_number'] + 1)

    # new feature 2: adjusted food index
    df['adjusted_food_index'] = df['food_availability'] / (df['bat_landing_number'] + 1)

    return df
