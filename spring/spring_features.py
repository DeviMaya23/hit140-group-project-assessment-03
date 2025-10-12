# spring_features.py
# Add new computed features for Spring analysis

import pandas as pd

def add_new_features(df):
    df = df.fillna(0)
    df['rat_activity_ratio'] = df['rat_minutes'] / (df['rat_arrival_number'] + 1)
    df['adjusted_food_index'] = df['food_availability'] / (df['bat_landing_number'] + 1)
    return df
