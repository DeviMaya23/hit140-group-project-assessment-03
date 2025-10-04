#"Comparing Dataset Row Counts Before and After Cleaning"
import init
import pandas as pd
# Loading the original (uncleaned) datasets directly from the CSV files
data1_original = pd.read_csv("Datasets/dataset1.csv")
data2_original = pd.read_csv("Datasets/dataset2.csv")
# Loading the cleaned versions of the datasets 
# (using our cleaning functions from init.py)

data1_cleaned = init.get_cleaned_dataset1()
data2_cleaned = init.get_cleaned_dataset2()
# it Shows how many rows were in dataset1 before cleaning 
# and how many are left after cleaning

print("Row count for dataset1 before cleanup:", len(data1_original))
print("Row count for dataset1 after cleanup:", len(data1_cleaned))
print("====================================")
# Doing the same check for dataset2

print("Row count for dataset2 before cleanup:", len(data2_original))
print("Row count for dataset2 after cleanup:", len(data2_cleaned))
