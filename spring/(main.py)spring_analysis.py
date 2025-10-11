# spring_analysis.py
# Author: Asha Devi
# Investigation B (Spring)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn
import math
from spring_features import add_new_features
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# ======== Load and Prepare Data ========
df = pd.read_csv("Datasets/dataset2.csv")

# Keep only Spring data (month >= 3)
df = df[df['month'] >= 3]

# Add new feature columns from spring_features.py
df = add_new_features(df)

# Drop text/time columns if not needed
if 'time' in df.columns:
    df = df.drop(columns=['time'])
if 'month' in df.columns:
    df = df.drop(columns=['month'])

print("Spring data shape:", df.shape)
print(df.head())

# ======== Simple Scatterplot ========
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', data=df)
plt.title('Spring: Bat vs Rat Arrivals')
plt.savefig('spring_scatter.png', dpi=150)
plt.show()

# ======== Correlation Heatmap ========
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Spring Correlation Heatmap')
plt.tight_layout()
plt.savefig('spring_heatmap.png', dpi=150)
plt.show()

# ======== Simple Linear Regression ========
x = sm.add_constant(df[['rat_arrival_number']])
y = df['bat_landing_number']
model = sm.OLS(y, x).fit()
pred = model.predict(x)

print("\nLinear Regression Summary (Spring):")
print(model.summary())

# ======== Error and Accuracy Metrics ========
mae = sklearn.metrics.mean_absolute_error(y, pred)
mse = sklearn.metrics.mean_squared_error(y, pred)
rmse = math.sqrt(mse)

# Calculate R² and NRMSE
r2 = r2_score(y, pred)
nrmse = rmse / (y.max() - y.min())

print("\nMAE:", mae)
print("RMSE:", rmse)
print("R² Score:", round(r2, 4))
print("NRMSE:", round(nrmse, 4))

# ======== Regression Plot ========
plt.figure(figsize=(8, 5))
sns.regplot(x='rat_arrival_number',
            y='bat_landing_number',
            data=df,
            ci=None,
            color='purple')
plt.title('Spring Regression: Bat vs Rat Arrivals')
plt.savefig('spring_regression.png', dpi=150)
plt.show()
