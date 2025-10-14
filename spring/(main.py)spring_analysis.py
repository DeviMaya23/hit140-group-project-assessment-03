# main.py
# Author: Asha Devi
# Investigation B (Spring)

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn.metrics
import math
from spring_features import add_new_features
import data_init as init

print("=== Starting Spring Investigation ===")

# ======== Load Cleaned Data ========
df = init.get_cleaned_dataset2()
print("Columns in dataset:", df.columns)

# ======== Filter for Spring Months (March to May) ========
if 'month' in df.columns:
    df = df[df['month'] >= 3]
print("Rows before feature creation:", len(df))

# ======== Add New Feature Columns ========
df = add_new_features(df)

# Drop text columns if they exist
for col in ['time', 'month']:
    if col in df.columns:
        df = df.drop(columns=[col])

print("Spring data shape:", df.shape)
print(df.head(), "\n")

# ======== Scatterplot ========
try:
    print("Creating scatter plot...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', data=df)
    plt.title('Spring: Bat vs Rat Arrivals')
    plt.xlabel('Rat Arrival Number')
    plt.ylabel('Bat Landing Number')
    plt.tight_layout()
    plt.savefig('spring_scatter.png', dpi=150)
    plt.close()
    print(" Scatter plot saved as spring_scatter.png")
except Exception as e:
    print(" Scatter plot error:", e)

# ======== Correlation Heatmap ========
try:
    print("Creating heatmap...")
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Spring Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('spring_heatmap.png', dpi=150)
    plt.close()
    print(" Heatmap saved as spring_heatmap.png")
except Exception as e:
    print(" Heatmap error:", e)

# ======== Linear Regression ========
try:
    print("Running linear regression...")
    x = sm.add_constant(df[['rat_arrival_number']])
    y = df['bat_landing_number']
    model = sm.OLS(y, x).fit()
    pred = model.predict(x)

    print("\nLinear Regression Summary (Spring):")
    print(model.summary())

    mae = sklearn.metrics.mean_absolute_error(y, pred)
    mse = sklearn.metrics.mean_squared_error(y, pred)
    rmse = math.sqrt(mse)
    r2 = sklearn.metrics.r2_score(y, pred)
    nrmse = rmse / (max(y) - min(y))

    print("\nModel Performance:")
    print(f"MAE   = {mae:.4f}")
    print(f"RMSE  = {rmse:.4f}")
    print(f"R²    = {r2:.4f}")
    print(f"NRMSE = {nrmse:.4f}")

    print("Creating regression plot...")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y, y=pred)
    sns.lineplot(x=y, y=y, color='red')
    plt.title('Spring: Predicted vs Actual Bat Landings')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.savefig('spring_regression.png', dpi=150)
    plt.close()
    print(" Regression plot saved as spring_regression.png")

except Exception as e:
    print(" Regression error:", e)

print("\n=== Script Completed ===")
# ======== Multiple Linear Regression ========
x_multi = sm.add_constant(df[['rat_arrival_number', 'rat_minutes', 'food_availability']])
y = df['bat_landing_number']

multi_model = sm.OLS(y, x_multi).fit()
multi_pred = multi_model.predict(x_multi)

print("\nMultiple Linear Regression Summary (Spring):")
print(multi_model.summary())

# Compare R² between simple and multiple regression
r2_multi = sklearn.metrics.r2_score(y, multi_pred)
print(f"\nModel Comparison:")
print(f"Single Regression R²: {r2:.4f}")
print(f"Multiple Regression R²: {r2_multi:.4f}")
