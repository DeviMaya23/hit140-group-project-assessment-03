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

# ======== Load Cleaned Data ========
df = init.get_cleaned_dataset2()

print(" Columns in dataset:", df.columns)

# ======== (TEMP) Remove Month Filter for Debugging ========
# If your CSV doesn't have 'month', this filter removes all rows — disable for now
# df = df[df['month'] >= 3]
print("Rows before feature creation:", len(df))

# ======== Add New Feature Columns ========
df = add_new_features(df)

# Drop text columns if they exist
for col in ['time', 'month']:
    if col in df.columns:
        df = df.drop(columns=[col])

print(" Spring data shape:", df.shape)
print(df.head(), "\n")

# ======== Scatterplot ========
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', data=df)
plt.title('Spring: Bat vs Rat Arrivals')
plt.xlabel('Rat Arrival Number')
plt.ylabel('Bat Landing Number')
plt.tight_layout()
plt.savefig('spring_scatter.png', dpi=150)
plt.show()

# ======== Correlation Heatmap ========
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Spring Correlation Heatmap')
plt.tight_layout()
plt.savefig('spring_heatmap.png', dpi=150)
plt.show()

# ======== Linear Regression ========
x = sm.add_constant(df[['rat_arrival_number']])
y = df['bat_landing_number']
model = sm.OLS(y, x).fit()
pred = model.predict(x)

print("\n Linear Regression Summary (Spring):")
print(model.summary())

# ======== Model Performance Metrics ========
mae = sklearn.metrics.mean_absolute_error(y, pred)
mse = sklearn.metrics.mean_squared_error(y, pred)
rmse = math.sqrt(mse)
r2 = sklearn.metrics.r2_score(y, pred)
nrmse = rmse / (max(y) - min(y))

print("\n Model Performance:")
print(f"MAE   = {mae:.4f}")
print(f"RMSE  = {rmse:.4f}")
print(f"R²    = {r2:.4f}")
print(f"NRMSE = {nrmse:.4f}")

# ======== Regression Plot ========
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y, y=pred)
sns.lineplot(x=y, y=y, color='red')
plt.title('Spring: Predicted vs Actual Bat Landings')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.tight_layout()
plt.savefig('spring_regression.png', dpi=150)
plt.show()

print("\n All graphs saved successfully as:")
print("spring_scatter.png, spring_heatmap.png, spring_regression.png")
