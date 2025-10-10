# spring_analysis.py
# Author: Neha
# Investigation B (Spring)

import init
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn
import math
from spring_features import add_new_features

# Load cleaned data
df = init.get_cleaned_dataset2()

# Keep only Spring (month >= 3)
df = df[df['month'] >= 3]

# Add new feature columns
df = add_new_features(df)

# Drop text columns not needed
df = df.drop(columns=['time', 'month'])

print("Spring data shape:", df.shape)
print(df.head())

# ======== Simple scatterplots ========
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', data=df)
plt.title('Spring: Bat vs Rat Arrivals')
plt.savefig('spring_scatter.png', dpi=150)
plt.show()

# ======== Correlation heatmap ========
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

mae = sklearn.metrics.mean_absolute_error(y, pred)
mse = sklearn.metrics.mean_squared_error(y, pred)
rmse = math.sqrt(mse)
print("\nMAE:", mae)
print("RMSE:", rmse)
