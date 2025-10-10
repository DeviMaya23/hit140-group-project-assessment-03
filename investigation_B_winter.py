import init
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import sklearn
import math


def getWinterData():
    df = init.get_cleaned_dataset2()

    # Feature Engineering
    # Add season column
    df['season'] = np.where(df['month'] < 3, 0, 1)

    # Drop month column, non numerical data
    df = df.drop(columns=['month'])
    df = df.drop(columns=['time'])
    df = df[df['season'] == 0]

    # Non linear transformation
    # Add squared value
    df['hours_after_sunset_squared'] = df['hours_after_sunset'] ** 2
    df['food_availability_squared'] = df['food_availability'] ** 2

    # Drop season column, used only to divide data
    df = df.drop(columns=['season'])  # winter

    return df


df = getWinterData()

# Scatterplot for all variables
fig, axs = plt.subplots(2, 2)

axs[0][0].scatter(df['rat_arrival_number'], df['bat_landing_number'])
axs[0][0].set_title('rat_arrival_number')


axs[0][1].scatter(df['rat_minutes'], df['bat_landing_number'])
axs[0][1].set_title('rat_minutes')

axs[1][0].scatter(df['hours_after_sunset'], df['bat_landing_number'])
axs[1][0].set_title('hours_after_sunset')

axs[1][1].scatter(df['food_availability'], df['bat_landing_number'])
axs[1][1].set_title('food_availability')

plt.suptitle('Winter Dataset')
plt.show()
plt.close()

# Non Linear Transformation
fig, axs = plt.subplots(2, 2)

axs[0][0].scatter(df['food_availability'], df['bat_landing_number'])
axs[0][0].set_title('food_availability')


axs[0][1].scatter(df['food_availability_squared'], df['bat_landing_number'])
axs[0][1].set_title('food_availability_squared')

axs[1][0].scatter(df['hours_after_sunset'], df['bat_landing_number'])
axs[1][0].set_title('hours_after_sunset')

axs[1][1].scatter(df['hours_after_sunset_squared'], df['bat_landing_number'])
axs[1][1].set_title('hours_after_sunset_squared')

plt.suptitle('Squared Dataset')
plt.show()
plt.close()


# Heatmap

# Plot correlation matrix
corr = df.corr()

# Plot the pairwise correlation as heatmap
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False,
    annot=True
)

# customise the labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

plt.show()
plt.close()


# Calculate single linear regression
print("------------")
print("Single Linear Regression")


x = df[['rat_arrival_number']]
y = df['bat_landing_number']

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
pred = model.predict(x)

print(model.params)

intercept = model.params["const"]
coeff = model.params["rat_arrival_number"]
print("-------")
print("Result:")
print("y = ", intercept, " + x *", coeff)


mae = sklearn.metrics.mean_absolute_error(y, pred)
mse = sklearn.metrics.mean_squared_error(y, pred)
rmse = math.sqrt(mse)


y_max = y.max()
y_min = y.min()

normalised_rmse = rmse/(y_max-y_min)
r_2 = sklearn.metrics.r2_score(y, pred)

print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("Normalised RMSE: ", normalised_rmse)
print("R2: ", r_2)

print()

# Calculate multiple linear regression
print("------------")
print("Multiple Linear Regression")

df = getWinterData()

x = df[['rat_arrival_number', 'rat_minutes', 'hours_after_sunset', 'food_availability', 'hours_after_sunset_squared', 'food_availability_squared']]
y = df['bat_landing_number']

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
pred = model.predict(x)

print(model.summary())

intercept = model.params["const"]
coeff_rat_arrival_number = model.params["rat_arrival_number"]
coeff_rat_minutes = model.params["rat_minutes"]
coeff_hours_after_sunset = model.params["hours_after_sunset"]
coeff_food_availability = model.params["food_availability"]
coeff_hours_after_sunset_squared = model.params["hours_after_sunset_squared"]
coeff_food_availability_squared = model.params["food_availability_squared"]

print("-------")
print("Result:")
print("y = ", intercept, " + x1 *", coeff_rat_arrival_number,
" + x2 *", coeff_rat_minutes,
" + x3 *", coeff_hours_after_sunset,
" + x4 *", coeff_hours_after_sunset_squared,
" + x5 *", coeff_food_availability,
" + x6 *", coeff_food_availability_squared,
)

mae = sklearn.metrics.mean_absolute_error(y, pred)
mse = sklearn.metrics.mean_squared_error(y, pred)
rmse = math.sqrt(mse)


y_max = y.max()
y_min = y.min()

normalised_rmse = rmse/(y_max-y_min)
r_2 = sklearn.metrics.r2_score(y, pred)

print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("Normalised RMSE: ", normalised_rmse)
print("R2 ", r_2)
