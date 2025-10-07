import init
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import math


df = init.get_cleaned_dataset2()

# Feature Engineering
# Add season column
df['season'] = np.where(df['month'] < 3, 0, 1)
# Add minutes_after_sunset
df['minutes_after_sunset'] = df['hours_after_sunset'] * 60

# Drop month column, non numerical data
df = df.drop(columns=['month'])

df_season0 = df[df['season'] == 0]
# df_season1 = df[df['season'] == 1]

# Drop season column, used only to divide data
df_season0 = df.drop(columns=['season'])  # winter


# Scatterplot for all variables
fig, axs = plt.subplots(2, 2)

axs[0][0].scatter(df_season0['rat_arrival_number'], df_season0['bat_landing_number'])
axs[0][0].set_title('rat_arrival_number')


axs[0][1].scatter(df_season0['rat_minutes'], df_season0['bat_landing_number'])
axs[0][1].set_title('rat_minutes')

axs[1][0].scatter(df_season0['minutes_after_sunset'], df_season0['bat_landing_number'])
axs[1][0].set_title('minutes_after_sunset')

axs[1][1].scatter(df_season0['food_availability'], df_season0['bat_landing_number'])
axs[1][1].set_title('food_availability')

plt.suptitle('Winter Dataset')
plt.show()
plt.close()



# Calculate correlation coefficient
r = df_season0['rat_arrival_number'].corr(df_season0['bat_landing_number'])
print("Correlation coefficient rat_arrival_number: ", r)

# Calculate single linear regression
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df_season0[['rat_arrival_number']], df_season0['bat_landing_number'], test_size=0.4, random_state=0)

model = sklearn.linear_model.LinearRegression()
model.fit(X_train, y_train)

coeff = model.coef_[0]
intercept = model.intercept_

print("X1 : ", coeff)
print("X0 : ", intercept)
print("-------")
print("Result:")
print("y = ", intercept, " - x *", coeff * -1)


y_pred = model.predict(X_test)
mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)


y_max = y_test.max()
y_min = y_test.min()

normalised_rmse = rmse/(y_max-y_min)

print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("Normalised RMSE: ", normalised_rmse)

print("------------")

# Calculate multiple linear regression

x = df_season0[['rat_arrival_number', 'rat_minutes', 'minutes_after_sunset', 'food_availability']]
y = df_season0['bat_landing_number']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.4, random_state=0)

model = sklearn.linear_model.LinearRegression()
model.fit(X_train, y_train)

# coeff = model.coef_[0]
intercept = model.intercept_

print("coefficient : ", model.coef_)
print("intercept : ", intercept)


y_pred = model.predict(X_test)
mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)


y_max = y_test.max()
y_min = y_test.min()

normalised_rmse = rmse/(y_max-y_min)
r_2 = sklearn.metrics.r2_score(y_test, y_pred)

print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("Normalised RMSE: ", normalised_rmse)
print("R^2 :", r_2)