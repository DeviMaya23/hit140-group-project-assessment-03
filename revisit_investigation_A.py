import init
import matplotlib.pyplot as plt
import sklearn
import math


df = init.get_cleaned_dataset2()

plt.scatter(df['rat_arrival_number'], df['bat_landing_number'])
plt.xlabel('Rats arrived')
plt.xticks(range(min(df['rat_arrival_number']), max(df['rat_arrival_number']), 2))
plt.ylabel('Bats landed')
plt.title('The amount of rats & bats that arrived in 30 minutes interval')
plt.show()


# Calculate correlation coefficient
r = df['rat_arrival_number'].corr(df['bat_landing_number'])
print("Correlation coefficient: ", r)

# Calculate single linear regression
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[['rat_arrival_number']], df['bat_landing_number'], test_size=0.4, random_state=0)

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