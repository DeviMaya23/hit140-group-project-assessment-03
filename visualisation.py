import init
import matplotlib.pyplot as plt
import seaborn as sns


df = init.get_cleaned_dataset2()

# Scatterplot for all variables
fig, axs = plt.subplots(2, 2)

axs[0][0].scatter(df['rat_arrival_number'], df['bat_landing_number'])
axs[0][0].set_xlabel("rat_arrival_number")

axs[0][1].scatter(df['rat_minutes'], df['bat_landing_number'])
axs[0][1].set_xlabel("rat_minutes")

axs[1][0].scatter(df['hours_after_sunset'], df['bat_landing_number'])
axs[1][0].set_xlabel("hours_after_sunset")

axs[1][1].scatter(df['food_availability'], df['bat_landing_number'])
axs[1][1].set_xlabel("food_availability")

plt.show()
plt.close()


# Boxplot for all variables
data = df['rat_arrival_number']
sns.boxplot(y=data)
plt.title("Outlier check for rat_arrival_number")
plt.savefig("result/rat_arrival_number_boxplot.png")
plt.show()
plt.close()

data = df['rat_minutes']
sns.boxplot(y=data)
plt.title("Outlier check for rat_minutes")
plt.savefig("result/rat_minutes_boxplot.png")
plt.show()
plt.close()

data = df['hours_after_sunset']
sns.boxplot(y=data)
plt.title("Outlier check for hours_after_sunset")
plt.savefig("result/hours_after_sunset_boxplot.png")
plt.show()
plt.close()

data = df['food_availability']
sns.boxplot(y=data)
plt.title("Outlier check for food_availability")
plt.savefig("result/food_availability_boxplot.png")
plt.show()
plt.close()

data = df['bat_landing_number']
sns.boxplot(y=data)
plt.title("Outlier check for bat_landing_number")
plt.savefig("result/bat_landing_number_boxplot.png")
plt.show()
plt.close()

