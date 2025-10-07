import init
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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




# Calculate Simple Regression
