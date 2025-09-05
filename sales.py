import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("fivethirtyeight")
sns.set_style('whitegrid')


df_train = pd.read_csv("train.csv")
df_features = pd.read_csv('features.csv')
df_stores = pd.read_csv('stores.csv')
print("Datasets loaded successfully")

df = pd.merge(df_train, df_stores, on = 'Store', how = 'left')
df = pd.merge(df, df_features, on = ['Store', 'Date'], how = 'left')

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

print("\nFirst 5 rows of merged DataFrame")
print(df.head())

# plots
monthly_sales = df.resample('M')['Weekly_Sales'].sum()
plt.figure(figsize=(15,7))
monthly_sales.plot()
plt.title('Monthly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Monthly Sales')
plt.show()

df['Month'] = df.index.month
monthly_avg_sales = df.groupby("Month")['Weekly_Sales'].mean()
plt.figure(figsize=(10,6))
sns.barplot(x=monthly_avg_sales.index, y = monthly_avg_sales, palette='viridis')
plt.title('Avg Sales by Month')
plt.xlabel('Month')
plt.ylabel('Avg Weekly Sales')
plt.show()

# check holiday szn
holiday_sales = df.groupby('IsHoliday')['Weekly_Sales'].mean()
plt.figure(figsize=(8, 5))
sns.barplot(x=holiday_sales.index, y=holiday_sales.values)
plt.title('Average Weekly Sales: Holiday vs. Non-Holiday')
plt.xticks(ticks=[0, 1], labels=['Non-Holiday', 'Holiday'])
plt.xlabel('Is Holiday?')
plt.ylabel('Average Weekly Sales')
plt.show()



