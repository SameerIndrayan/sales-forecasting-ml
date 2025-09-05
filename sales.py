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

# plot test
plt.figure(figsize=(15,7))
df.groupby(df.index)['Weekly_Sales'].sum().plot
plt.title('Overall Weekly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Weekly Sales')
plt.show()

df['Month'] = df.index.month
monthly_avg_sales = df.groupby("Month")['Weekly_Sales'].mean()
plt.figure(figsize=(10,6))
sns.barplot(x=monthly_avg_sales.index, y = monthly_avg_sales, palette='viridis')
plt.title('Avg Sales by Month')
plt.xlabel('Month')
plt.ylabel('Avg Weekly Sales')
plt.show()




