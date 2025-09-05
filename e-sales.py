from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

# load

df = pd.read_csv('amazon_sale_report.csv')
print("Dataset loaded successfully!")

# Data Cleaning 

# Check for missing vals and dupes
print("\nMissing values before cleaning:")
print(df.isnull().sum())
df.drop_duplicates(inplace=True)

# remove rows with missing 'Amount' 
df.dropna(subset=['Amount'], inplace=True)

# convert 'Date' column to a proper datetime object.
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
df.dropna(subset=['Amount', 'Qty'], inplace=True) # Drop rows where conversion failed

print("\nDataset cleaned successfully!")
print("DataFrame shape after cleaning:", df.shape)


daily_sales = df.resample('D')['Amount'].sum().to_frame()

plt.figure(figsize=(15, 7))
daily_sales.plot(ax=plt.gca(), legend=False)
plt.title('Overall Total Sales Trend (After Cleaning)')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount (INR)')
plt.show()

# 4.2 Average Sales by Fulfilment Method
fulfilment_sales = df.groupby('Fulfilment')['Amount'].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=fulfilment_sales.index, y=fulfilment_sales.values, palette='coolwarm')
plt.title('Average Sales by Fulfilment Method')
plt.xlabel('Fulfilment Method')
plt.ylabel('Average Sales Amount (INR)')
plt.show()

# build linear reg model

forecast_df = daily_sales.copy()
forecast_df.reset_index(inplace=True)

forecast_df['Day_Index'] = (forecast_df['Date'] - forecast_df['Date'].min()).dt.days

X = forecast_df[['Day_Index']]
y = forecast_df['Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train linear reg model
model = LinearRegression()
model.fit(X_train, y_train)

# predict on the test data
y_pred = model.predict(X_test)
print(f"\nModel R^2 score: {model.score(X_test, y_test):.2f}")

# visualize

last_date = forecast_df['Date'].max()
future_dates = pd.date_range(start=last_date, periods=30, freq='D')
future_day_indices = np.array([(d - forecast_df['Date'].min()).days for d in future_dates]).reshape(-1, 1)

future_predictions = model.predict(future_day_indices)

predicted_series = pd.Series(future_predictions, index=future_dates)

plt.figure(figsize=(15, 8))
plt.plot(daily_sales.index, daily_sales['Amount'], label='Historical Sales')
plt.plot(predicted_series.index, predicted_series.values, label='30-Day Forecast', linestyle='--')
plt.title('Historical Sales and Simple 30-Day Forecast')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount (INR)')
plt.legend()
plt.show()

print("\nProject complete. This script demonstrates data cleaning, EDA, and basic forecasting.")
