from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

# Load the Dataset

df = pd.read_csv('Amazon Sale Report.csv')
print("Dataset loaded successfully!")
print("Error: 'Amazon Sale Report.csv' not found. Please ensure the file is in the same directory.")




