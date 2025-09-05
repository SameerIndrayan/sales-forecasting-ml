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


