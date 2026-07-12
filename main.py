import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('segmentation data.csv')

df=data.drop('ID',axis=1)

df.info()
print(df.head())
print(df.describe())

scaler=StandardScaler()
x_scaled=scaler.fit_transform(df)
df_scaled=pd.DataFrame(x_scaled,columns=df.columns)
print(df_scaled.describe())