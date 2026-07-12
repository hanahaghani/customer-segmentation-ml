import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data=pd.read_csv('segmentation data.csv')

df=data.drop('ID',axis=1)

df.info()
print(df.head())
print(df.describe())

scaler=StandardScaler()
x_scaled=scaler.fit_transform(df)
df_scaled=pd.DataFrame(x_scaled,columns=df.columns)
print(df_scaled.describe())

inertia=[]

for k in range(1,21):
    kmeans= KMeans(n_clusters=k,random_state=42)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)
    

plt.plot (range(1,21),inertia,marker='o')
plt.xlabel('number of clusters(k)')
plt.ylabel('inertia')
plt.title('elbow method')
plt.savefig('img/lbow.png')
plt.show()
