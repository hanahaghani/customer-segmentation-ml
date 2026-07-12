import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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

kmeans_ml=KMeans(n_clusters=8,random_state=42)
cluster_km=kmeans_ml.fit_predict(x_scaled)

df['cluster-km']=cluster_km

print(df['cluster-km'].value_counts())

pca_km=PCA(n_components=3)
x_pca_km=pca_km.fit_transform(x_scaled)

df_pca_km=pd.DataFrame(x_pca_km,columns=['pc1','pc2','pc3'])
df_pca_km['cluster-km']=cluster_km

fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111,projection='3d')

ax.scatter(
    x_pca_km[:,0],
    x_pca_km[:,1],
    x_pca_km[:,2],
    c=df['cluster-km'],
    cmap='viridis',
    s=40
)
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')
plt.title('customer segmentation clusters (pca visualization)') 
plt.savefig('img/pca-km-segmantation.png')
plt.show()


score=silhouette_score(x_scaled,cluster_km)
print(f'silhouette score:{score:.3f}')