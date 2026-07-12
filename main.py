#libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage,dendrogram


#load data
data=pd.read_csv('segmentation data.csv')

#EDA

df=data.drop('ID',axis=1)

df.info()
print(df.head())
print(df.describe())

scaler=StandardScaler()
x_scaled=scaler.fit_transform(df)
df_scaled=pd.DataFrame(x_scaled,columns=df.columns)
print(df_scaled.describe())

#kmeans 

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
plt.title('customer segmentation clusters (pca visualization kmeans)') 
plt.savefig('img/pca-km-segmantation.png')
plt.show()

score=silhouette_score(x_scaled,cluster_km)
print(f'silhouette score:{score:.3f}')

#hierarchical

linked=linkage(x_scaled,method='ward')
plt.figure(figsize=(12,6))
dendrogram(linked)
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('distance')
plt.savefig('img/hierarchical-plot.png')
plt.show()

hierarchical_ml=AgglomerativeClustering(n_clusters=5)
cluster_hil=hierarchical_ml.fit_predict(x_scaled)

df['cluster-hil']=cluster_hil
print(df['cluster-hil'].value_counts())

pca_hil=PCA(n_components=3)
x_pca_hil=pca_hil.fit_transform(x_scaled)

df_pca_hil=pd.DataFrame(x_pca_hil,columns=['pc1','pc2','pc3'])
df_pca_hil['cluster-hil']=cluster_hil

fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111,projection='3d')

ax.scatter(
    x_pca_km[:,0],
    x_pca_km[:,1],
    x_pca_km[:,2],
    c=df['cluster-hil'],
    cmap='viridis',
    s=40
)
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')
plt.title('customer segmentation clusters (pca visualization hierarchical)') 
plt.savefig('img/pca-hil-segmantation.png')
plt.show()

score=silhouette_score(x_scaled,cluster_hil)
print(f'silhouette score:{score:.3f}')
