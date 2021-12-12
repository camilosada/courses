import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import KMeans
from sklearn import datasets

df = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/wine/winequality-red.csv', sep=';')
plt.hist(df['quality'])
df.groupby('quality').mean()

# Normalizacion de los datos
df_norm = (df - df.min())/(df.max()-df.min())

# Clustering jerarquico con scikit-learn
clus = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(df_norm)

md_H = pd.Series(clus.labels_) # etiquetas para saber a que cluster pertenecen
plt.hist(md_H)
plt.title('Histograma de los clusters')
plt.xlabel('Clusters')
plt.ylabel('Numero de vinos del cluster')

clus.children_

Z=linkage(df_norm, 'ward')
plt.figure(figsize=(25,10))
plt.title('Dendrograma de los vinos')
plt.xlabel('ID del vino')
plt.ylabel('Distancia')
dendrogram(Z, leaf_font_size=8.)
plt.show()

# Clustering con K-means
model = KMeans(n_clusters=6)
model.fit(df_norm)
model.labels_
md_K = pd.Series(model.labels_)
plt.hist(md_K)
model.cluster_centers_

df_norm['clust_h'] = md_H
df_norm['clust_k'] = md_K
df_norm.head()

model.inertia_ # factor para ver la eficiencia del modelo(valor de la suma de los cuadrados internos, no esta normalizado)

# Interpretacion final
df_norm.groupby('clust_k').mean() # agrupo por cluster para ver las caracteristicas promedio de cada grupo(cluster)
