import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np 
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

np.random.seed(4711)
a = np.random.multivariate_normal([10,0],[[3,1],[1,4]], size = [100,])
b = np.random.multivariate_normal([0,20],[[3,1],[1,4]], size = [50,])
X = np.concatenate([a,b])
plt.scatter(X[:,0], X[:,1])
plt.show()

Z = linkage(X,'ward')
# comprobar el coeficiente de cophenet
# compara las distancias 2 a 2 de nuestras muestras con las que se llevaron a cabo con el clustering jerarquico
# cuanto mas cercano a 1, mejor sera el clustering, ya que preservara las distancias originales que hubiera entre las 
# distintas observaciones
c, coph_dis = cophenet(Z, pdist(X))
c # en c puedo ver que tan precisa ha sido (conservacion de la distancia originales con los nuevos clusters que se han originado)

plt.figure(figsize = (25,10))
plt.title('Dendrograma del clustering jerargico truncado')
plt.xlabel('Indices de las muestras')
plt.ylabel('Distancia')
d = dendrogram(Z, leaf_font_size=20, color_threshold=0.7*180,
             truncate_mode='lastp', p=10,show_leaf_counts=False, show_contracted= True)
plt.show()

# Metodo del codo

last = Z[-10:,2]
last_revertido = last[::-1]
idx = np.arange(1,len(last)+1)
plt.plot(idx, last_revertido)

# Recuperar los clusters y sus elementos
###### forma a partir de la distancia
max_d = 20
clusters = fcluster(Z, max_d, criterion='distance')
###### forma a partir de num de clusters
k = 3
clusters = fcluster(Z, k, criterion='maxclust')

plt.figure(figsize=(10,8))
plt.scatter(X[:,0],X[:,1], c = clusters, cmap='prism')
plt.show()

