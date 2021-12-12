# Propagacion de la afinidad 

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

centers = [[1,1], [-1,-1], [1,-1]]
X, labels = make_blobs(n_samples=300, centers=centers,cluster_std=0.5,random_state=0)
# labels: cluster al que realmente pertenecen
plt.scatter(X[:,0],X[:,1], c=labels, s=5,cmap='autumn')

af = AffinityPropagation(preference = -50).fit(X)
cluster_center_ids = af.cluster_centers_indices_ # se la posision de los centros en el array
cluster_labels = af.labels_ # se a que clust pertenecen segun la prediccion
n_clust = len(cluster_center_ids) # cantidad de clust que me genero la propagacion

metrics.homogeneity_score(labels, cluster_labels) #homogeneidad (cuantas fueron correctamente clasificadas)
metrics.completeness_score(labels,cluster_labels) #completitud (tasa de verdaderos y falsos positivos)
metrics.v_measure_score(labels,cluster_labels) # v mesured
metrics.adjusted_rand_score(labels, cluster_labels) # r2 ajustado
metrics.adjusted_mutual_info_score(labels,cluster_labels) # info mutua ajustada
metrics.silhouette_score(X,labels, metric = 'sqeuclidean') # coeficiente de la silueta

plt.figure(figsize=(16,9))
plt.clf()

colors = cycle('bgrcmyk')
for k, col in zip(range(n_clust), colors):
    clas_members = (cluster_labels==k)
    clus_center = X[cluster_center_ids[k]]
    plt.plot(X[clas_members,0], X[clas_members,1],col + '.')
    plt.plot(clus_center[0],clus_center[1],'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    for x in X[clas_members]:
        plt.plot([clus_center[0],x[0]],[clus_center[1], x[1]], col)

plt.title('numero estimado de clusters %d' %n_clust)
plt.show()