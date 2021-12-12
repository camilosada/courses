import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score

x1 = np.array([3,1,1,2,1,6,6,6,5,6,7,8,9,8,9,9,8])
x2 = np.array([5,4,5,6,5,8,6,7,6,7,1,2,1,2,3,2,3])
X = np.array(list(zip(x1,x2))).reshape(len(x1),2)

plt.plot()
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Dataset a clasificar')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x1,x2)
plt.show()

max_k = 10
K = range(1,max_k)
ssw = [] # suma de los cuadrados internos
color_palette = [plt.get_cmap('Spectral')(float(i)/max_k) for i in K]
##### No anda color_palette = [plt.cm.spectral(float(i)/max_k)for i in K]
centroid = [sum(X)/len(X) for i in X]
sst = sum(np.min(cdist(X, centroid, 'euclidean'), axis = 1))

for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)

    centers = pd.DataFrame(kmeanModel.cluster_centers_)
    labels = kmeanModel.labels_

    ssw_k = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis = 1))
    ssw.append(ssw_k)

    label_color = [color_palette[i] for i in labels]
    # Fabricaremos una silueta para cada cluster
    # Por seguridad, no hacemos silueta si k = 1 o k = len(X)
    if 1<k<len(X):
        ## crear un subplot de una fila y dos columnas
        fig, (axis1, axis2) = plt.subplots(1,2)
        fig.set_size_inches(20,8)

        # El primer subplot contendra la silueta, que puede tener valores desde -1 a 1
        # En nuestro caso, ya controlamos que los valores estan entre -0.1 y 1
        axis1.set_xlim([-0.1,1.0])
        # El numero de clusters a insertar determinara el tamano de cada barra
        # El coeficiente (n_clusters+1)*10 sera el espacio en blanco que dejaremos
        # entre siluetas individuales de cada cluster para separarlas
        axis2.set_ylim([0,len(X)+(k+1)*10])

        silhouette_avg = silhouette_score(X,labels)
        print('* para k =',k,'el promedio de la silueta es de :'+silhouette_avg)
        sample_silhouette_values = silhouette_samples(X,labels)



