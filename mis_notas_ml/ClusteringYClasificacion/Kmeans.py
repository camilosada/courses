import numpy as np
from scipy.cluster.vq import vq
from scipy.cluster.vq import kmeans

data = np.random.random(90).reshape(30,3)
# Defino dos centroides al azar (es decir que voy a hacer todo para dos clusters)
c1 = np.random.choice(range(len(data)))
c2 = np.random.choice(range(len(data)))
clust_centers = np.vstack([data[c1],data[c2]]) # candidatos a ser los centroides

vq(data, clust_centers) # info del clust al que pertenecen y a que dist se encuentran del centroide correspondiente

kmeans(data, clust_centers) # el ultimo numero es la suma de los errores al cuadrado normalizado

kmeans(data, 2) # aca en vez de pasarle los centroides le paso la cant de clust que quiero

