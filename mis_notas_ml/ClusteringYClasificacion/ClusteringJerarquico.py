import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/movies/movies.csv', sep=';')
data.head()
movies = data.columns.values.tolist()[1:]
# Antes de calcular las distancias hay que normalizar para que se tengan en cuenta todos los valores 
# y no solo aquellos que son numericamente mas grandes
# No hace falta normalizar los valores ya que se suponen que estan puntuadas dentro de un mismo rango
dd1 = pd.DataFrame(distance_matrix(data[movies], data[movies], p = 1)) # Distancia Manhattan
dd2 = pd.DataFrame(distance_matrix(data[movies], data[movies], p = 2)) # Distancia Euclidea
dd3 = pd.DataFrame(distance_matrix(data[movies], data[movies], p = 10)) # Distancia Minkowski

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(xs = data['star_wars'], ys = data['lord_of_the_rings'], zs = data['harry_potter'])

Z = linkage(data[movies], 'ward')
# En Z: [num1, num2, distancia, cantidad de elementos unidos]
plt.figure(figsize = (25,10))
plt.title('Dendrograma jerarquico para el clustering')
plt.xlabel('ID de los usuarios de Netflix')
plt.ylabel('Distancia')
d=dendrogram(Z, leaf_font_size=15)
plt.show()


Z = linkage(data[movies], 'average')
plt.figure(figsize = (25,10))
plt.title('Dendrograma jerarquico para el clustering')
plt.xlabel('ID de los usuarios de Netflix')
plt.ylabel('Distancia')
dendrogram(Z, leaf_font_size=15)
plt.show()

Z = linkage(data[movies], 'single')
plt.figure(figsize = (25,10))
plt.title('Dendrograma jerarquico para el clustering')
plt.xlabel('ID de los usuarios de Netflix')
plt.ylabel('Distancia')
dendrogram(Z, leaf_font_size=15)
plt.show()