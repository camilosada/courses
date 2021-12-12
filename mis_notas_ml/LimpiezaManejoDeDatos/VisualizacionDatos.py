import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
FullPath = "C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt"
data = pd.read_csv(FullPath)
data.head()

#% matplotlib inline # para que queden alineados los graficos (en Jupiter)
##### savefig('path donde lo quieras guardar.jpeg')

## Scatter plot (nube de puntos)
data.plot(kind='scatter', x='Day Mins', y='Day Charge')
#### Subplots
figure, axs = plt.subplots(2,2,sharex=True,sharey=True)
data.plot(kind='scatter', x='Day Mins', y='Day Charge', ax= axs[0][0])

## Histograma de frecuencias
plt.hist(data['Day Calls'], bins= 30) # bins: numero de divisiones del histograma
                                      # tambien puedo hacer bins=[0,60,90,100,150]
# Regla de Sturges: cantidad de divisiones q hay que hacer en el hist= 1+ log2(M), M: num de muestras
k=int(np.ceil(1+np.log2(3333)))
plt.hist(data['Day Calls'], bins= k)
plt.xlabel('Numero de llamadas al dia')
plt.ylabel('Frecuencia')
plt.title('Histograma de numero de llamadas al dia')

## Boxplot, diagrama de caja y bigotes
#### la caja sirve para visualizar donde se encuentran los valores centrales, donde se concentra la info
#### El borde inferior de la caja define el percentil 25% y el superior el 75%
##### AyudaMemoria: percentil 25% es el primer 25% de los datos luego de ser ordenados de menor a mayor
#### La raya amarilla es el valor que esta en el 50% (es decir la media)
plt.boxplot(data['Day Calls'])
plt.ylabel('Numero de llamadas diarias')
#### Rango intercuartilico(IQR): tamano de la caja
IQR=data['Day Calls'].quantile(0.75)-data['Day Calls'].quantile(0.25)
#### Bigote inferior
data['Day Calls'].quantile(0.25)-1.5*IQR
#### Bigote superior
data['Day Calls'].quantile(0.75)+1.5*IQR
#### Por fuera de los bigotes, estan los outliers(valores fuera de lugar), representados con circulos
