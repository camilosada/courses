# Transformar las variables en relaciones no lineales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # no funciona muy bien
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


data_auto = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/auto/auto-mpg.csv')
data_auto.head()
data_auto.shape

data_auto['mpg']=data_auto['mpg'].dropna()
data_auto['horsepower']= data_auto['horsepower'].dropna()
plt.plot(data_auto['horsepower'], data_auto['mpg'], 'ro')

plt.xlabel('Horse Power')
plt.ylabel('Consumo millas por galeon')
plt.title('CV vs MPG')

## Modelo de regresion linear

X= pd.DataFrame(data_auto['horsepower'].fillna(data_auto['horsepower'].mean()))
# Convierto X en DataFrame por la funcion fit() que me lo pide, otra forma es pasarlo a array X[:, np.newaxis]
Y = data_auto['mpg'].fillna(data_auto['mpg'].mean())
lm = LinearRegression()
lm.fit(X,Y)
plt.plot(X,Y,'ro')
plt.plot(X,lm.predict(X), c = 'blue')
lm.score(X,Y)

SSD = np.sum((Y-lm.predict(X)) **2) 
RSE = np.sqrt(SSD/(len(X)-1))
Error = RSE/ np.mean(Y)

## Modelo de regresion cuadratico
# mpg = a + b * horsepower^2
X2=X**2
lm2 = LinearRegression()
lm2.fit(X2,Y)
lm2.score(X2, Y)

SSD2 = np.sum((Y-lm2.predict(X2)) **2) 
RSE2 = np.sqrt(SSD2/(len(X2)-1))
Error2 = RSE2/ np.mean(Y)


plt.plot(X,Y,'ro')
plt.plot(X,lm.predict(X2), 'bo')
## Modelo de regresion linear y cuadratico
# mpg = a + b*horspower + c* horsepower^2

poly = PolynomialFeatures(degree = 2)
X3 = poly.fit_transform(X)
lm3 = linear_model.LinearRegression()
lm3.fit(X3,Y)
lm3.score(X3,Y)

plt.plot(X,Y,'ro')
plt.plot(X,lm.predict(X3), 'bo')