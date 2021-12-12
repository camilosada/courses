# El problema de los outliers
# Transformar las variables en relaciones no lineales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # no funciona muy bien
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


data_auto = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/auto/auto-mpg.csv')
data_auto.head()
X= pd.DataFrame(data_auto['displacement'].fillna(data_auto['displacement'].mean()))
# Convierto X en DataFrame por la funcion fit() que me lo pide, otra forma es pasarlo a array X[:, np.newaxis]
Y = data_auto['mpg'].fillna(data_auto['mpg'].mean())

lm = LinearRegression()
lm.fit(X, Y)
lm.score(X,Y)

# Encontrar outliers (acordarme que puedo usar el modelo de caja y bigote para encontrar outliers)
data_auto[(data_auto['displacement']>250)&(data_auto['mpg']>35)] # para encontrar el outlier
data_auto[(data_auto['displacement']>300)&(data_auto['mpg']>20)] # para encontrar el outlier
# Saco los outliers del df
data_auto_clean = data_auto.drop([395,258,305,372])
# Creo el modelo
X2= pd.DataFrame(data_auto_clean['displacement'].fillna(data_auto_clean['displacement'].mean()))
Y2 = data_auto_clean['mpg'].fillna(data_auto_clean['mpg'].mean())
lm = LinearRegression()
lm.fit(X2, Y2)
lm.score(X2,Y2)
