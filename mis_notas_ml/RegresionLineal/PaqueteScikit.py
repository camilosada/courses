# Paquete scikit learn para regresion linear y la seleccion de rasgos

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression # tiene implementada la funcion de regresion directamente

data = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/ads/Advertising.csv')
data.head()

feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
Y = data['Sales']
estimator = SVR(kernel = 'linear')
selector  = RFE(estimator, 2, step = 1)
selector = selector.fit(X,Y)
selector.support_

selector.ranking_ # las variables selecionadas aparecen con 1, el resto las ordena con orden creciente
                  # dependiendo de su significatividad con respecto al modelo

# Hago la regresion linear
X_pred = X[['TV','Radio']]
lm = LinearRegression()
lm.fit(X_pred, Y)
# aca le pido los parametros 
lm.intercept_
lm.coef_
# con esto obtengo el R2 ajustado
lm.score(X_pred, Y)