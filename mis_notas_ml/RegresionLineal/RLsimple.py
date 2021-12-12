# Regresion linear simple
# Utilizacion del paquete statsmodels
 
import pandas as pd
import statsmodels.formula.api as snf
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/ads/Advertising.csv')
data.head()

lm = snf.ols(formula = 'Sales~TV', data = data).fit()  #linear model
# Sales~TV = variable predictora en funcion de TV
lm.params # parametros para crear la recta
# Entonces el modelo linear predictivo seria: sales = intercept + TV * (el valor que me devuelve con el nombre TV)
lm.pvalues # los resultados que me dan son muy chicos -> entonces puedo garantizar que los parametros no son cero
lm.rsquared # esto es el R2
lm.rsquared_adj # este suele dar mejor valor que el lm.rsquared, pero la dif es muy chica
lm.summary() #para ver todos los parametros

sales_pred = lm.predict(pd.DataFrame(data['TV']))

data.plot(kind = 'scatter', x = 'TV', y = 'Sales')
plt.plot(pd.DataFrame(data['TV']), sales_pred, c = 'red', linewidth = 2)

# Calcular el RSE (Error estandar de los residuos)
data['Sales_pred'] = 7.032594 + 0.047537 * data['TV']
data['SSD'] = (data['Sales'] - data['Sales_pred']) ** 2
SSD = sum(data['SSD'])
RSE = np.sqrt(SSD/(len(data)-2))
sales_m = np.mean(data['Sales'])
error = RSE/sales_m # 23% del modelo que no queda correctamente explicada
plt.hist(data['Sales'] - data['Sales_pred']) # para ver como se distribuyen los errores respecto del modelo 
# para mejorar el modelo se puede hacer un modelo de regresion lineal MULTIPLE agregando las otras columnas del dataset
