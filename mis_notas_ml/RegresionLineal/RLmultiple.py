# Regresion linear multiple

import pandas as pd
import statsmodels.formula.api as snf
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/ads/Advertising.csv')
data.head()

# Utilizando TV + Newspaper
lm = snf.ols(formula = 'Sales~TV + Newspaper', data = data).fit()  #linear model

lm.params # parametros para crear la recta

lm.pvalues # los resultados que me dan son muy chicos -> entonces puedo garantizar que los parametros no son cero
lm.rsquared_adj 
sales_pred = lm.predict(data[['TV', 'Newspaper']])
SSD = sum((data['Sales'] - sales_pred) ** 2)
SRE = np.sqrt(SSD/(len(data)-3))
sales_m = np.mean(data['Sales'])
error = SRE/sales_m

# Utilizando TV + Radio
### Al agregar el parametro de radio vemos como sube notoriamente el R2, y como disminuye Prob(F-statistic)
lm = snf.ols(formula = 'Sales~TV + Radio', data = data).fit()  #linear model
lm.summary()
sales_pred = lm.predict(data[['TV', 'Radio']])
SSD = sum((data['Sales'] - sales_pred) ** 2)
SRE = np.sqrt(SSD/(len(data)-3))
sales_m = np.mean(data['Sales'])
error = SRE/sales_m

# Utilizando TV + Radio + Newspaper
### en este caso podemos ver q la prediccion empeora y es debido a la info aportada por newspaper
### se ve un pvalor=0.860(muy cercano a 1, osea muy grande => aceptamos hipotesis nula), se ve q
### el intervalo de confianza incluye el cero
### esto se debe a que existe una multicolinearidad entre Radio y Newspaper
lm = snf.ols(formula = 'Sales~TV + Radio + Newspaper', data = data).fit()  #linear model
lm.summary()
sales_pred = lm.predict(data[['TV', 'Newspaper', 'Radio']])
SSD = sum((data['Sales'] - sales_pred) ** 2)
SRE = np.sqrt(SSD/(len(data)-3))
sales_m = np.mean(data['Sales'])
error = SRE/sales_m
