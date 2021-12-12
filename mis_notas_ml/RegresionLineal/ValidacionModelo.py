# Validacion del modelo
# Dividir el dataset en conjunto de entremaniento y de testing

import pandas as pd
import statsmodels.formula.api as snf
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/ads/Advertising.csv')
data.head()

# genero numeros aleatorios con una distribucion normal(misma cant que largo de data)
a = np.random.randn(len(data))
plt.hist(a)
# divido 20% para testing y 80% para entrenamiento
check = (a<0.8)
training = data[check]
testing = data[~check]

lm = snf.ols(formula = 'Sales~TV+Radio', data = training).fit()
lm.summary()

# Validacion del modelo con el conjunto de testing
sales_pred = lm.predict(testing)
SSD = sum((testing['Sales']-sales_pred)**2)
RSE = np.sqrt(SSD/(len(testing)-2-1))
error = RSE/(np.mean(testing['Sales']))
