# Modelos de regresion lineal
## Modelo con datos simulados
### y = a + b * x
### x: 100 valores distribuidos segun una N(1.5,2.5)
### Ye = 5 + 1.9 * x + e
### e estara distribuido segun una N(0, 0.8)

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

x = 1.5 + 2.5 * np.random.randn(100)
res = 0 + 0.8 * np.random.randn(100) # residuos

y_pred = 5 + 1.9 * x # predeccion
y_act = 5 + 1.9 * x + res # lo que mido

x_list = x.tolist()
y_pred_list = y_pred.tolist()
Y_act_list = y_act.tolist()

data = pd.DataFrame(
    {
        'x':x_list,
        'y_actual':Y_act_list,
        'y_prediccion':y_pred_list
    }
)
y_values_mean = [np.mean(y_act) for i in range(1, len(x_list)+1)]
plt.plot(x,y_pred)
plt.plot(x,y_act,'ro')
plt.plot(x, y_values_mean, 'g')
plt.title('Valor actual vs prediccion')

## Suma de los cuadrados de la regresion
data['SSR'] = (data['y_prediccion'] - np.mean(y_act)) ** 2 
## Suma de los cuadrados de la diferencia
data['SSD'] = (data['y_prediccion'] - data['y_actual']) ** 2 # este en general se calcula diferencia directamente
## Suma de los cuadrados totales
data['SST'] = (data['y_actual'] - np.mean(y_act)) ** 2

SSRa = sum(data['SSR'])
SSDa = sum(data['SSD'])
SSTa = sum(data['SST'])

SSRa + SSDa # = SST
R2a = SSRa/SSTa


# Obtener la recta de regresion
x_mean = np.mean(data['x'])
y_mean = np.mean(data['y_actual'])

data['beta_n'] = (data['x']-x_mean)*(data['y_actual']-y_mean) # covarianza
data['beta_d'] = (data['x']-x_mean) ** 2 # varianza

beta = sum(data['beta_n'])/sum(data['beta_d'])
alpha = y_mean - beta * x_mean
### Modelo linear obtenido por regresion: y = alpha + beta * x

data['y_model'] = alpha + beta * data['x']


SSRb = sum((data['y_model'] - y_mean) ** 2 )
SSDb = sum((data['y_model'] - data['y_actual']) ** 2 )
SSTb= sum((data['y_actual'] - y_mean) ** 2)
R2b = SSRb/SSTb
R2a, R2b 

plt.plot(x,y_pred,'k')
plt.plot(x,y_act,'ro')
plt.plot(x, y_values_mean, 'g')
plt.plot(x,data['y_model'],'b')
plt.title('Valor actual vs prediccion')


# Error estandar de los residuos (RSE)

RSE = np.sqrt(SSDb/(len(data)-2)) # desviacion tipica de los residuos 
Porcentaje_error = RSE / y_mean 
