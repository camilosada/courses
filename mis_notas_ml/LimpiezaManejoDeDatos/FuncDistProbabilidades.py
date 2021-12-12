# Funciones de distribucion de probabilidades

import numpy as np
import matplotlib.pyplot as plt

## Distribucion uniforme
a = 1
b = 100
n = 200
data = np.random.uniform(a,b,n)
plt.hist(data)

## Distribucion normal (campana de Gauss)
data = np.random.randn(10000)
x = range(1, 101)
plt.plot(x, data)
plt.hist(data)

plt.plot(x,sorted(data)) # Funcion de distribucion acomodada

# Si quiero especificar la media y la desviacion estandar
mu = 5.5
sd = 2.5
data = 5.5 + 2.5 * np.random.randn(10000)
plt.hist(data)

data = np.random.randn(2,5) # Dos arrays con dist normal, de 5 valores


## Simulacion de Monte Carlo
### Generamos dos numeros aleatorios x e y entre 0 y 1
### Calculamos x*x+y*y
### Si el valor es inferior a 1 -> estamos dentro del circulo
### Si el valor es superior a 1 -> estamos fuera del circulo
### Calculamos el numero total de veces que estas dentro del circulo y lo 
### dividimos entre el numero total de intentos para obtener una aproximacion 
### de la probabilidad de caer dentro del circulo
### Usamos dicha probabilidad para aproximar el valor de pi

n=1000
value = 0
x = np.random.uniform(0,1,n).tolist()
y = np.random.uniform(0,1,n).tolist()
for i in range(n):
    z = np.sqrt(x[i] * x[i] + y[i] * y[i])
    if z<=1:
        value += 1
float_value = float(value)
pi_value = float_value * 4 /n


