# Probabilidades condicionadas

import pandas as pd
df = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/gender-purchase/Gender Purchase.csv')
df.head()

# P_m= probabilidad de que compre sabiendo que es hombre = compras realizadas por hombres/ total de hombres
# P_f= probabilidad de que compre sabiendo que es mujer = compras realizadas por mujeres/ total de mujeres

# Ratio de probabilidades(odds)
# odds de compras hechas por hombres = p_m/(1-p_m) = cantidad de hombres que compran/ cant hombres que no compra
# odds de compras hechas por mujeres = p_f/(1-p_f) = cantidad de mujeres que compran/ cant mujeres que no compra

# * Si el ratio es superior a 1, es mas probable el exito que el fracaso
# * Si el ratio es 1, exito y fracaso son equiprobables(p=0.5)
# * Si el ratio menor a 1, el fracaso es mas probable que el exito


# De la regresion lineal a la logistica
# P = 1 / (1 + e^ -(a + b*x))
# * Si a + b*x es muy pequeno(negativo) -> P tiende a 0
# * Si a + b*x es 0 -> P = 0.5
# * Si a + b*x es muy grande(positivo) -> P tiende a 1

# Para la regresion logistica multiple
# P = 1 / (1 + e^ -(a + sum(bi*xi)))