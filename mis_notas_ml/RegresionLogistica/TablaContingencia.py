
# Matematicas detras de la regresion logistica
# Las tablas de contingencia
import pandas as pd
df = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/gender-purchase/Gender Purchase.csv')
df.head()
contingency_table = pd.crosstab(df['Gender'], df['Purchase'])

contingency_table.sum(axis = 1) # cantidad de hombres y mujeres
contingency_table.sum(axis = 0) # cuantos entraron y cuantos no
# para calcular porcentaje:
contingency_table.astype('float').div(contingency_table.sum(axis = 1), axis = 0)

