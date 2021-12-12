# Tratamiento de variables categoricas
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/ecom-expense/Ecom expense.csv')
df.head()

# Creo variables dummy que representes a las variables categoricas
dummy_gender = pd.get_dummies(df['Gender'], prefix = 'Gender')
dumy_city_tier = pd.get_dummies(df['City Tier'], prefix = 'City')

# agrego los dummy a mi df
column_names = df.columns.values.tolist()
df_new = df[column_names].join(dummy_gender) # no es necesario poner df[], con df solo anda(pone todas las columnas)
column_names2 = df_new.columns.values.tolist()
df_new = df_new[column_names2].join(dumy_city_tier) 

# Elijo las columnas para hacer la regresion
feature_cols = ['Monthly Income', 'Transaction Time',
                'Gender_Female', 'Gender_Male',
                'City_Tier 1', 'City_Tier 2', 'City_Tier 3',
                 'Record']
X = df_new[feature_cols]
Y = df_new['Total Spend']


# Hago la regresion
lm = LinearRegression()
lm.fit(X,Y)
lm.intercept_
lm.coef_
# Para ver cada variable con su coeficiente
list(zip(feature_cols,lm.coef_))
# para ver que tan bueno es el modelo veo el R2:
lm.score(X,Y) 

# escribo el modelo, este modelo es el completo y el que engloba todo: (mas abajo ver en enmascarado de var. redundantes)
df_new['prediction'] =-79.41713030136998 + df_new['Monthly Income']* 0.14753898049205735 +df_new['Transaction Time']* 0.15494612549589526+df_new['Gender_Female']* (-131.02501325554596)+df_new['Gender_Male'] *131.025013255546+df_new['City_Tier 1']* 76.76432601049537+df_new['City_Tier 2'] *55.138974309232324+df_new['City_Tier 3'] *(-131.90330031972775)+df_new['Record'] *772.2334457445642
# Pero en realidad lo vemos asi (en total hay 6):
# * Si es hombre y vive en CT1 -> df_new['prediction'] =-79.41713030136998 + df_new['Monthly Income']* 0.14753898049205735 +df_new['Transaction Time']* 0.15494612549589526+131.025013255546+76.76432601049537+df_new['Record'] *772.2334457445642
# queda en hombre y CTY 1 y en el resto 0
# * Si es hombre y vive en CTY2 ->..............

### Otra forma de hacer la prediccion
# df_new["prediction"] = lm.predict(pd.DataFrame(df_new[feature_cols]))

SSD = np.sum((df_new['prediction'] - df_new['Total Spend'])**2)
RSE = np.sqrt(SSD/(len(df)-len(feature_cols)-1))
error = RSE/np.mean(df['Total Spend'])

# Enmascarado de variables categoricas redundantes: eliminar variables dummy
dummy_gender = pd.get_dummies(df['Gender'], prefix= 'Gender').iloc[:,1:] # con iloc estoy agarrando todas las filas y de la primer col en adelante
dummy_city_tier = pd.get_dummies(df['City Tier'], prefix= 'City').iloc[:,1:] # con iloc estoy agarrando todas las filas y de la primer col en adelante

df_new = df[column_names].join(dummy_gender)
column_names2 = df_new.columns.values.tolist()
df_new = df_new[column_names2].join(dummy_city_tier)

# Elijo las columnas para hacer la regresion
feature_cols = ['Monthly Income', 'Transaction Time',
                 'Gender_Male',
                 'City_Tier 2', 'City_Tier 3',
                 'Record']
X = df_new[feature_cols]
Y = df_new['Total Spend']
lm = LinearRegression()
lm.fit(X,Y)
lm.intercept_
list(zip(feature_cols,lm.coef_))
lm.score(X,Y) # Se puede ver que a pesar de haber sacado las col dummy female y CTY1, el R2 me da igual 
              # de esta forma se simplifica el modelo
              
