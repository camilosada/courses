import pandas as pd

FullPath = "C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/titanic/titanic3.csv"
data = pd.read_csv(FullPath)
data.head() # primeros datos de la tabla
data.shape # numero de filas y columnas de la tabla
data.tail() # ultimos datos de la tabla
data.columns.values # valor de las columnas de cabecera
data.describe() # estadisticas basicas de las variables numericas
data.dtypes # tipo de dato de cada variable (object = string)
pd.isnull(data["body"]) # saber si el valor es nulo
pd.notnull(data["body"]) # saber si el valor es no nulo
pd.isnull(data["body"]).values.ravel().sum() # values: me lo convierte a objeto array, 
                                             # ravel: construye un unico array de datos,
                                             # sum: me suma todos los 1 de los true que me devuelve isnull()
# Los valores que faltan en un dataset pueden venir por dos razones
## *Extraccion de los datos: por incompatibilidad entre el servidor de la base de datos y el proceso de extraccion
## *Recoleccion de los datos

# Que hacer con los valores NULL??
### Un primer enfoque que es bastante drastico: eliminar la fila/columna entera

data.dropna(axis=0, how='any') # axis=0/1: borra toda la fila/columna,
                               # how me da la condicion, all: si todas las columnas son NULL
                               # any: si alguna columna es NULL

### Computo de valores faltantes
#### Rellenar con determinado valor el valor NULL
data.fillna(0) # Rellena los NULL con 0
data2 = data
data2['body'] = data['body'].fillna(0)
data2['home.dest'] = data['home.dest'].fillna('desconocido')

#### Sustituir los valores que faltan por la media, mediana, etc

data['age'].fillna(data['age'].mean()) #reemplazo por la media

data['age'][1291] # valor en fila 1291 de la columna age
data['age'].fillna(method='ffill') # ffill(foward fill) completa con el valor(conocido) de la pos anterior
data['age'].fillna(method='backfill') # completa con el proximo valor conocido
# ATENCION: si uso ffill si el primer valor de la columna es NULL, este queda NULL ya que no existe valor anterior 

# Creacion de variables Dummy
dummy_sex = pd.get_dummies(data['sex'], prefix='sex') # me crea columnas con las variables de sex y pone uno y cero segun corresponda a cada fila
colum_name = data.columns.values.tolist() # primer fila de data en un array (solo para ver)
data3 = data.drop('sex',axis=1) # elimino columna (axis=1) de data
pd.concat([data3, dummy_sex], axis=1) # concatena las tablas

