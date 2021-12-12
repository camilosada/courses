import pandas as pd

FullPath = "C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt"
data = pd.read_csv(FullPath)
data.head() # primeros datos de la tabla

## Crear un subconjunto de datos
account_length = data['Account Length']
account_length.head()
##### cuando seleccionamos una sola columna del df, obtenemos un objeto 
##### de tipo series(vectores)
type(account_length)
#### Extraer varias columnas del df
subset = data[['Account Length', 'Phone', 'Eve Charge', 'Day Calls']]
subset.head()
##### Esto es importante hacerlo antes de comenzar con los analisis para quedarse solo con los datos que nos 
##### van a ser utiles ya que de esta forma se reduce la cantidad de espacio en disco y en memoria necesaria
##### Siempre combiene quedarse con la lista de columnas mas chica (entre las deseadas y no deseadas), para evitar errores
### Ejemplo, las no deseadas son menos que las deseadas:
no_desired_columns = data[['Int\'l Plan', 'VMail Message', 'Intl Mins']]
all_columns = data.columns.values.tolist()
desired_columns = [x for x in all_columns if x not in no_desired_columns]
subset2 = data[desired_columns]
subset2.head()

#### Seleccionar filas
data[3:8] # Obtengo de la fila num 3 a la num 7 (el ultimo no aparece, es decir el 8)
data[300:] # Obtengo desde el 300 hasta el ultimo

##### Seleccionar usuarios con Day Mins>210
condicion = data['Day Mins']>295 # Esto me devuelve T o F segun cumple la condicion o no
data1 = data[condicion]
data1

##### Seleccionar usuarios State=NY
condicion2 = data['State'] =='NY'
data2 = data[condicion2]
data2

##### Seleccionar con dos condiciones
###### AND -> & 
condicion3 = ((data['State'] =='NY') & (data['Day Mins']>300))
data3 = data[condicion3]
data3
###### OR -> |
condicion3 = ((data['State'] =='NY') | (data['Day Mins']>300))
data3 = data[condicion3]
data3

##### Seleccionar comparando dos filas
condicion4 = (data['Day Mins'] < data['Night Mins'] )
data4 = data[condicion4]
data4

### Filtrar por fila y por columna a la vez
#### Minutos de dia, de noche y longitud de los primeros 50 individuos
subset_first_50 = data [['Day Mins', 'Night Mins', 'Account Length']][:50] # Primer corchete columnas, segundo filas
subset_first_50

##### Seleccionar filas y columnas de la tabla
###### acceso por posicion
data.iloc[1:10,3:6] # primero filas, despues columnas
data.iloc[1:10,[1,5]]
###### acceso por etiqueta
data.loc[1:5,['Area Code','Day Mins']]


### Crear una columna nueva en el df
#### Agrego una columna nueva llamada Total Mins y le asigno como valores la suma de las otras columnas
data['Total Mins'] = data['Day Mins'] + data['Night Mins'] + data['Eve Mins']
data['Total Mins'].head()