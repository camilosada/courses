import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

red_wine_path = "C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/wine/winequality-red.csv"
red_wine = pd.read_csv(red_wine_path, sep=';')
white_wine_path = "C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/wine/winequality-white.csv"
white_wine = pd.read_csv(white_wine_path, sep=';')
white_wine.head()

#### En Python tenemos dos tipos de ejes, 
#### axis= 0 es el eje horizontal, axis =1 es el eje vertical

wine_data = pd.concat([red_wine,white_wine], axis = 0)
wine_data.head()
wine_data.shape

## Cargar varios ficheros

#### Cargar el primer fichero
#### Hacer un bucle para cargar cada uno de los ficheros(importante que haya correlacion con los nombres)
filePath =  "C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/distributed-data/"
data = pd.read_csv(filePath + '001.csv')
final_len = len(data) ## para controlar que se hayan cargado bien los datos
for i in range(2,233):
    if i<10:
        fileName = '00' + str(i)
    elif 10>= i <100:
        fileName = '0' + str(i)
    elif i>100:
        fileName = str(i)
    file = filePath + fileName + '.csv'
    temp_data = pd.read_csv(file)
    data = pd.concat([data,temp_data], axis=0)
    final_len += len(temp_data) 
    
final_len == data.shape[0]
#########

filePath = 'C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/athletes/'
data_main = pd.read_csv(filePath + 'Medals.csv', encoding= 'ISO-8859-1')
a = data_main['Athlete'].unique().tolist()
data_country = pd.read_csv(filePath + 'Athelete_Country_Map.csv', encoding= 'ISO-8859-1')
data_country
data_country[data_country['Athlete'] == 'Aleksandar Ciric']
data_sport = pd.read_csv(filePath + 'Athelete_Sports_Map.csv', encoding= 'ISO-8859-1')
data_sport.head()
data_sport[(data_sport['Athlete'] == 'Chen Jing') | (data_sport['Athlete'] == 'Matt Ryan')]

## Merge (por defecto es inner)
data_main_country = pd.merge(left = data_main, right = data_country, # left/right de que lado va cada data
                              left_on = 'Athlete', right_on = 'Athlete') # left_on/right_on que columnas une
data_country_dp = data_country.drop_duplicates(subset = 'Athlete')

## Tipos de Join
##### Primero voy a eliminar algunos atletas de la lista
out_athletes = np.random.choice(data_main['Athlete'], size = 6, replace = False)
data_country_dlt = data_country_dp[(~data_country_dp['Athlete'].isin(out_athletes)) & 
                                    (data_country_dp['Athlete'] != 'Michael Phelps')]

### Inner Join devuelve las filas con los valores que se encuentran en ambos conjuntos
##### En data_main tengo todos los datos y en data_country_dlt falta la info de 7 atletas

merged_inner = pd.merge(left = data_main, right = data_country_dlt, # left/right de que lado va cada data
                              how = 'inner', left_on = 'Athlete', right_on = 'Athlete') # left_on/right_on que columnas une
len(data_main)
len(merged_inner)
### Left Join devuelve las filas del dataset situado a la izq(lo que se encuentra en la interseccion tiene 
### info del derecho mientras que lo que esta a la izq se completa con NULL )

merged_left = pd.merge(left = data_main, right = data_country_dlt, # left/right de que lado va cada data
                              how = 'left', left_on = 'Athlete', right_on = 'Athlete') # left_on/right_on que columnas une
len(merged_left)
### Right Join devuelve las filas del dataset situado a la der

merged_rigth = pd.merge(left = data_main, right = data_country_dlt, # left/right de que lado va cada data
                              how = 'right', left_on = 'Athlete', right_on = 'Athlete') # left_on/right_on que columnas une
len(merged_rigth)
### Outer Join devuelve las filas de ambos conjuntos completando con NULL donde no hay interseccion
##### Voy a agregar una persona para verificar que efectivamente es todo el conjjunto
data_country_dlt.append(
    {
        'Athlete':'Camila Losada',
        'Country': 'Argentina'
    }, ignore_index = True    # para que lo agregue a lo ultimo
)
merged_outer = pd.merge(left = data_main, right = data_country_dlt, # left/right de que lado va cada data
                              how = 'outer', left_on = 'Athlete', right_on = 'Athlete') # left_on/right_on que columnas une
len(merged_outer)