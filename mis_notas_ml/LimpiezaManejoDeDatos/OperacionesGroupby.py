import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#---------------------- Creo el DF --------------------------
gender = ['Male', 'Female']
income = ['Poor', 'Middle Clase', 'Rich']
gender_data = []
income_data = []
n = 500
for i in range(0,500):
    gender_data.append(np.random.choice(gender))
    income_data.append(np.random.choice(income))
height = 160 + 30 * np.random.randn(n)
weight = 65 + 25 * np.random.randn(n)
age = 30 + 12 * np.random.randn(n)
income = 160000 + 3500 * np.random.randn(n)
data = pd.DataFrame(
    {
        'Gender':gender_data,
        'Economic Status':income_data,
        'Height':height,
        'Weight':weight,
        'Age':age,
        'Income':income
    }
)
#-------------------------------------------------------------------

## Agrupacion de datos

grouped_gender = data.groupby('Gender')
##### groupby me genera objetos tipo groupby
grouped_gender.groups
len(grouped_gender) # con esto se el numero de grupos 
for names, groups in grouped_gender:
    print(names)
    print(groups)

grouped_gender.get_group('Female')

double_group = data.groupby(['Gender','Economic Status'])
for names, groups in double_group:
    print(names)
    print(groups)
len(double_group)

## Operaciones sobre datos agrupados
double_group.sum()
double_group.mean()
double_group.size()
double_group.describe()
grouped_income = double_group['Income']
grouped_income.describe()

double_group.aggregate( ## le aplico alguna operacion a las columnas que quiero
    {
        'Income':np.sum,
        'Age':np.mean,
        'Height':np.std
    }
)

double_group.aggregate(
    {
        'Age':np.mean,
        'Height': lambda h:(np.mean(h)/np.std(h)) # Tipificacion
    }
)

### Para aplicarle varias operaciones a todas las variables(columnas)
double_group.aggregate([np.sum, np.mean, np.std, lambda h:(np.mean(h)/np.std(h))])


# Agrupacion de datos

double_group['Age'].filter(lambda x: x.sum()>2400)
a = double_group.transform(lambda x:((x-x.mean())/x.std())) # Se estandarizaron los datos (si veo el hist es una campana de Gauss)
plt.hist(a['Age'])

#### Si hubiese valores Na puedo hacer
fill_Na_mean = lambda x: x.fillna(x.mean())
double_group.transform(fill_Na_mean)

### Operaciones diversas muy utiles
double_group.head(1) # Me devuelve primera fila de cada grupo
double_group.nth(20) # Me devuelve el elemento n de cada grupo
## Antes de agrupar ordeno los valores
data_sorted = data.sort_values(['Age','Income']) # Me ordena por edad y en caso de empate por income
data_sorted_grouped = data_sorted.groupby(['Gender'])
data_sorted_grouped.head(1)