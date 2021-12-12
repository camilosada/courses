# Analisis exploratorio de los datos

import pandas as pd 
import numpy as np 

data = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/bank/bank.csv', sep=';')
data.head()
data.columns.values
data['y'] = (data['y'] == 'yes').astype(int)

data['education'].unique() # veo todas las posibilidades con las que se completa esta columna
# cambio los nombres para faciliar y para dejar todo en el mismo formato de escritura
data['education'] = np.where(data['education']=='basic.4y','Basic', data['education'])
data['education'] = np.where(data['education']=='basic.6y','Basic', data['education'])
data['education'] = np.where(data['education']=='basic.9y','Basic', data['education'])

data['education'] = np.where(data['education']=='high.school','High School', data['education'])
data['education'] = np.where(data['education']=='professional.course','Professional Course', data['education'])
data['education'] = np.where(data['education']=='university.degree','University Degree', data['education'])

data['education'] = np.where(data['education']=='iliterate','Iliterate', data['education'])
data['education'] = np.where(data['education']=='unknown','Unknown', data['education'])

data['education'].unique()

data['y'].value_counts()
data.groupby('y').mean()

pd.crosstab(data.education, data.y).plot(kind='bar')

table = pd.crosstab(data.marital,data.y)
# divido para que me quede entre 0 y 1 y sea facil hacer la comparacion
table.div(table.sum(1).astype(float), axis = 0).plot(kind='bar', stacked = True)

