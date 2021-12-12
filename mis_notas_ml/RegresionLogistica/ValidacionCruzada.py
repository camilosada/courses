# Validacion cruzada
# sirve para evitar overfiting 

import pandas as pd 
import numpy as np 
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.model_selection import cross_val_score

#--------------- Armado de X e Y para hacer el modelo---------
data = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/bank/bank.csv', sep=';')
data.head()
data['y'] = (data['y'] == 'yes').astype(int)
# Luego de hacer la exploracion de datos y ver cuales podrian tener impacto en la prediccion, selecciono esa columna
categories = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 
                'month', 'day_of_week', 'poutcome']
for category in categories:
    cat_list = 'cat'+'_'+category
    cat_dummies = pd.get_dummies(data[category], prefix=category)
    data_new = data.join(cat_dummies)
    data = data_new
data_vars = data.columns.values.tolist()
to_keep = [v for v in data_vars if v not in categories]
to_keep = [v for v in to_keep if v not in ['default']]
bank_data = data[to_keep]

bank_data_vars = bank_data.columns.values.tolist()
Y = ['y']
X = [v for v in bank_data_vars if v not in Y]

# Seleccion de rasgos para el modelo
n = 12
Lr = LogisticRegression(solver='liblinear')
rfe = RFE(Lr, n)
rfe = rfe.fit(bank_data[X], bank_data[Y].values.ravel())
rfe.support_
rfe.ranking_
list(zip(bank_data_vars,rfe.support_, rfe.ranking_))

cols = ['previous', 'euribor3m','job_management', 'job_technician', 'month_aug', 'month_dec', 'month_jul', 'month_jun',
        'month_mar', 'month_nov', 'day_of_week_wed', 'poutcome_nonexistent']

X = bank_data[cols]
Y = bank_data['y']
#---------------------------------------------------------------

scores = cross_val_score(linear_model.LogisticRegression(solver='liblinear'), X, Y, scoring = 'accuracy', cv=10)
#### cambiando el cv(cantidad de datos que agarro en cada iteracion es con lo que voy cambiando entre los distintos metodos de validacion)
scores.mean() # miro el medio siempre para ver que tanto mejoro