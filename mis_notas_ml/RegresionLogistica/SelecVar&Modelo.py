# conversion de las variables categoricas a dummies
import pandas as pd 
import numpy as np 
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics 

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

# Implementacion del modelo
## primero uso statsmodels.api para armar el modelo y verificar los valores (miro mas que nada el pvalor)
logit_model =  sm.Logit(Y,X)
result = logit_model.fit()
result.summary2()

## ahora uso sklearn import linear_model para ajustar el modelo y ver exactamente el resultado
logit_model = linear_model.LogisticRegression()
logit_model.fit(X,Y)
logit_model.score(X,Y)


# Evaluacion y validacion del modelo
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3,random_state=0)
lm = linear_model.LogisticRegression()
lm.fit(X_train,Y_train)

probs = lm.predict_proba(X_test) # probabilidad de estar seguros del resultado del modelo(primer columna)
                                 # segunda colum me da la probabilidad de salida
prediction = lm.predict(X_test) # resultado de la prediccion (en este caso 0:no compra,1:compra)
# por defecto Yp = 0 si p<0.5, 1 si p>0.5 (0.5 es epsilon)

# para cabiar epsilon defino el threshold:
prob=probs[:,1]
prob_df = pd.DataFrame(prob)
threshold = 0.4
prob_df['prediction'] = np.where(prob_df[0]>threshold,1,0)
prob_df.head()
pd.crosstab(prob_df.prediction, columns = 'count')
390/len(prob_df) # veo que aumente la probabilidad de venta a 31%
metrics.accuracy_score(Y_test, prob_df.prediction) # eficacia del modelo(epsilon = threshold)
metrics.accuracy_score(Y_test,prediction)# se ve como aumento la eficacia del modelo utilizando conjunto de testing (epsilon =0.5)
