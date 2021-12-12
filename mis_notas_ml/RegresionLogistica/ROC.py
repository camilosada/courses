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
import matplotlib.pyplot as plt 
from sklearn import metrics
from ggplot import *

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

# Evaluacion y validacion del modelo
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3,random_state=0)
lm = linear_model.LogisticRegression(solver='liblinear')
lm.fit(X_train,Y_train)


probs = lm.predict_proba(X_test) # probabilidad de estar seguros del resultado del modelo(primer columna)
                                 # segunda colum me da la probabilidad de salida
#prediction = lm.predict(X_test) # resultado de la prediccion (en este caso 0:no compra,1:compra)
# por defecto Yp = 0 si p<0.5, 1 si p>0.5 (0.5 es epsilon)

# para cabiar epsilon defino el threshold:
prob=probs[:,1]
prob_df = pd.DataFrame(prob)
threshold = 0.1
prob_df['prediction'] = np.where(prob_df[0]>=threshold,1,0)
prob_df['actual'] = list(Y_test) # si no le pongo el list hay problema con el indice de ubicacion
prob_df.head(20)

################ Matriz de confucion y curvas ROC
confusion_matrix = pd.crosstab(prob_df.prediction, prob_df.actual)
TN = confusion_matrix[0][0] # True negative
TP = confusion_matrix[1][1] # True positive
FP = confusion_matrix[0][1] # False negative
FN = confusion_matrix[1][0] # False positive
sensibility = TP/(TP + FN)
especifidad_1 = 1-TN/(TN+FP)  


##### Curva ROC
# Cuanto menor es el threshold mas alta es la sensibilidad y (1-especifidad)
thresholds = [0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.3, 0.4]
sens = [1]
espec_1 = [1]

for t in thresholds:
    prob_df['prediction'] = np.where(prob_df[0]>=t,1,0)
    prob_df['actual'] = list(Y_test) # si no le pongo el list hay problema con el indice de ubicacion

    confusion_matrix = pd.crosstab(prob_df.prediction, prob_df.actual)
    TN = confusion_matrix[0][0] # True negative
    TP = confusion_matrix[1][1] # True positive
    FP = confusion_matrix[0][1] # False negative
    FN = confusion_matrix[1][0] # False positive

    sensibility = TP/(TP + FN)
    sens.append(sensibility)
    especifidad_1 = 1-TN/(TN+FP) 
    espec_1.append(especifidad_1) 
sens.append(0) 
espec_1.append(0) 

plt.plot(espec_1,sens, marker = 'o', linestyle = '--', color = 'r')
x = [i*0.01 for i in range(100)]
y = [i*0.01 for i in range(100)]
plt.plot(x,y)
plt.xlabel('1-Especifidad')
plt.ylabel('Sensibilidad')
plt.title('Curva ROC')

# Cuanto mas grande es el area por debajo de la curva ROC, mejor es el modelo
espc_1, sensit, _ = metrics.roc_curve(Y_test, prob)

df = pd.DataFrame({
    'x': espc_1,
    'y': sensit
})

ggplot(df, aes(x = 'x', y = 'y')) + geom_line() + geom_abline(linetype = 'dashed') + xlim(0,1) + ylim(0,1)
auc = metrics.auc(espc_1,sensit) # area bajo la curva, esta entre 0 y 1
ggplot(df,aes(x = 'x', y = 'y')) + geom_area(alpha = 0.25) + geom_line(aes(y = 'y')) + ggtitle('Curva ROC y AUC = %s'% str(auc))