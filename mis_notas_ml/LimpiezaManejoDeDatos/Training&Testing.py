import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FullPath = "C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt"
data = pd.read_csv(FullPath)

## Conjunto de entrenamiento y conjunto de testing

### Dividir utilizando la distribucion normal
a = np.random.randn(len(data))
plt.hist(a)
check = (a<0.8)
check
plt.hist(check.astype(int))
training = data[check]
testing = data[~check]
testing

### Con la libreria sklearn (libreria de aprendisaje estadistico de Python)
from sklearn.model_selection import train_test_split
import sklearn
train, test = train_test_split(data, test_size = 0.2)

### Usando una funcion de Shuffle
sklearn.utils.shuffle(data)
cut_id = int(0.75*len(data))
train_data = data[:cut_id]
test_data = data[cut_id+1:]