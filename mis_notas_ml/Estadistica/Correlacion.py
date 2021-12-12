import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_ads = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/ads/Advertising.csv')
data_ads.head()

data_ads['corrn'] = (data_ads['TV'] - np.mean(data_ads['TV'])) * (data_ads['Sales'] - np.mean(data_ads['Sales']))
data_ads['corr1'] = (data_ads['TV'] - np.mean(data_ads['TV'])) ** 2 
data_ads['corr2'] = (data_ads['Sales'] - np.mean(data_ads['Sales'])) ** 2

coef_correlacion = sum(data_ads['corrn']) / (np.sqrt(sum(data_ads['corr1'] ) * sum(data_ads['corr2'] )))
coef_correlacion

## Funcion que me lo hace directamente
data_ads.corr()
plt.matshow(data_ads.corr())

