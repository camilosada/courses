# Multicolinealidad
# Hace relacion a la correlacion entre variables predictoras del modelo

# Cuando hay una correlacion signiificativa entre las variables predictoras,
# puede causar problemas en el modelo, incrmenta la variabilidad del coeficiente
# estimado para la variable predictora (una interactua negativamente con la otra)

# Factor de la inflacion de la varianza (VIF): se calcula para cada variable y si el valor es
# muy alto, esa variable hay que eliminarla
# si VIF = 1 : no hay correlacion entre variable
# si 1>VIF<5 : la variables correlacionadas moderadamente (aceptable)
# si VIF > 5 : la variable debe ser eliminada

import pandas as pd
import statsmodels.formula.api as snf
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/ads/Advertising.csv')
data.head()

# Newspaper ~ TV + Radio -> VIF = 1/(1-R2)
lm_n = snf.ols(formula = 'Newspaper ~ TV + Radio', data = data).fit()
rsquared_n = lm_n.rsquared
VIF_n = 1/(1-rsquared_n)
VIF_n
# TV ~ Newspaper + Radio -> VIF = 1/(1-R2)
lm_tv = snf.ols(formula = 'TV ~ Newspaper + Radio', data = data).fit()
rsquared_tv = lm_tv.rsquared
VIF_tv = 1/(1-rsquared_tv)
VIF_tv
# Radio ~ TV + Newspaper -> VIF = 1/(1-R2)
lm_r = snf.ols(formula = 'Radio ~ TV + Newspaper', data = data).fit()
rsquared_r = lm_r.rsquared
VIF_r = 1/(1-rsquared_r)
VIF_r

# los VIF de la radio y el newspaper son casi iguales y eso quiere decir que 
# estan corelacionadas, pero no con la TV. Y entre la radio y el newspaper, me 
# quedo con la relacion TV + radio que es la que mas bajo VIF me da