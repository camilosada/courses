import numpy as np
import pandas as pd

FullPath = "C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt"
data = pd.read_csv(FullPath)

# Generar num aleatorio entre 1 y 100
np.random.randint(1,100)

# Forma clasica de generar num aleatorio entre 0 y 1
np.random.random()

# Funcion que genere una lista de n numeros aleatorios enteros dentro de intervalo (a,b)
def rand_list(n,a,b):
    x=[]
    for i in range(n):
        x.append(np.random.randint(a,b))
    return x

rand_list(5,4,8)

# Libreria qe ya tiene la funcion anterior
import random
random.randrange(0,100,5) # el ultimo elemento de la funcion es para q sean multiplos de ese numero

# Shuffling
a = np.arange(5) # Genera un array en el rango q le paso
np.random.shuffle(a) #  Cambia de posicion los datos del array

# Choice 
# Elegir al azar un elemento de la lista
columnas = data.columns.values.tolist()
np.random.choice(columnas)

# Seed
# Establecer una semilla es importante para la reproductivilidad del experimento
# Ya que si pongo una semilla los numeros aleatorios siempre van a ser los mismos
np.random.seed(2018) 
np.random.random()