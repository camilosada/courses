import pandas as pd
import numpy as np

# Dummy Data Sets
data = pd.DataFrame(
    {
        'A': np.random.randn(10),
        'B': 1.5 + 2.5 * np.random.randn(10),
        'C': np.random.uniform(5,32,10)
    }
)

data = pd.read_csv("C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt")
data.head()
columns_names = data.columns.values.tolist()
a = len(columns_names)

new_data = pd.DataFrame(
    {
        'Columns names': columns_names,
        'A': np.random.randn(a),
        'B': np.random.uniform(0,1,a)
    }, index = range(42, 42+a)
)

# Agregacion de datos por categoria

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
