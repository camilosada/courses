
import downloadDataFromURL
""" 
import pandas as pd
mainPath= "C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/"
data = pd.read_csv(mainPath + "titanic/titanic3.csv")
data2=pd.read_csv(mainPath + "customer-churn-model/Customer Churn Model.txt")
data2.head()

data2_columns=pd.read_csv(mainPath + "customer-churn-model/Customer Churn Columns.csv")
data2_columns_list=data2_columns["Column_Names"].tolist()
data2=pd.read_csv(mainPath + "customer-churn-model/Customer Churn Model.txt", header=None, names=data2_columns_list)


print(data.columns.values)
print(data2)
print("Hola Mundo")

medals_url = "http://winterolympicsmedals.com/medals.csv"
medals_data = pd.read_csv(medals_url)
medals_data.head() """


url="http://winterolympicsmedals.com/medals.csv"
fileName="C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/pruebaMia/aaa"
extension=".csv"
data_Df = downloadDataFromURL.downloadFromURL(url,fileName,extension)
print(data_Df.head())