

def downloadFromURL(url,fileName,extension, sep = ",", delim = "\n", encoding = 'utf-8'):
    import pandas as pd
    ## Leer datos desde URL externa
   

    ### leer con metodo open
    import csv # metodos para gestionar ficheros tipo csv
    import urllib3 # para navegar y acceder a informacion desde una url
    http=urllib3.PoolManager()
    r= http.request('GET',url)
    r.status #el estado 200 es que se pudo conectar con el servidor sin problema
    data = r.data

    ### el objeto data4 contiene un string binario, asi que lo convierto en un string
    ### decodificandolo en UTF-8
    data_str = data.decode(encoding)

    #### divido el string en un array de filas, separandolo por \n
    data_str_lines = data_str.split(delim)

    #### extraigo la primer linea que son los nombres de las columnas(cabecera)
    data_str_nameCols = data_str_lines[0].split(sep)
    data_str_numCols = len(data_str_nameCols)

    #### genero DICCIONARIO vacio que es donde va a ir toda la info procesada desde la URL externa
    counter = 0
    main_dic = {}
    for col in data_str_nameCols:
        main_dic[col]=[]

    # Proceso fila a fila para rellenar el diccionario
    for line in data_str_lines:
        if counter>0:
            values = line.strip().split(sep)
            for i in range(data_str_numCols):
                main_dic[data_str_nameCols[i]].append(values[i])
        counter += 1

    # Convierto el diccionario a dataFrame (corroborar que los datos sean correctos)
    data_df = pd.DataFrame(main_dic)

    # Elijo direccion donde guardarlo
    #filename= "C:/Users/camil/OneDrive/Escritorio/PracticaAI/python-ml-course-master/datasets/pruebaMia/medals_data4"

    # Lo guardo en CSV, Excel o JSON (hecho con los 3 ejemplos)
    data_df.to_csv(fileName + extension)
    #medals_data4_df.to_excel(fullPath + ".xls")
    #medals_data4_df.to_json(fullPath + ".json")
    
    return data_df
