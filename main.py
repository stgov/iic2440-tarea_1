import numpy as np
import pandas as pd
import re

def k_shingles(df, k):
    shingles = []
    for words in df['text']:
        for i in range(0, len(words)):
            shingles.append(' '.join(words[i:i+k]))
    return list(set(shingles)) # Retornamos valores unicos de shingles.


# Leemos datos y formateamos columna de tweets.
# Borramos tweets repetidos (Se borran muchos, pasamos como de 4 millones a 1.3 millones)
# Borramos puntos y comas, transformamos a lower_case y borramos emojis.

tweets = pd.read_csv('/home/bcm/Desktop/PUC/Procesamiento de datos masivos/T1_Grupo/iic2440-tarea_1/Datos/tweets_2022_abril_junio.csv')
tweets = tweets.drop_duplicates(subset='text', keep=False)
tweets['text'] = tweets['text'].str.replace('[.,]', '', regex=True).str.lower().apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x)).str.split()

# Calculamos los posibles shingles unicos segun un 'k'
shingles = k_shingles(tweets, 2)

# Ahora hay que hacer una matriz caracteristica en donde las filas son los shingles, y las columnas son los tweets, y despues calcular la distancia de Jaccard.
# Pero son demasiados los datos, asique no podemos, la matriz seria (numero_tweets) * (shingles) = 1.371.764* 3.139.706 = Mi computador casi explota al intentar crear esta matriz.


