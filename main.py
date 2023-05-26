from scipy.sparse import coo_matrix
import numpy as np
from scipy.spatial import distance
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer
from datasketch import MinHash
import random
import time

# Funciones

def limpiar_retweet(tweet:str):
    if tweet.startswith('RT'):
        tweet = ''.join(tweet.split(': ')[1:])
    return tweet

def limpiar_hashtag(tweet:str):
    tweet = re.sub(r'#\w+', '', tweet)
    return tweet

def limpiar_url(tweet:str):
    tweet = re.sub(r'http\S+', '', tweet)
    return tweet

def limpiar_emoji(tweet:str):
    tweet = re.sub(r'\\x\w+', '', tweet)
    return tweet

def limpiar_puntuacion(tweet:str):
    tweet = re.sub(r'[^\w\s@]', '', tweet)
    return tweet

def limpiar_espacios(tweet:str):
    tweet = re.sub(r'\s+', ' ', tweet)
    return tweet

def limpiar_mayusculas(tweet:str):
    tweet = tweet.lower()
    return tweet

def remplazar_tildes(tweet:str):
    tweet = tweet.replace('á', 'a')
    tweet = tweet.replace('é', 'e')
    tweet = tweet.replace('í', 'i')
    tweet = tweet.replace('ó', 'o')
    tweet = tweet.replace('ú', 'u')
    return tweet

def limpiar_texto(tweet:str):
    tweet = limpiar_retweet(tweet)
    tweet = limpiar_hashtag(tweet)
    tweet = limpiar_url(tweet)
    tweet = limpiar_emoji(tweet)
    tweet = limpiar_puntuacion(tweet)
    tweet = limpiar_espacios(tweet)
    tweet = limpiar_mayusculas(tweet)
    tweet = remplazar_tildes(tweet)
    tweets = tweet.strip()
    return tweet

def shingles(k, words: np.ndarray):
    shingles = []
    for i in range(0, len(words)):
        if len((' '.join(words[i:i+k])).split()) != k:
            pass
        else:
            shingles.append(' '.join(words[i:i+k]))
    return list(set(shingles))

# Abrimos datos y los limpiamos

tweets = pd.read_csv(
    '/home/bcm/Desktop/PUC/Procesamiento de datos masivos/T1_Grupo/iic2440-tarea_1/Datos/tweets_2022_abril_junio.csv',
    usecols=['id', 'screen_name', 'text'],
    index_col='id',
    dtype={'screen_name': str, 'text': str},
    nrows=10000
    )

# data = {
#     'screen_name': ['r', 'r', 'r'],
#     'text': [
#         "The night is dark and the moon is red",
#         "The moon in the night is red",
#         "I can see moon is red, the night is dark"
#     ]
# }
# tweets = pd.DataFrame(data)

tweets = tweets.dropna()
tweets['text'] = tweets.loc[:, 'text'].apply(limpiar_texto)
tweets = tweets.drop_duplicates(subset='text', keep=False)


# Creamos Shingles

k = 3 # Shingles size
tweets['shingles'] = tweets['text'].str.split().apply(lambda x: np.array(x)).apply(lambda x: shingles(k, x))
shings = tweets['shingles'].to_numpy()
u_shingles = np.unique(np.concatenate(shings)) # Shingles unicos en una unica lista

# Conseguimos coordenadas de filas y columnas en donde shingle esta en tweet.
start_time = time.time()

rows = []
cols = []
data = []

for i, shingle in enumerate(u_shingles):
    for j, tweet in enumerate(tweets['text']):
        if shingle in tweet:
            rows.append(i)
            cols.append(j)
            data.append(1)
sparse_matrix = coo_matrix((data, (rows, cols)), shape=((len(u_shingles), len(tweets['text']))))


end_time = time.time()
execution_time = end_time - start_time
print(execution_time)

### Hasta aqui bien, tuve que arreglar los shingles, y drop duplicates despues


# Creamos minhashes
start_time = time.time()

minhashes = []
num_hashes = 2
num_perm = sparse_matrix.shape[0]

for i in range(num_hashes):
    minhash = MinHash(num_perm=num_perm, seed=i)

    for shingle in u_shingles:
        minhash.update(shingle.encode('utf-8'))
    minhashes.append(minhash)

end_time = time.time()
execution_time = end_time - start_time
print(execution_time)

# Creamos permute, que es la matriz con los minhashes y los valores randoms permutados

permute = np.zeros((len(minhashes), len(minhashes[0].hashvalues)))

for i, minhash in enumerate(minhashes):
    hash_values = minhash.hashvalues
    permute[i] = hash_values
permute = permute.T

# Ocupando permute y la caracteristica (Sparse matrix), calculamos la signature

num_documents = sparse_matrix.shape[1]
num_signatures = len(minhashes)
signature_matrix = np.zeros((num_signatures, num_documents))

sparse_matrix = sparse_matrix.tocsc()

for i in range(num_documents):
    non_zero_rows = np.nonzero(sparse_matrix[:, i])[0]

    if non_zero_rows.size > 0:
        for s in range(num_signatures):
            signature_matrix[s, i] = np.min(permute[non_zero_rows, s])

# Hasta aqui bien caso base, tal vez medio sospechoso que numeros de hash sean tan grandes?

unique_columns, counts = np.unique(signature_matrix, axis=1, return_counts=True)
repeated_columns = unique_columns[:, counts > 1]
indices_repetidos = []
if repeated_columns.size > 0:
    for column in repeated_columns.T:
        indices = np.where((signature_matrix == column[:, None]).all(axis=0))[0]
        indices_repetidos.append(indices)
else:
    print("No hay indices repetidos")