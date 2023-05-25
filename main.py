from scipy.sparse import coo_matrix
import numpy as np
from scipy.spatial import distance
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer
from datasketch import MinHash
import random

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
        shingles.append(' '.join(words[i:i+k]))
    return list(set(shingles))

def jaccard_difference(set1, set2):
    intersection = np.intersect1d(set1, set2)
    union = np.union1d(set1, set2)
    if len(union) == 0:
        return 0.0  # or any other default value
    return 1.0 - len(intersection) / len(union)


# Abrimos datos y los limpiamos

tweets = pd.read_csv(
    '/home/bcm/Desktop/PUC/Procesamiento de datos masivos/T1_Grupo/iic2440-tarea_1/Datos/tweets_2022_abril_junio.csv',
    usecols=['id', 'screen_name', 'text'],
    index_col='id',
    dtype={'screen_name': str, 'text': str},
    nrows=1000
    )
tweets = tweets.drop_duplicates(subset='text', keep=False)
tweets = tweets.dropna()
tweets['text'] = tweets.loc[:, 'text'].apply(limpiar_texto)


# Creamos Shingles

k = 2 # Shingles size
tweets['shingles'] = tweets['text'].str.split().apply(lambda x: np.array(x)).apply(lambda x: shingles(k, x))
shings = tweets['shingles'].to_numpy()
u_shingles = np.unique(np.concatenate(shings)) # Shingles unicos en una unica lista

# Caracteristica no me daba una sparse buena, me decia solamente que en las filas 0 habian datos != 0, pero no para el resto de las filas.

# mlb = MultiLabelBinarizer(sparse_output=True)
# caracteristica = mlb.fit_transform(shings)


# Creamos caracteristica y la pasamos a formato sparse



# Conseguimos coordenadas de filas y columnas en donde shingle esta en tweet.

rows = []
cols = []
data = []

for i, tweet in enumerate(tweets['text']):
    for j, shingle in enumerate(u_shingles):
        if shingle in tweet:
            rows.append(i)
            cols.append(j)
            data.append(1)

sparse_matrix = coo_matrix((data, (rows, cols)), shape=(len(tweets['text']), len(u_shingles))).T


# Creamos minhashes

minhashes = []
num_hashes = 10
num_perm = sparse_matrix.shape[0]


for i in range(num_hashes):
    seed = random.randint(0, 2**32 - 1)
    minhash = MinHash(num_perm=num_perm, seed=seed)

    for shingle in u_shingles:
        minhash.update(shingle.encode('utf-8'))
    mapped_values = [int(value) % sparse_matrix.shape[0] for value in minhash.hashvalues]
    minhash.hashvalues = mapped_values
    minhashes.append(minhash)

# Creamos permute, que es la matriz con los minhashes y los valores randoms permutados

permute = np.zeros((len(minhashes), len(minhashes[0].hashvalues)))

for i, minhash in enumerate(minhashes):
    # Get the hash values from the MinHash object
    hash_values = minhash.hashvalues

    # Assign the hash values to the corresponding row in the signature matrix
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

unique_columns, counts = np.unique(signature_matrix, axis=1, return_counts=True)
repeated_columns = unique_columns[:, counts > 1]
indices_repetidos = []
if repeated_columns.size > 0:
    for column in repeated_columns.T:
        indices = np.where((signature_matrix == column[:, None]).all(axis=0))[0]
        indices_repetidos.append(indices)
        print(tuple(indices))
else:
    print("No hay indices repetidos")