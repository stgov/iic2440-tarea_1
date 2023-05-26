import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import save_npz, csr_matrix

def shingles(k, words: np.ndarray) -> list:
    shingles = []
    for i in range(0, len(words)):
        shingles.append(' '.join(words[i:i+k]))
    return list(set(shingles))

def caracteristica(k:int=2, sample_size:int=None) -> csr_matrix:
    tweets = pd.read_parquet('../Data/tweets.parquet')
    if sample_size:
        tweets = tweets.sample(sample_size)
    tweets['shingles'] = tweets['text'].str.split().apply(lambda x: np.array(x)).apply(lambda x: shingles(k, x))
    shings = tweets['shingles'].to_numpy()
    mlb = MultiLabelBinarizer(sparse_output=True)
    caracteristica = mlb.fit_transform(shings)

    return caracteristica, tweets