import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats

def gevcTrain(train):
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(train)
    n, _ = train.shape
    distances = np.array([neigh.kneighbors([train[i]])[0][0][1] for i in range(0, n)])
    distances = distances[distances!=0]
    #stats.exponweib.fit(data, 1, 1, scale=0.2, loc=0)
    params = stats.exponweib.fit(distances, 1, 1, scale=0.2, loc=0)
    return params

def gevcTest(train, test, pre, prob=False, alpha=0.001):
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(train)
    n, _ = test.shape
    distances = np.array([neigh.kneighbors([test[i]])[0][0][1] for i in range(0, n)])
    out = 1 - stats.exponweib.pdf(distances, *pre)
    if prob==False:
        # 1 if known, 0 if unknown
        out = [1 if k > alpha else 0 for k in out]
    return out
