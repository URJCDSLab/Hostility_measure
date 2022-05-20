# Generation of the Normal multivariate artificial data

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def normal_generator3(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, seed0, seed1, seed2):
    mn0 = multivariate_normal(mean=mu0, cov=sigma0)
    X0 = mn0.rvs(size=n0, random_state=seed0)

    mn1 = multivariate_normal(mean=mu1, cov=sigma1)
    X1 = mn1.rvs(size=n1, random_state=seed1)

    mn2 = multivariate_normal(mean=mu2, cov=sigma2)
    X2 = mn2.rvs(size=n2, random_state=seed2)

    X = np.vstack((X0, X1, X2))
    y = np.array([0] * len(X0) + [1] * len(X1) + [2] * len(X2))

    data = pd.DataFrame(X, columns=['x', 'y'])

    # Plot

    labels = list(data.index)
    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)
    idx_2 = np.where(y == 2)
    plt.scatter(data.iloc[idx_0].x, data.iloc[idx_0].y, s=70, c='k', marker=".", label='0')
    plt.scatter(data.iloc[idx_1].x, data.iloc[idx_1].y, s=70, c='c', marker="+", label='1')
    plt.scatter(data.iloc[idx_2].x, data.iloc[idx_2].y, s=70, c='purple', marker="*", label='2')

    plt.xticks([])
    plt.yticks([])
    plt.show()

    return X, y


seed0 = 1
seed1 = 2
seed2 = 3
n0 = 3000
n1 = 3000
n2 = 3000


## Dataset multiclass 1
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [3, 3]
sigma1 = [[1, 0], [0, 1]]
mu2 = [2, -1]
sigma2 = [[3, 1], [1, 1]]

X, y = normal_generator3(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, seed0, seed1, seed2)



## Dataset multiclass 2
mu0 = [0, 0]
sigma0 = [[80, 0], [0, 5]]
mu1 = [9, 0]
sigma1 = [[1, 2], [2, 5]]
mu2 = [-6, -4]
sigma2 = [[5, 0], [0, 1]]

X, y = normal_generator3(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, seed0, seed1, seed2)
