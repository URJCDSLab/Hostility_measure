# Generation of the Normal bivariate artificial data

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt



def normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2):
    mn0 = multivariate_normal(mean=mu0, cov=sigma0)
    X_neg = mn0.rvs(size=n0, random_state=seed1)

    mn1 = multivariate_normal(mean=mu1, cov=sigma1)
    X_pos = mn1.rvs(size=n1, random_state=seed2)

    X = np.vstack((X_neg, X_pos))
    y = np.array([0] * len(X_neg) + [1] * len(X_pos))

    data = pd.DataFrame(X,columns=['x','y'])

    # Plot
    # For labels
    labels = list(data.index)
    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)
    plt.scatter(data.iloc[idx_0].x, data.iloc[idx_0].y, s=70, c='k', marker=".", label='negative')
    plt.scatter(data.iloc[idx_1].x, data.iloc[idx_1].y, s=70, c='c', marker="+", label='positive')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return X, y




# Parameters
seed1 = 1
seed2 = 2
n0 = 3000
n1 = 3000

# Dataset 1
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [3, 3]
sigma1 = [[1, 0], [0, 1]]

X, y = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)


# Dataset 2
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [2, 2]
sigma1 = [[1, 0], [0, 1]]

X, y  = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)



# Dataset 3
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [1, 1]
sigma1 = [[1, 0], [0, 1]]

X, y = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)


# Dataset 4
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [0, 0]
sigma1 = [[1, 0], [0, 1]]

X, y = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)



# dataset 5
mu0 = [0, 0]
sigma0 = [[25, 0], [0, 25]]
mu1 = [0, 0]
sigma1 = [[1, 0], [0, 1]]

X, y = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)


# dataset 6
mu0 = [0, 0]
sigma0 = [[80, 0], [0, 5]]
mu1 = [-10, 0]
sigma1 = [[7, 0], [0, 1]]

X, y = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)


# dataset 7
mu0 = [0, 0]
sigma0 = [[80, 0], [0, 5]]
mu1 = [9, 0]
sigma1 = [[1, 2], [2, 5]]

X, y = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)

# dataset 8
mu0 = [0, 0]
sigma0 = [[80, 0], [0, 5]]
mu1 = [20, 3]
sigma1 = [[9, 2], [2, 9]]

X, y = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)

# dataset 9
mu0 = [0, 0]
sigma0 = [[80, 0], [0, 5]]
mu1 = [8, 0]
sigma1 = [[8, 0], [0, 8]]

X, y = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)




