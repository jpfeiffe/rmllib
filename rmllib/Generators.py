import numpy as np
import numpy.random as rnd
import pandas


def ManualEdgeGenerator(y, conditionals, sparsity=85, noise=.3):
    y = y.copy()
    y[y.Y==0] = -1
    matrix = y.T.values * y.values
    matrix[matrix < 0] = 0
    matrix = matrix.astype(float)
    matrix[matrix == 1] = conditionals[True]
    matrix[matrix == 0] = conditionals[False]
    matrix += rnd.normal(scale=noise, size=matrix.shape)
    matrix = np.minimum(matrix, matrix.T)
    np.fill_diagonal(matrix, 0)
    cutoffs = np.percentile(matrix, sparsity, axis=0)

    matrix[matrix >= cutoffs] = 1
    matrix[matrix < cutoffs] = 0

    E = pandas.DataFrame(matrix, index=y.index)
    return E

def CorrelatedEdgeGenerator(x, sparsity=50, noise=.01):
    '''
    Builds randomized edges based on a feature matrix

    :param X: dataframe of features
    :param sparsity: fraction of edge values = 0.  Multiply by 100 (np.percentile)
    :param noise: how much to pertub the edges
    '''
    matrix = np.dot(x.values, x.T.values)
    np.fill_diagonal(matrix, 0)
    matrix = matrix / matrix.max()
    matrix = matrix + rnd.normal(loc=0, scale=noise, size=matrix.shape)
    cutoffs = np.percentile(matrix, sparsity, axis=0)
    matrix[matrix >= cutoffs] = 1
    matrix[matrix < cutoffs] = 0
    matrix = np.maximum(matrix, matrix.T)
    np.fill_diagonal(matrix, 0)

    E = pandas.DataFrame(matrix, index=x.index)
    return E