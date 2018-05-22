import numpy as np
import numpy.random as rnd
import pandas

def CorrelatedEdgeGenerator(x, sparsity=95, noise=.1):
    '''
    Builds randomized edges based on a feature matrix

    :param X: dataframe of features
    :param sparsity: fraction of edge values = 0.  Multiply by 100 (np.percentile)
    :param noise: how much to pertub the edges
    '''
    matrix = np.dot(x.as_matrix(), x.T.as_matrix())
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