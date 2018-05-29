'''
    Joel Pfeiffer
    jpfeiffe@gmail.com
'''
import numpy as np
import numpy.random as rnd
import pandas
from scipy.sparse import csr_matrix, bsr_matrix

def matched_edge_generator(labels, mu_match=.52, mu_nomatch=.48, std=.3, sparsity=95, symmetric=True, remove_loops=True, sparse=True, **kwargs):
    '''
    Generates a network given a set of labels.  If the labels are equal, samples from a normal
    with one std, and if they aren't equal it samples from another.  Sparsity sets the cutoff point.

    :y: Labels to utilize
    :mu_match: Normal mean for matching labels
    :mu_nomatch: Normal mean for non-matching labels
    :std: standard deviation of the normals
    :sparsity: Fraction of non-zero edges
    :symmetric: Is the returned matrix symmetric
    :remove_loops: Take out self loops (generally good)
    '''

    # Build matrix of mu values
    labels = labels.Y.copy()
    matrix = labels.values.dot(labels.T.values).astype(float)
    matrix[matrix > .5] = mu_match
    matrix[matrix < .5] = mu_nomatch

    # Sample with given std
    matrix += rnd.normal(scale=std, size=matrix.shape)

    if symmetric:
        matrix = np.minimum(matrix, matrix.T)

    if remove_loops:
        np.fill_diagonal(matrix, 0)

    # Apply cutoffs
    cutoffs = np.percentile(matrix, sparsity, axis=0)
    matrix[matrix >= cutoffs] = 1
    matrix[matrix < cutoffs] = 0

    if sparse:
        matrix = csr_matrix(matrix.astype(np.int64))
        return matrix

    return matrix
