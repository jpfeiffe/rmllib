'''
    Author: Joel Pfeiffer
    Email: jpfeiffe@gmail.com
    Date Created: 5/22/2018
    Python Version: 3.6
'''
import numpy as np
import numpy.random as rnd
import pandas

def matched_edge_generator(labels, mu_match=.55, mu_nomatch=.48, std=.3, sparsity=95, symmetric=True, remove_loops=True):
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
    labels = labels.copy()
    labels[labels.Y == 0] = -1
    matrix = (labels.T.values * labels.values).astype(float)
    matrix[matrix > 0] = mu_match
    matrix[matrix < 0] = mu_nomatch

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

    # Try and keep the sparse DF representation throughout
    return pandas.DataFrame(matrix, index=labels.index)
