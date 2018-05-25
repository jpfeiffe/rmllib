'''
    Joel Pfeiffer
    jpfeiffe@gmail.com
'''
import numpy as np
import numpy.random as rnd
import pandas
from scipy.sparse import csr_matrix
import time

def edge_rejection_generator(labels, negative_acception_probability=.75, density=.05, **kwargs):
    '''
    Generates a network given a set of labels.  If the labels are equal, samples from a normal
    with one std, and if they aren't equal it samples from another.  density sets the cutoff point.

    :y: Labels to utilize
    :mu_match: Normal mean for matching labels
    :mu_nomatch: Normal mean for non-matching labels
    :std: standard deviation of the normals
    :density: Number of nonzero edges
    :remove_loops: Take out self loops (generally good)
    '''

    # Probability a proposed sample is rejected.
    fraction_positive = labels.Y.mean()
    probability_nomatch = 2*fraction_positive*(1-fraction_positive) / (fraction_positive*fraction_positive + (1-fraction_positive)*(1-fraction_positive) + 2*fraction_positive*(1-fraction_positive))
    rejection_probability = probability_nomatch * (1-negative_acception_probability)

    # Compute how many draws we need such that in expectation we're about right for our density
    desired_edges = density*len(labels)*len(labels)/2
    samples_needed = int(1 + desired_edges / (1-rejection_probability))

    # Propose some edges
    proposed_edges = rnd.randint(len(labels), size=(samples_needed,2))

    # Find the matched ones (always accept), then compute reject probabilities
    matched = labels['Y'].values[proposed_edges[:, 0]] == labels['Y'].values[proposed_edges[:, 1]]
    accept = rnd.random(len(proposed_edges)) < negative_acception_probability

    # These are the winners
    keep = np.logical_or(matched, accept)

    # Symmetric, drop duplicates
    proposed_edges = proposed_edges[keep, :]
    reversed_edges = np.array((proposed_edges[:, 1], proposed_edges[:, 0])).T
    proposed_edges = np.unique(np.vstack((proposed_edges, reversed_edges)), axis=0)

    # Create the sparse representation
    matrix = csr_matrix((np.ones(len(proposed_edges)), (proposed_edges[:,0], proposed_edges[:,1])), shape=(len(labels), len(labels)), dtype=np.int8)
    
    print('Average Degree:', matrix.sum(axis=1).mean())

    # Small correction hack
    zeroed = np.where(matrix.sum(axis=1)==0)[0]
    if len(zeroed) > 0:
        print('Warning; Performing correction hack on ', len(zeroed), 'instances')
        offset = rnd.randint(low=1, high=len(labels))
        paired = (zeroed + offset) % len(labels)

        matrix[zeroed, paired] = 1
        matrix[paired, zeroed] = 1

    return matrix
