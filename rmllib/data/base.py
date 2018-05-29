'''
    Author: Joel Pfeiffer
    Email: jpfeiffe@gmail.com
    Date Created: 5/22/2018
    Python Version: 3.6
'''
import copy
import pandas
import numpy.random as rnd
import numpy as np
import itertools
from scipy.sparse import lil_matrix


def class_transform_to_dataframe(init_mat, islabel=False, classes=None, sparse=False):
    if islabel:
        if len(init_mat.shape) == 1:
            init_mat = init_mat[:, np.newaxis]
        assert(init_mat.shape[1] == 1)
        classes = ['Y']
    elif classes is not None:
        assert(len(classes) == init_mat.shape[1])
    else:
        classes = list(map(lambda x: 'Feature_' + str(x), range(init_mat.shape[1])))

    classes = sorted(classes)
    column_index = []

    for i,class_name in enumerate(classes):
        column_index += itertools.product([class_name], sorted(np.unique(init_mat[:, i])))

    df = None

    if not sparse:
        # Construct dataframe
        df = pandas.DataFrame(columns=pandas.MultiIndex.from_tuples(column_index))

        for i,class_name in enumerate(classes):
            for class_val in sorted(np.unique(init_mat[:, i])):
                df.loc[:, (class_name, class_val)] = init_mat[:, i] == class_val
    else:
        mat = lil_matrix((init_mat.shape[0], len(column_index)))
        ind = 0
        for i,class_name in enumerate(classes):
            for class_val in sorted(np.unique(init_mat[:, i])):
                mat[:, ind] = (init_mat[:, i] == class_val)[:, np.newaxis]
                ind += 1
        df = pandas.SparseDataFrame(mat, columns=pandas.MultiIndex.from_tuples(column_index))

    df = df.astype(float)

    return df



class Dataset:
    '''
    Basic functions every dataset should have.
    '''
    
    def __init__(self, name='dataset', labels=None, features=None, edges=None, mask=None, **kwargs):
        '''
        Each dataset is required to have features X, labels Y, edges E and a Mask that 
        corresponds to the labeled set
        '''
        self.name = name
        self.labels = labels
        self.features = features
        self.edges = edges
        self.mask = mask
        return

    def is_sparse_features(self):
        '''
        checks whether the features are in a sparse representation
        '''
        return type(self.features) == pandas.SparseDataFrame

    def copy(self):
        '''
        Creates deep copy of the object (prevents datasets from interfering with each other)
        '''
        return copy.deepcopy(self)

    def node_sample_mask(self, labeled_frac=.5):
        '''
        masks the labeled data by doing random node sampling

        :param frac: fraction of unlabeled dataset
        '''
        self.mask = pandas.DataFrame(rnd.random_sample(len(self.labels)) < labeled_frac,\
                                     columns=['Labeled'])
        self.mask['Unlabeled'] = ~self.mask.Labeled

        return self

    def create_training(self, labeled_frac=None):
        '''
        Creates a training dataset by masking the unlabeled values to -1
        '''
        if labeled_frac:
            self.node_sample_mask(labeled_frac)
        elif self.mask is None:
            raise Exception("Labeled Mask not created")

        train = self.copy()
        train.labels.Y[train.mask.Unlabeled] = np.nan
        return train
