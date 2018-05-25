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
        self.mask = pandas.DataFrame(rnd.random_sample(self.labels.shape) < labeled_frac,\
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
        train.labels.Y[train.mask.Unlabeled] = -1
        return train
