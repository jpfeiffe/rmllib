'''
    Author: Joel Pfeiffer
    Email: jpfeiffe@gmail.com
    Date Created: 5/22/2018
    Python Version: 3.6
'''
import sklearn.datasets
import pandas

from .base import Dataset
from ..generate import matched_edge_generator

class BostonMedians(Dataset):
    '''
    Simple boston dataset with randomized edge data
    '''
    def __init__(self, subfeatures=None, **kwargs):
        '''
        Builds our dataset by
        (a) loading sklearn Boston dataset
        (b) binarizing it via the median of the feature values
        (c) generating random edges

        :subfeatures: Subsets of features available in the boston dataset.  Primarily for simulating weakened feature signals.
        :kwargs: Arguments for matched_edge_generator
        '''
        super().__init__(kwargs)
        boston = sklearn.datasets.load_boston()
        self.features = pandas.DataFrame(boston['data'], columns=boston['feature_names'])
        self.labels = pandas.DataFrame(boston['target'], columns=['Y'])

        # Booleanize feat by medians
        self.features = self.features - self.features.median(axis=0)
        self.features[self.features < 0] = 0
        self.features[self.features > 0] = 1

        # Booleanize target by medians
        self.labels = self.labels - self.labels.median(axis=0)
        self.labels[self.labels < 0] = 0
        self.labels[self.labels > 0] = 1

        self.labels = self.labels.astype(int)
        self.features = self.features.astype(int)

        # Simple correlation for edges       
        self.edges = matched_edge_generator(self.labels, **kwargs)

        if subfeatures:
            self.features = self.features[subfeatures]

        return


