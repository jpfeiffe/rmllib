'''
    Author: Joel Pfeiffer
    Email: jpfeiffe@gmail.com
    Date Created: 5/22/2018
    Python Version: 3.6
'''
import sklearn.datasets
import pandas
import numpy as np

from ..base import Dataset
from ..base import class_transform_to_dataframe
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
        super().__init__(**kwargs)
        boston = sklearn.datasets.load_boston()
        init_features = pandas.DataFrame(boston['data'], columns=boston['feature_names'])

        if subfeatures:
            init_features = init_features[subfeatures]

        init_labels = pandas.DataFrame(boston['target'], columns=['Y'])
        init_labels = init_labels - init_labels.median(axis=0)
        init_labels.Y = np.where(init_labels.Y > 0, 1, 0).astype(int)


        # Booleanize feat by medians
        init_features = init_features - init_features.median(axis=0)
        init_features[init_features < 0] = 0
        init_features[init_features > 0] = 1
        init_features = init_features.astype(int)

        # Create dataframe
        self.labels = class_transform_to_dataframe(init_labels.Y.values, islabel=True)
        self.features = class_transform_to_dataframe(init_features.values, islabel=False, classes=init_features.columns.values)

        # Simple correlation for edges       
        self.edges = matched_edge_generator(self.labels, **kwargs)

        return


