import sklearn.datasets
import numpy as np
import numpy.random as rnd
import pandas
from .Generators import ManualEdgeGenerator

class Dataset():
    def __init__(self):
        self.X = None
        self.Y = None
        self.E = None
        self.Mask = None
        return

    def labelmask(self, labeled_frac=.5):
        '''
        masks the labeled data to create an unlabeled set

        :param frac: fraction of unlabeled dataset
        '''
        self.Mask = pandas.DataFrame(rnd.random_sample(self.Y.shape) < labeled_frac, columns=['Labeled'])
        self.Mask['Unlabeled'] = ~self.Mask.Labeled

class BostonMedians(Dataset):
    '''
    Simple boston dataset with randomized edge data
    '''
    def __init__(self, subfeatures=None):
        super()
        d = sklearn.datasets.load_boston()
        self.X = pandas.DataFrame(d['data'], columns=d['feature_names'])
        self.Y = pandas.DataFrame(d['target'], columns=['Y'])

        # Booleanize feat by medians
        self.X = self.X - self.X.median(axis=0)
        self.X[self.X < 0] = 0
        self.X[self.X > 0] = 1

        # Booleanize target by medians
        self.Y = self.Y - self.Y.median(axis=0)
        self.Y[self.Y < 0] = 0
        self.Y[self.Y > 0] = 1

        self.Y = self.Y.astype(int)
        self.X = self.X.astype(int)

        conditionals = {}
        conditionals[True] = .51
        conditionals[False] = .48

        # Simple correlation for edges       
        self.E = ManualEdgeGenerator(self.Y, conditionals)

        if subfeatures:
            self.X = self.X[subfeatures]
        return
