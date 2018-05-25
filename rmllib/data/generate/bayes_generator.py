import numpy.random as rnd
import pandas

from .matched_edge_generator import matched_edge_generator
from ..base import Dataset

class BayesSampleDataset(Dataset):

    def __init__(self, n_rows=100, n_features=2, positive_prior=.5, generator=matched_edge_generator, **kwargs):
        super().__init__(**kwargs)
        labels = rnd.random(n_rows) < positive_prior
        thresholds = rnd.random((2, n_features))

        features = rnd.random((n_rows, n_features))
        features[labels, :] = features[labels, :] < thresholds[1, :]
        features[~labels, :] = features[~labels, :] < thresholds[0, :]

        self.labels = pandas.DataFrame(labels.astype(int), columns=['Y'])
        self.features = pandas.DataFrame(features.astype(int), columns=list(map(lambda x: 'Feat_' + str(x), range(n_features))))

        # Simple correlation for edges       
        self.edges = generator(self.labels, **kwargs)

        return