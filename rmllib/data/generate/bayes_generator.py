import numpy.random as rnd
import pandas

from .matched_edge_generator import matched_edge_generator
from ..base import Dataset
from ..base import class_transform_to_dataframe

class BayesSampleDataset(Dataset):

    def __init__(self, n_rows=100, n_features=2, positive_prior=.5, generator=matched_edge_generator, **kwargs):
        super().__init__(**kwargs)
        labels = rnd.random(n_rows) < positive_prior
        thresholds = rnd.random((2, n_features))

        features = rnd.random((n_rows, n_features))
        features[labels, :] = features[labels, :] < thresholds[1, :]
        features[~labels, :] = features[~labels, :] < thresholds[0, :]

        self.features = class_transform_to_dataframe(features.astype(int), islabel=False, sparse=True)
        self.labels = class_transform_to_dataframe(labels.astype(int), islabel=True)

        # Simple correlation for edges       
        self.edges = generator(self.labels, **kwargs)

        return