import pandas
import numpy as np
import scipy.special
import scipy.stats


def ExpectationMaximization(local_model, **kwargs):
    class ExpectationMaximization(local_model):
        def __init__(self, learn_iter=10, **kwargs):
            super().__init__(**kwargs)
            self.learn_iter=learn_iter

        def fit(self, data):
            super().set_learn_method('r_iid')
            super().fit(data)
            self.set_learn_method('r_joint')

            for _ in range(self.learn_iter):
                probabilities = super().predict_proba(data)
                data.labels.Y.loc[data.mask.Unlabeled] = probabilities[:, 1]
                super().fit(data, rel_update_only=True)

            return self

    return ExpectationMaximization
