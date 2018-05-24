import time


def VariationalInference(local_model):
    class VariationalInference(local_model):
        '''
        Variational inference will augment the relational training
        '''
        def __init__(self, infer_iter=10, **kwargs):
            '''
            Initialize both the base model and our inference step

            :param infer_iter: Number of iterations of VI
            :param kwargs: Arguments for local model.  Setting infer_method is useless as it will be overwritten as
                needed by this method
            '''
            super().__init__(**kwargs)
            self.infer_iter=infer_iter

        def predict_proba(self, data, rel_update_only=False):
            '''
            Make predictions

            :param data: Network dataset to make predictions on
            '''
            if not rel_update_only:
                self.set_infer_method('r_iid')
                self.probabilities = super().predict_proba(data)
            self.set_infer_method('r_joint')

            for _ in range(self.infer_iter):
                time_start = time.time()
                data.labels.Y.loc[data.mask.Unlabeled] = self.probabilities[:, 1]
                self.probabilities = super().predict_proba(data, rel_update_only=True)

            return self.probabilities

    return VariationalInference
