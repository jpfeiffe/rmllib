'''
Implements the relational naive bayes model
'''
import pandas
import numpy as np
import scipy
import time

from .base import LocalModel

class RelationalNaiveBayes(LocalModel):
    '''
    Basic RNB implementation.  Can do iid learning to collective inference.
    '''
    def __init__(self, **kwargs):
        '''
        Sets up the NB specific parameters
        '''
        super().__init__(**kwargs)
        self.class_log_prior_ = None
        self.feature_log_prob_ = None


    def predict_proba(self, data, rel_update_only=False):
        '''
        Make predictions

        :param data: Network dataset to make predictions on
        '''
        un_features = data.features.iloc[data.mask.Unlabeled.nonzero()[0], :].values

        if not rel_update_only:
            logneg = (1-un_features).dot(self.feature_log_prob_x_.loc[(0,0),:].values)\
                    + un_features.dot(self.feature_log_prob_x_.loc[(0,1),:].values)
            logpos = (1-un_features).dot(self.feature_log_prob_x_.loc[(1,0),:].values)\
                    + un_features.dot(self.feature_log_prob_x_.loc[(1,1),:].values)
            base_logits = np.stack((self.class_log_prior_[0] + logneg, self.class_log_prior_[1] + logpos), axis=1)
            self.base_logits = base_logits.copy()

        else:
            base_logits = self.base_logits.copy()

        # IID predictions
        if self.infer_method == 'iid':
            base_conditionals = np.exp(base_logits)

        # Relational IID Predictions
        elif self.infer_method == 'r_iid':
            all_to_unlabeled_edges = data.edges[data.mask.Unlabeled.nonzero()[0], :]

            pos = self.unlabeled_confidence * all_to_unlabeled_edges.dot(data.mask.Labeled*data.labels.Y.values)
            neg = self.unlabeled_confidence * all_to_unlabeled_edges.dot(data.mask.Labeled*(1-data.labels.Y.values))

            y_logpos = self.feature_log_prob_y_.loc[(1,1)] * pos +\
                        self.feature_log_prob_y_.loc[(1,0)] * neg
            y_logneg = self.feature_log_prob_y_.loc[(0,1)] * pos +\
                        self.feature_log_prob_y_.loc[(0,0)] * neg

            base_logits += np.stack((y_logneg, y_logpos), axis=1)
            base_conditionals = np.exp(base_logits)

        # Relational Join Predictions
        elif self.infer_method == 'r_joint' or self.infer_method == 'r_twohop':
            all_to_unlabeled_edges = data.edges[data.mask.Unlabeled.nonzero()[0], :]

            pos = self.unlabeled_confidence * all_to_unlabeled_edges.dot(data.labels.Y.values)
            neg = self.unlabeled_confidence * all_to_unlabeled_edges.dot((1-data.labels.Y.values))

            y_logpos = self.feature_log_prob_y_.loc[(1,1)] * pos +\
                        self.feature_log_prob_y_.loc[(1,0)] * neg
            y_logneg = self.feature_log_prob_y_.loc[(0,1)] * pos +\
                        self.feature_log_prob_y_.loc[(0,0)] * neg

            base_logits += np.stack((y_logneg, y_logpos), axis=1)
            base_conditionals = np.exp(base_logits)



            # rel = pandas.DataFrame(\
            #                         ,\
            #                         index=un_features.T.index, columns=['PosN', 'NegN']).rename_axis("X")

            # y_logpos = rel['PosN']*self.feature_log_prob_['Y_N'].loc[(1, 1)]\
            #                 + rel['NegN']*self.feature_log_prob_['Y_N'].loc[(1, 0)]
            # y_logneg = rel['PosN']*self.feature_log_prob_['Y_N'].loc[(0, 1)]\
            #                 + rel['NegN']*self.feature_log_prob_['Y_N'].loc[(0, 0)]

            # base_logits += np.stack((y_logneg, y_logpos), axis=1)
            # base_conditionals = np.exp(base_logits)
            # if self.infer_method == 'r_twohop':
            #     all_to_all_edges = data.edges.copy().div(data.edges.sum(axis=1), axis=0) * self.twohop_confidence
            #     print(all_to_all_edges.sum(axis=1))
            #     exit()
                

        # Relational Joint Predictions
        confidence = base_conditionals.sum(axis=1)[:, np.newaxis]
        predictions = base_conditionals / confidence

        if self.calibrate:
            logits = scipy.special.logit(predictions[:, 1])
            logits -= np.percentile(logits, data.labels.loc[data.mask.Labeled].mean()*100)
            predictions[:, 1] = scipy.special.expit(logits)
            predictions[:, 0] = 1 - predictions[:, 1]

        return predictions

    def predict(self, data):
        '''
        Returns the predicted labels on the dataset

        :param data: Network dataset to make predictions on
        '''
        return np.argmax(self.predict_proba(data), axis=1)

    def fit(self, data, rel_update_only=False):
        # Only use the labeled data to fit
        lab_features = data.features.iloc[data.mask.Labeled.nonzero()[0], :]
        lab_labels = data.labels.iloc[data.mask.Labeled.nonzero()[0], :]

        # Prior distributions
        self.class_log_prior_ = np.log([1-lab_labels.Y.mean(), lab_labels.Y.mean()])

        if not rel_update_only:
            # Compute X log conditional values
            log_posx = np.log(lab_labels.join(lab_features).groupby('Y').mean())
            log_posx['X'] = 1
            log_posx = log_posx.reset_index().set_index(['Y', 'X'])
            log_negx = np.log(1-lab_labels.join(lab_features).groupby('Y').mean())
            log_negx['X'] = 0
            log_negx = log_negx.reset_index().set_index(['Y', 'X'])

            self.feature_log_prob_x_ = pandas.concat([log_posx, log_negx])

        if self.learn_method == 'r_iid':
            lab_to_all_edges = data.edges[data.mask.Labeled.nonzero()[0], :]

            # Create basic Y | Y_N counts
            neighbor_counts = pandas.DataFrame(0, index=self.feature_log_prob_x_.index, columns=['Y_N'])

            neighbor_counts.loc[(1,1), 'Y_N'] = np.sum(lab_to_all_edges.dot(data.mask.Labeled*data.labels.Y.values) * (data.labels.Y.values[data.mask.Labeled]))
            neighbor_counts.loc[(1,0), 'Y_N'] = np.sum(lab_to_all_edges.dot(data.mask.Labeled*(1-data.labels.Y.values)) * (data.labels.Y[data.mask.Labeled]))

            neighbor_counts.loc[(0,1), 'Y_N'] = np.sum(lab_to_all_edges.dot(data.mask.Labeled*data.labels.Y.values) * (1-data.labels.Y[data.mask.Labeled]))
            neighbor_counts.loc[(0,0), 'Y_N'] = np.sum(lab_to_all_edges.dot(data.mask.Labeled*(1-data.labels.Y.values)) * (1-data.labels.Y[data.mask.Labeled]))

            neighbor_counts['Total'] = neighbor_counts.groupby(level=0).transform('sum')
            neighbor_counts['Y_N'] /= neighbor_counts['Total']
            self.feature_log_prob_y_ = np.log(neighbor_counts['Y_N'])

        elif self.learn_method == 'r_joint' or self.learn_method == 'r_twohop':
            lab_to_all_edges = data.edges[data.mask.Labeled.nonzero()[0], :]

            # Create basic Y | Y_N counts
            neighbor_counts = pandas.DataFrame(0, index=self.feature_log_prob_x_.index, columns=['Y_N'])

            neighbor_counts.loc[(1,1), 'Y_N'] = np.sum(lab_to_all_edges.dot(data.labels.Y.values) * (data.labels.Y.values[data.mask.Labeled]))
            neighbor_counts.loc[(1,0), 'Y_N'] = np.sum(lab_to_all_edges.dot((1-data.labels.Y.values)) * (data.labels.Y[data.mask.Labeled]))

            neighbor_counts.loc[(0,1), 'Y_N'] = np.sum(lab_to_all_edges.dot(data.labels.Y.values) * (1-data.labels.Y[data.mask.Labeled]))
            neighbor_counts.loc[(0,0), 'Y_N'] = np.sum(lab_to_all_edges.dot((1-data.labels.Y.values)) * (1-data.labels.Y[data.mask.Labeled]))

            neighbor_counts['Total'] = neighbor_counts.groupby(level=0).transform('sum')
            neighbor_counts['Y_N'] /= neighbor_counts['Total']
            self.feature_log_prob_y_ = np.log(neighbor_counts['Y_N'])
            
        return self
