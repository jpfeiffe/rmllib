import pandas
import numpy as np
import scipy.special
import scipy.stats

class RNB:
    def __init__(self, learnmethod='iid', infermethod='vi', calibrate=True, inferenceiters=10, unlabeled_confidence=1):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.learnmethod = learnmethod
        self.infermethod = infermethod
        self.inferenceiters = inferenceiters
        self.unlabeled_confidence = unlabeled_confidence
        self.calibrate = calibrate

    def predict(self, data):
        unlabeledX = data.X.loc[data.Mask.Unlabeled, :].T

        # Other features taken care of...
        logneg = (unlabeledX.where(unlabeledX != 0, self.feature_log_prob_.loc[0,:].T[0], axis=0) + unlabeledX.where(unlabeledX != 1, self.feature_log_prob_.loc[0,:].T[1], axis=0)).T.sum(axis=1)
        logpos = (unlabeledX.where(unlabeledX != 0, self.feature_log_prob_.loc[1,:].T[0], axis=0) + unlabeledX.where(unlabeledX != 1, self.feature_log_prob_.loc[1,:].T[1], axis=0)).T.sum(axis=1)

        base_logneg = self.class_log_prior_[0] + logneg 
        base_logpos = self.class_log_prior_[1] + logpos

        # Relational estimates
        if self.infermethod == 'iid':
            neg = np.exp(base_logneg)
            pos = np.exp(base_logpos)

        elif self.infermethod == 'riid' or self.infermethod == 'vi':
            labeledY = data.Y.loc[data.Mask.Labeled, :]
            labelToUnlabelE = data.E.loc[data.Mask.Unlabeled, data.Mask.Labeled]
            rel = pandas.DataFrame(\
                                    np.hstack((np.dot(labelToUnlabelE.values, labeledY.values), np.dot(labelToUnlabelE.values, 1-labeledY.values))),\
                                    index=unlabeledX.T.index, columns=['PosN', 'NegN']).rename_axis("X")

            y_logpos = rel['PosN']*self.feature_log_prob_['Y_N'].loc[(1, 1)]\
                            + rel['NegN']*self.feature_log_prob_['Y_N'].loc[(1, 0)]
            y_logneg = rel['PosN']*self.feature_log_prob_['Y_N'].loc[(0, 1)]\
                            + rel['NegN']*self.feature_log_prob_['Y_N'].loc[(0, 0)]

            neg = np.exp(base_logneg + y_logneg)
            pos = np.exp(base_logpos + y_logpos)

        predictions = pos / (neg+pos)
        if self.calibrate:
            logits = scipy.special.logit(predictions)
            average = data.Y.loc[data.Mask.Labeled].mean()
            logits -= np.percentile(logits, average*100)
            predictions = scipy.special.expit(logits)

        # Joint inference
        if self.infermethod == 'vi':
            ECopy = data.E.copy()
            allToUnlabelE = ECopy.loc[data.Mask.Unlabeled, :]
            allToUnlabelE.loc[:, data.Mask.Unlabeled] *= self.unlabeled_confidence
            # Current Predictions     
            for i in range(self.inferenceiters):
                Q = data.Y.copy()
                Q.loc[data.Mask.Unlabeled, :] = predictions.values.reshape(len(predictions),1)

                rel = pandas.DataFrame(np.hstack((\
                                        np.dot(allToUnlabelE.values, Q.values),\
                                        np.dot(allToUnlabelE.values, 1-Q.values)\
                                        )),index=unlabeledX.T.index, columns=['PosN', 'NegN']).rename_axis("X")

                y_logpos = rel['PosN']*self.feature_log_prob_['Y_N'].loc[(1, 1)]\
                                + rel['NegN']*self.feature_log_prob_['Y_N'].loc[(1, 0)]
                y_logneg = rel['PosN']*self.feature_log_prob_['Y_N'].loc[(0, 1)]\
                                + rel['NegN']*self.feature_log_prob_['Y_N'].loc[(0, 0)]

                neg = np.exp(base_logneg + y_logneg)
                pos = np.exp(base_logpos + y_logpos)

                predictions = pos / (neg+pos)
                if self.calibrate:
                    logits = scipy.special.logit(predictions)
                    average = data.Y.loc[data.Mask.Labeled].mean()
                    logits -= np.percentile(logits, average*100)
                    predictions = scipy.special.expit(logits)

        return predictions


    def fit(self, data):
        # Only use the labeled data to fit
        labeledX = data.X.loc[data.Mask.Labeled, :]
        labeledY = data.Y.loc[data.Mask.Labeled, :]

        self.class_log_prior_ = np.log([1-labeledY.Y.mean(), labeledY.Y.mean()])

        # Compute X log conditional values
        log_posx = np.log(labeledY.join(labeledX).groupby('Y').mean())
        log_posx['X'] = 1
        log_posx = log_posx.reset_index().set_index(['Y', 'X'])
        log_negx = np.log(1-labeledY.join(labeledX).groupby('Y').mean())
        log_negx['X'] = 0
        log_negx = log_negx.reset_index().set_index(['Y', 'X'])

        self.feature_log_prob_ = pandas.concat([log_posx,log_negx])
        
        if self.learnmethod == 'riid':
            labeledE = data.E.loc[data.Mask.Labeled, data.Mask.Labeled]
            # Create basic Y | Y_N conditionals
            y = pandas.DataFrame(0, index=self.feature_log_prob_.index, columns=['Y_N'])

            y.loc[(1,1), 'Y_N'] = np.dot(np.dot(labeledY.T.values, labeledE.values), labeledY.values)[0]
            y.loc[(1,0), 'Y_N'] = np.dot(np.dot(labeledY.T.values, labeledE.values), 1-labeledY.values)[0]
            y.loc[(0,1), 'Y_N'] = np.dot(np.dot(1-labeledY.T.values, labeledE.values), labeledY.values)[0]
            y.loc[(0,0), 'Y_N'] = np.dot(np.dot(1-labeledY.T.values, labeledE.values), 1-labeledY.values)[0]

            y['Total'] = y.groupby(level=0).transform('sum')
            y['Y_N'] /= y['Total']
            del y['Total']

            y['Y_N'] = np.log(y['Y_N'])

            self.feature_log_prob_ = self.feature_log_prob_.join(y)

        elif self.learnmethod == 'r_uncertain':
            labeledE = data.E.loc[data.Mask.Labeled, :]

            # Create basic Y | Y_N conditionals
            y = pandas.DataFrame(0, index=self.feature_log_prob_.index, columns=['Y_N'])

            y.loc[(1,1), 'Y_N'] = np.dot(np.dot(labeledY.T.values, labeledE.values), data.Y.values)[0]
            y.loc[(1,0), 'Y_N'] = np.dot(np.dot(labeledY.T.values, labeledE.values), 1-data.Y.values)[0]
            y.loc[(0,1), 'Y_N'] = np.dot(np.dot(1-labeledY.T.values, labeledE.values), data.Y.values)[0]
            y.loc[(0,0), 'Y_N'] = np.dot(np.dot(1-labeledY.T.values, labeledE.values), 1-data.Y.values)[0]

            y['Total'] = y.groupby(level=0).transform('sum')
            y['Y_N'] /= y['Total']
            del y['Total']

            y['Y_N'] = np.log(y['Y_N'])

            self.feature_log_prob_ = self.feature_log_prob_.join(y)
            

        return self


class EMWrapper:
    def __init__(self, basemodel, emiters=10, **kwargs):
        self.basemodel = basemodel(learnmethod='riid', **kwargs)
        self.emiters=emiters

    def fit(self, data):
        self.basemodel.fit(data)
        self.basemodel.learnmethod = 'r_uncertain'
        self.predictions = self.basemodel.predict(data)

        for i in range(self.emiters):
            data.Y[data.Mask.Unlabeled] = self.predictions.values.reshape(len(self.predictions), 1)
            self.basemodel.fit(data)
            self.predictions = self.basemodel.predict(data)

        return self

    def predict(self, data):
        return self.basemodel.predict(data)
