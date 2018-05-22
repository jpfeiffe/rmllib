import pandas
import numpy as np
import scipy.special
import scipy.stats

class Learner():
    def __init__(self):
        return

class RNB(Learner):
    def __init__(self):
        super().__init__()
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
    
    def predict_proba(self, data, infermethod='vi', calibrate=True, iters=10, unlabeled_confidence=1):
        assert(infermethod =='iid' or infermethod =='riid' or infermethod == 'vi')

        unlabeledX = data.X.loc[data.Mask.Unlabeled, :].T

        # Other features taken care of...
        logneg = (unlabeledX.where(unlabeledX != 0, self.feature_log_prob_.loc[0,:].T[0], axis=0) + unlabeledX.where(unlabeledX != 1, self.feature_log_prob_.loc[0,:].T[1], axis=0)).T.sum(axis=1)
        logpos = (unlabeledX.where(unlabeledX != 0, self.feature_log_prob_.loc[1,:].T[0], axis=0) + unlabeledX.where(unlabeledX != 1, self.feature_log_prob_.loc[1,:].T[1], axis=0)).T.sum(axis=1)

        base_logneg = self.class_log_prior_[0] + logneg 
        base_logpos = self.class_log_prior_[1] + logpos

        # Relational estimates
        if infermethod == 'iid':
            neg = np.exp(base_logneg)
            pos = np.exp(base_logpos)

        elif infermethod == 'riid' or infermethod == 'vi':
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
        if calibrate:
            logits = scipy.special.logit(predictions)
            average = data.Y.loc[data.Mask.Labeled].mean()
            logits -= np.percentile(logits, average*100)
            predictions = scipy.special.expit(logits)

        # Joint inference
        if infermethod == 'vi':
            ECopy = data.E.copy()
            allToUnlabelE = ECopy.loc[data.Mask.Unlabeled, :]
            allToUnlabelE.loc[:, data.Mask.Unlabeled] *= unlabeled_confidence
            # Current Predictions     
            for i in range(iters):
                Q = data.Y.copy()
                Q[data.Mask.Unlabeled] = predictions

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
                if calibrate:
                    logits = scipy.special.logit(predictions)
                    average = data.Y.loc[data.Mask.Labeled].mean()
                    logits -= np.percentile(logits, average*100)
                    predictions = scipy.special.expit(logits)

        return predictions


    def fit(self, data, learnmethod='iid'):
        assert(learnmethod=='iid' or learnmethod=='riid')

        # Only use the labeled data to fit
        labeledX = data.X.loc[data.Mask.Labeled, :]
        labeledY = data.Y.loc[data.Mask.Labeled, :]
        labeledE = data.E.loc[data.Mask.Labeled, data.Mask.Labeled]

        self.class_log_prior_ = np.log([1-labeledY.Y.mean(), labeledY.Y.mean()])

        # Compute X log conditional values
        log_posx = np.log(labeledY.join(labeledX).groupby('Y').mean())
        log_posx['X'] = 1
        log_posx = log_posx.reset_index().set_index(['Y', 'X'])
        # print(log_posx)
        # exit(0)
        log_negx = np.log(1-labeledY.join(labeledX).groupby('Y').mean())
        log_negx['X'] = 0
        log_negx = log_negx.reset_index().set_index(['Y', 'X'])

        self.feature_log_prob_ = pandas.concat([log_posx,log_negx])
        
        if learnmethod == 'riid':
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

        return self


class RNB_EM(RNB):
    def __init__(self):
        super().__init__()

    def fit(self, data, method='em', iters=10):
        assert(method=='em' and iters > 0)

        super().fit(data, method='riid')

        print(self.feature_log_prob_)
        exit()


        # Only use the labeled data to fit
        labeledX = data.X.loc[data.Mask.Mask, :]
        labeledY = data.Y.loc[data.Mask.Mask, :]
        labeledE = data.E.loc[data.Mask.Mask, data.Mask.Mask]

        self.class_log_prior_ = np.log([1-labeledY.Y.mean(), labeledY.Y.mean()])

        # Compute X log conditional values
        log_posx = np.log(labeledY.join(labeledX).groupby('Y').mean())
        log_posx['X'] = 1
        log_posx = log_posx.reset_index().set_index(['Y', 'X'])
        # print(log_posx)
        # exit(0)
        log_negx = np.log(1-labeledY.join(labeledX).groupby('Y').mean())
        log_negx['X'] = 0
        log_negx = log_negx.reset_index().set_index(['Y', 'X'])

        self.feature_log_prob_ = pandas.concat([log_posx,log_negx])
        
        if method=='riid':
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

        return self