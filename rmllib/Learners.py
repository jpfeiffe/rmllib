import pandas
import numpy as np

class Learner():
    def __init__(self):
        return

class RNB():
    def __init__(self):
        super()
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
    
    def predict_proba(self, data, method='iid'):
        assert(method=='iid' or method=='riid')

        unlabeledX = data.X.loc[~data.Mask.Mask, :].T
        labeledY = data.Y.loc[data.Mask.Mask, :]
        labelToUnlabelE = data.E.loc[~data.Mask.Mask, data.Mask.Mask]

        # Other features taken care of...
        logneg = (unlabeledX.where(unlabeledX != 0, self.feature_log_prob_.loc[0,:].T[0], axis=0) + unlabeledX.where(unlabeledX != 1, self.feature_log_prob_.loc[0,:].T[1], axis=0)).T.sum(axis=1)
        logpos = (unlabeledX.where(unlabeledX != 0, self.feature_log_prob_.loc[1,:].T[0], axis=0) + unlabeledX.where(unlabeledX != 1, self.feature_log_prob_.loc[1,:].T[1], axis=0)).T.sum(axis=1)

        logneg = self.class_log_prior_[0] + logneg 
        logpos = self.class_log_prior_[1] + logpos

        # Relational estimates
        if method == 'riid':
            rel = pandas.DataFrame(\
                                    np.hstack((np.dot(labelToUnlabelE.values, labeledY.values), np.dot(labelToUnlabelE.values, 1-labeledY.values))),\
                                    index=unlabeledX.T.index, columns=['PosN', 'NegN']).rename_axis("X")

            y_logpos = rel['PosN']*self.feature_log_prob_['Y_N'].loc[(1, 1)]\
                            + rel['NegN']*self.feature_log_prob_['Y_N'].loc[(1, 0)]
            y_logneg = rel['PosN']*self.feature_log_prob_['Y_N'].loc[(0, 1)]\
                            + rel['NegN']*self.feature_log_prob_['Y_N'].loc[(0, 0)]

            logneg += y_logneg
            logpos += y_logpos

        neg = np.exp(logneg)
        pos = np.exp(logpos)
        
        return pos / (pos + neg)


    def fit(self, data, method='iid', iters=0):
        assert(method=='iid')

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
        for i in range(iters):
            continue

        return self
