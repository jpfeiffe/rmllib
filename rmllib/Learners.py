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
    
    def fit(self, data, iters=0):
        # Only use the labeled data to fit
        labeledX = data.X.loc[data.Mask.Mask, :]
        labeledY = data.Y.loc[data.Mask.Mask, :]
        labeledE = data.E.loc[data.Mask.Mask, data.Mask.Mask]

        self.class_log_prior_ = np.log([labeledY.Y.mean(), 1 - labeledY.Y.mean()])

        # Compute X log conditional values
        log_pos = np.log(labeledY.join(labeledX).groupby('Y').mean())
        log_pos['X'] = 1
        log_pos = log_pos.reset_index().set_index(['Y', 'X'])
        log_neg = np.log(1-labeledY.join(labeledX).groupby('Y').mean())
        log_neg['X'] = 0
        log_neg = log_neg.reset_index().set_index(['Y', 'X'])

        self.feature_log_prob_ = pandas.concat([log_pos,log_neg])
        
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



        return
