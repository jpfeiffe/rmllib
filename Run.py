import argparse
import rmllib.Data
import rmllib.Generators
import rmllib.Learners
import numpy.random as random
import sklearn.metrics
import pandas
pandas.options.mode.chained_assignment = None 

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-d', '--dataset', default='Boston', choices=['Boston'], help='Dataset to Load')
    PARSER.add_argument('-s', '--seed', type=int, default=16, help='Random seed for numpy')
    PARSER.add_argument('-p', '--positiveprobability', type=float, default=.5, help='Fraction of positive samples')
    ARGS = PARSER.parse_args()

    # Seed numpy
    random.seed(ARGS.seed)

    DATA = rmllib.Data.BostonMedians(subfeatures=['RM', 'AGE'])
    DATA.labelmask(labeled_frac=.3)

    # Train data
    RNB = rmllib.Learners.RNB().fit(DATA, learnmethod='riid')

    P_IID = RNB.predict_proba(DATA, infermethod='iid')
    P_RIID = RNB.predict_proba(DATA, infermethod='riid')
    P_VI = RNB.predict_proba(DATA, infermethod='vi', iters=10, calibrate=True, unlabeled_confidence=1)

    print('IID Average Prediction:', P_IID.mean(), 'AUC:', sklearn.metrics.roc_auc_score(DATA.Y.Y[DATA.Mask.Unlabeled], P_IID))
    print('RIID Average Prediction:', P_RIID.mean(), 'AUC:', sklearn.metrics.roc_auc_score(DATA.Y.Y[DATA.Mask.Unlabeled], P_RIID))
    print('VI Average Prediction:', P_VI.mean(), 'AUC:', sklearn.metrics.roc_auc_score(DATA.Y.Y[DATA.Mask.Unlabeled], P_VI))


    #RNB_EM = rmllib.Learners.RNB_EM().fit(DATA)
    #print(RNB_EM.class_log_prior_)