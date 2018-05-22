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

    TRUE_DATA = rmllib.Data.BostonMedians(subfeatures=['RM', 'AGE'])
    TRUE_DATA.labelmask(labeled_frac=.3)

    DATA = TRUE_DATA.createtraining()

    # Train data
    RNB_IID = rmllib.Learners.RNB(learnmethod='iid', infermethod='iid', calibrate=True).fit(DATA.copy())
    RNB_RIID = rmllib.Learners.RNB(learnmethod='riid', infermethod='riid', calibrate=True).fit(DATA.copy())
    RNB_VI = rmllib.Learners.RNB(learnmethod='riid', infermethod='vi', calibrate=True, inferenceiters=10, unlabeled_confidence=1).fit(DATA.copy())
    # RNB_EM = rmllib.Learners.EMWrapper(rmllib.Learners.RNB, calibrate=True, inferenceiters=10, unlabeled_confidence=1).fit(DATA.copy())

    P_IID = RNB_IID.predict(DATA)
    P_RIID = RNB_RIID.predict(DATA)
    P_VI = RNB_VI.predict(DATA)
    # P_EM = RNB_EM.predict(DATA)

    print('IID Average Prediction:', P_IID.mean(), 'AUC:', sklearn.metrics.roc_auc_score(TRUE_DATA.Y.Y[TRUE_DATA.Mask.Unlabeled], P_IID))
    print('RIID Average Prediction:', P_RIID.mean(), 'AUC:', sklearn.metrics.roc_auc_score(TRUE_DATA.Y.Y[TRUE_DATA.Mask.Unlabeled], P_RIID))
    print('VI Average Prediction:', P_VI.mean(), 'AUC:', sklearn.metrics.roc_auc_score(TRUE_DATA.Y.Y[TRUE_DATA.Mask.Unlabeled], P_VI))
    # print('EM Average Prediction:', P_EM.mean(), 'AUC:', sklearn.metrics.roc_auc_score(TRUE_DATA.Y.Y[TRUE_DATA.Mask.Unlabeled], P_EM))


    #RNB_EM = rmllib.Learners.RNB_EM().fit(DATA)
    #print(RNB_EM.class_log_prior_)