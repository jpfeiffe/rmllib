import argparse
import pandas
import numpy.random as random
import sklearn.metrics

from rmllib.data.build import BostonMedians
from rmllib.models.local import RelationalNaiveBayes
from rmllib.models.inference import VariationalInference
from rmllib.models.inferning import ExpectationMaximization

pandas.options.mode.chained_assignment = None

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-d', '--dataset', default='Boston', choices=['Boston'], help='Dataset to Load')
    PARSER.add_argument('-s', '--seed', type=int, default=17, help='Random seed for numpy')
    ARGS = PARSER.parse_args()

    # Seed numpy
    random.seed(ARGS.seed)

    DATA = BostonMedians(subfeatures=['RM', 'AGE'])

    TRAIN_DATA = DATA.create_training(labeled_frac=.1)

    # Train data
    RNB_IID = RelationalNaiveBayes(learn_method='iid', infer_method='iid', calibrate=True).fit(TRAIN_DATA.copy())
    RNB_RIID = RelationalNaiveBayes(learn_method='r_iid', infer_method='r_iid', calibrate=True).fit(TRAIN_DATA.copy())
    RNB_VI = VariationalInference(RelationalNaiveBayes)(infer_iter=10, learn_method='r_iid', calibrate=True).fit(TRAIN_DATA.copy())
    RNB_EM = ExpectationMaximization(VariationalInference(RelationalNaiveBayes))(learn_iter=3, infer_iter=10, calibrate=True).fit(TRAIN_DATA.copy())

    P_IID = RNB_IID.predict_proba(TRAIN_DATA)
    P_RIID = RNB_RIID.predict_proba(TRAIN_DATA)
    P_VI = RNB_VI.predict_proba(TRAIN_DATA)
    P_EM = RNB_EM.predict_proba(TRAIN_DATA)

    print('IID Average Prediction:', P_IID[:, 1].mean(), 'AUC:', sklearn.metrics.roc_auc_score(DATA.labels.Y[DATA.mask.Unlabeled], P_IID[:, 1]))
    print('RIID Average Prediction:', P_RIID[:, 1].mean(), 'AUC:', sklearn.metrics.roc_auc_score(DATA.labels.Y[DATA.mask.Unlabeled], P_RIID[:, 1]))
    print('VI Average Prediction:', P_VI[:, 1].mean(), 'AUC:', sklearn.metrics.roc_auc_score(DATA.labels.Y[DATA.mask.Unlabeled], P_VI[:, 1]))
    print('EM Average Prediction:', P_EM[:, 1].mean(), 'AUC:', sklearn.metrics.roc_auc_score(DATA.labels.Y[DATA.mask.Unlabeled], P_EM[:, 1]))
