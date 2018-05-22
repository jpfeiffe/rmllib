import argparse
import rmllib.Data
import rmllib.Generators
import rmllib.Learners
import numpy.random as random
import sklearn.metrics

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-d', '--dataset', default='Boston', choices=['Boston'], help='Dataset to Load')
    PARSER.add_argument('-s', '--seed', type=int, default=16, help='Random seed for numpy')
    PARSER.add_argument('-p', '--positiveprobability', type=float, default=.5, help='Fraction of positive samples')
    ARGS = PARSER.parse_args()

    # Seed numpy
    random.seed(ARGS.seed)

    DATA = rmllib.Data.BostonMedians()
    DATA.labelmask()

    # Train data
    RNB = rmllib.Learners.RNB().fit(DATA)

    pred_iid = RNB.predict_proba(DATA, method='iid')
    pred_riid = RNB.predict_proba(DATA, method='riid')

    print(sklearn.metrics.roc_auc_score(DATA.Y.Y[~DATA.Mask.Mask], pred_iid))
    print(sklearn.metrics.roc_auc_score(DATA.Y.Y[~DATA.Mask.Mask], pred_riid))
