import argparse
import pandas
import numpy.random as random
import sklearn.metrics

from rmllib.data.load import BostonMedians
from rmllib.data.generate import BayesSampleDataset
from rmllib.data.generate import edge_rejection_generator
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

    # DATA = BostonMedians(subfeatures=['RM', 'AGE'], sparse=True)
    DATASETS = []
    TRAIN_DATASETS = []
    MODELS = []

    # DATASETS.append(BayesSampleDataset(name='Dense 100', n_rows=100, sparse=False))
    # DATASETS.append(BayesSampleDataset(name='Sparse 100', n_rows=1000, sparse=True, generator=edge_rejection_generator))
    DATASETS.append(BostonMedians(name='Boston Medians', subfeatures=['RM', 'AGE'], sparse=False))

    # DATASETS.append(BayesSampleDataset(name='Dense 1000', n_rows=1000, sparse=False))
    # DATASETS.append(BayesSampleDataset(name='Sparse 1000', n_rows=1000, sparse=True))

    # # DATASETS.append(BayesSampleDataset(name='Dense 10000', n_rows=100000, sparse=False))
    # DATASETS.append(BayesSampleDataset(name='Sparse 25000', n_rows=25000, sparse=True))

    MODELS.append(RelationalNaiveBayes(name='NB', learn_method='iid', infer_method='iid', calibrate=True))
    MODELS.append(RelationalNaiveBayes(name='RNB', learn_method='r_iid', infer_method='r_iid', calibrate=True))
    MODELS.append(VariationalInference(RelationalNaiveBayes)(name='RNB_VI', learn_method='r_iid', infer_method='r_iid', calibrate=True))
    MODELS.append(ExpectationMaximization(VariationalInference(RelationalNaiveBayes))(name='RNB_EM_VI', learn_method='r_iid', infer_method='r_iid', calibrate=True))

    for dataset in DATASETS:
        TRAIN_DATA = dataset.create_training(labeled_frac=.1)
        
        for model in MODELS:
            train_data = TRAIN_DATA.copy()
            model.fit(train_data)
            model.predictions = model.predict_proba(train_data)
            print(model.name, 'Average Prediction:', model.predictions[:, 1].mean(), 'AUC:', sklearn.metrics.roc_auc_score(dataset.labels.Y[dataset.mask.Unlabeled], model.predictions[:, 1]))

    exit()
        

    # print(DATA.labels)
    # print(DATA.name)
    # # DATA = BayesSampleDataset(name='Sparse 100', size=100, mu_match=.56, sparse=True)
    # TRAIN_DATA = DATA.create_training(labeled_frac=.1)
    # print(DATA.name)
    # MODELS = []
    # # MODELS.append(RelationalNaiveBayes(name='NB', learn_method='iid', infer_method='iid', calibrate=True))
    # # MODELS.append(VariationalInference(RelationalNaiveBayes)(name='RNB VI', infer_iter=10, learn_method='r_iid', calibrate=True))
    # # MODELS.append(ExpectationMaximization(VariationalInference(RelationalNaiveBayes))(name='RNB EM', learn_iter=3, infer_iter=10, calibrate=True))

    # for model in MODELS:
    #     train_data = TRAIN_DATA.copy()
    #     model.fit(train_data)
    #     model.predictions = model.predict_proba(train_data)

    #     #print(model.name, 'Average Prediction:', model.predictions[:, 1].mean(), 'AUC:', sklearn.metrics.roc_auc_score(DATA.labels.Y[DATA.mask.Unlabeled], model.predictions[:, 1]))
