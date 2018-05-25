import argparse
import pandas
import numpy.random as random
import sklearn.metrics
import time

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

    DATASETS = []
    MODELS = []

    DATASETS.append(BostonMedians(name='Boston Medians', subfeatures=['RM', 'AGE'], sparse=True).node_sample_mask(.1))
    # DATASETS.append(BayesSampleDataset(name='Sparse 1,000,000', n_rows=1000000, n_features=3, generator=edge_rejection_generator, density=.000025, sparse=True).node_sample_mask(.01))

    # MODELS.append(RelationalNaiveBayes(name='NB', learn_method='iid', infer_method='iid', calibrate=True))
    # MODELS.append(RelationalNaiveBayes(name='RNB', learn_method='r_iid', infer_method='r_iid', calibrate=True))
    MODELS.append(VariationalInference(RelationalNaiveBayes)(name='RNB_VI', learn_method='r_iid', infer_method='r_iid', calibrate=True))
    # MODELS.append(ExpectationMaximization(VariationalInference(RelationalNaiveBayes))(name='RNB_EM_VI', learn_method='r_iid', infer_method='r_iid', calibrate=True))

    print('Begin Evaluation')
    for dataset in DATASETS:
        TRAIN_DATA = dataset.create_training()
        
        for model in MODELS:
            print('\n' + model.name + ": Begin Train")
            train_data = TRAIN_DATA.copy()
            start_time = time.time()
            model.fit(train_data)
            print(model.name, 'Training Time:', time.time() - start_time)
            model.predictions = model.predict_proba(train_data)
            print(model.name, 'Total Time:', time.time() - start_time)            
            print(model.name, 'Average Prediction:', model.predictions[:, 1].mean(), 'AUC:', sklearn.metrics.roc_auc_score(dataset.labels.Y[dataset.mask.Unlabeled], model.predictions[:, 1]))
            print(model.name + ": End Train")

    print('End Evaluation')
