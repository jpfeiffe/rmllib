import argparse
import rmllib.Data
import rmllib.Generators
import rmllib.Learners
import numpy.random as random

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

    rmllib.Learners.RNB().fit(DATA)
