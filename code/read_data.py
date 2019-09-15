import pandas as pd
import numpy as np


def read_data():
    TRAIN_PATH = 'train/'
    TEST_PATH = 'test/'
    X1 = pd.read_csv(TRAIN_PATH + 'X1.csv')
    X2 = pd.read_csv(TRAIN_PATH + 'X2.csv')
    X3 = pd.read_csv(TRAIN_PATH + 'X3.csv')
    Y = pd.read_csv(TRAIN_PATH + 'Y.csv')
    X1_test = pd.read_csv(TEST_PATH + 'X1.csv')
    X2_test = pd.read_csv(TEST_PATH + 'X2.csv')
    X3_test = pd.read_csv(TEST_PATH + 'X3.csv')
    return X1, X2, X3, Y, X1_test, X2_test, X3_test


def agg_and_merge(X1, X2, X3):
    X = pd.merge(X1, X3, on='id')
    # log A
    X2['log_A'] = np.log(X2['A'] + 1)
    X2_aggregations = []
    # length
    X2_aggregations.append(X2.groupby('id')['A'].count().reset_index().rename(columns={'A': 'length_A'}))
    # unique
    X2_aggregations.append(X2.groupby('id')['A'].nunique().reset_index().rename(columns={'A': 'unique_A'}))
    # first
    X2_aggregations.append(X2.groupby('id')['A'].first().reset_index().rename(columns={'A': 'first_A'}))
    # last
    X2_aggregations.append(X2.groupby('id')['A'].last().reset_index().rename(columns={'A': 'last_A'}))
    # min
    X2_aggregations.append(X2.groupby('id')['A'].min().reset_index().rename(columns={'A': 'min_A'}))
    X2_aggregations.append(X2.groupby('id')['log_A'].min().reset_index().rename(columns={'log_A': 'log_min_A'}))
    # max
    X2_aggregations.append(X2.groupby('id')['A'].max().reset_index().rename(columns={'A': 'max_A'}))
    X2_aggregations.append(X2.groupby('id')['log_A'].max().reset_index().rename(columns={'log_A': 'log_max_A'}))
    # mean
    X2_aggregations.append(X2.groupby('id')['A'].mean().reset_index().rename(columns={'A': 'mean_A'}))
    X2_aggregations.append(X2.groupby('id')['log_A'].mean().reset_index().rename(columns={'log_A': 'log_mean_A'}))
    # median
    X2_aggregations.append(X2.groupby('id')['A'].median().reset_index().rename(columns={'A': 'median_A'}))
    X2_aggregations.append(X2.groupby('id')['log_A'].median().reset_index().rename(columns={'log_A': 'log_median_A'}))
    ## log_var
    # X2_aggregations.append(X2.groupby('id')['log_A'].var().reset_index().rename(columns={'log_A': 'log_var_A'}))
    # merge everything
    for X2_agg in X2_aggregations:
        X = pd.merge(X, X2_agg, on='id')
    return X