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
    # rename cols
    Y = Y.rename(columns={f'{i}': f'target_{i}' for i in range(1, 6)})
    X1_map = {f'{i}': f'X_1_{i}' for i in range(1, 26)}
    X3_map = {f'{i}': f'X_3_{i}' for i in range(1, 453)}
    X1 = X1.rename(columns=X1_map)
    X1_test = X1_test.rename(columns=X1_map)
    X3 = X3.rename(columns=X3_map)
    X3_test = X3_test.rename(columns=X3_map)
    # type cast
    for col in Y.columns:
        if col != 'id':
            Y[col] = Y[col].astype(int)
    return X1, X2, X3, Y, X1_test, X2_test, X3_test


def agg_and_merge(X1, X2, X3, embeddings):
    X = pd.merge(X1, X3, on='id')
    # log A
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
    # max
    X2_aggregations.append(X2.groupby('id')['A'].max().reset_index().rename(columns={'A': 'max_A'}))
    # merge everything
    for X2_agg in X2_aggregations:
        X = pd.merge(X, X2_agg, how='left', on='id')
    for e in embeddings:
        X = pd.merge(X, e, how='left', on='id')
    X.fillna(-1000, inplace=True)
    return X