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
    return X1, X2, X3, Y, X1_test, X2_test, X3_test


def calc_diffs(arr):
    return arr[1:] - arr[:-1]


def agg_and_merge(X1, X2, X3, embeddings):
    X = pd.merge(X1, X3, on='id')
    # copy and calc diffs for further use
    X2_cp = X2.copy()
    X2_cp['A_list'] = X2_cp['A'].apply(lambda x: [x])
    X2_cp = X2_cp[['id', 'A_list']].groupby('id')['A_list'].agg(sum).reset_index()
    X2_cp['A_list'] = X2_cp['A_list'].apply(lambda x: np.array(x))
    X2_cp['A_list_diffs'] = X2_cp['A_list'].apply(lambda x: calc_diffs(x))
    X2_cp['A_diffs_sum'] = X2_cp['A_list_diffs'].apply(lambda x: np.sum(x))
    X2_cp['A_diffs_first'] = X2_cp['A_list_diffs'].apply(lambda x: x[0] if len(x) > 0 else -1)
    X2_cp['A_diffs_last'] = X2_cp['A_list_diffs'].apply(lambda x: x[-1] if len(x) > 0 else -1)
    X2_cp['A_diffs_min'] = X2_cp['A_list_diffs'].apply(lambda x: np.min(x) if len(x) > 0 else -1)
    X2_cp['A_diffs_max'] = X2_cp['A_list_diffs'].apply(lambda x: np.max(x) if len(x) > 0 else -1)
    X2_cp['A_diffs_mean'] = X2_cp['A_list_diffs'].apply(lambda x: np.mean(x) if len(x) > 0 else -1)
    X2_cp['A_diffs_median'] = X2_cp['A_list_diffs'].apply(lambda x: np.median(x) if len(x) > 0 else -1)
    X2_cp['A_diffs_var'] = X2_cp['A_list_diffs'].apply(lambda x: np.var(x) if len(x) > 0 else -1)
    X2_cp['ratio_diffs_A'] = X2_cp['A_diffs_min'] / X2_cp['A_diffs_max']
    X2_cp['abs_ratio_diffs_A'] = np.abs(X2_cp['A_diffs_min'] / X2_cp['A_diffs_max'])
    X2_cp['mean_median_diffs_A'] = X2_cp['A_diffs_mean'] == X2_cp['A_diffs_median']
    X2_cp['first_less_last_diffs_A'] = X2_cp['A_diffs_first'] < X2_cp['A_diffs_last']
    X2_cp['first_more_last_diffs_A'] = X2_cp['A_diffs_first'] > X2_cp['A_diffs_last']
    X2_cp.fillna(0, inplace=True)
    X2_cp.drop(columns=['A_list', 'A_list_diffs'], inplace=True)
    # log A
    X2['log_A'] = np.log(X2['A'] + 1)
    X2_aggregations = []
    # length
    X2_aggregations.append(X2.groupby('id')['A'].count().reset_index().rename(columns={'A': 'length_A'}))
    # unique
    X2_aggregations.append(X2.groupby('id')['A'].nunique().reset_index().rename(columns={'A': 'unique_A'}))
    # sum
    X2_aggregations.append(X2.groupby('id')['A'].sum().reset_index().rename(columns={'A': 'sum_A'}))
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
    # var
    X2_aggregations.append(X2.groupby('id')['A'].var().reset_index().rename(columns={'A': 'var_A'}))
    X2_aggregations.append(X2.groupby('id')['log_A'].var().reset_index().rename(columns={'log_A': 'log_var_A'}))
    # merge everything
    for X2_agg in X2_aggregations:
        X = pd.merge(X, X2_agg, on='id')
    X = pd.merge(X, X2_cp, on='id')
    for e in embeddings:
        X = pd.merge(X, e, on='id')
    X.fillna(0, inplace=True)
    return X