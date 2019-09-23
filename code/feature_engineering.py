import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import umap


def eng(X_train, X_test):
    n_train = len(X_train)
    X = pd.concat([X_train, X_test], ignore_index=True)
    # drop trash
    for col in X.columns:
        if X[col].nunique() == 1:
            X.drop(columns=[col], inplace=True)
    # various
    X['id_odd'] = X['id'] % 2
    X['id_3'] = X['id'] % 3
    X['id_5'] = X['id'] % 5
    X['id_7'] = X['id'] % 7
    X['id_11'] = X['id'] % 11
    # X 1
    X['X_1_2_equal_X_1_3'] = X['X_1_2'] == X['X_1_3']
    X_train, X_test = X[:n_train], X[n_train:]
    return X_train, X_test


def add_emedding_features(X_train, X_test, **options):
    n_train = len(X_train)
    X = pd.concat([X_train, X_test], ignore_index=True)
    print('umap...')
    embedding1 = umap.UMAP(**options).fit_transform(X)
    embedding2 = umap.UMAP(**options).fit_transform(MinMaxScaler().fit_transform(X))
    n_components = options.get('n_components', 2)
    for i in range(n_components):
        X[f'embedding1_{i}'] = embedding1[:, i]
        # X[f'embedding2_{i}'] = embedding2[:, i]
    X_train, X_test = X[:n_train], X[n_train:]
    return X_train, X_test


def normalize(X_train, X_test):
    n_train = len(X_train)
    X = pd.concat([X_train, X_test], ignore_index=True)
    # normalization
    for col in X:
        if X[col].nunique() > 2 and col != 'id':
            X[col] = (X[col] - X[col].min()) / (X[col].max() - X[col].min())
    X_train_norm, X_test_norm = X[:n_train], X[n_train:]
    return X_train_norm, X_test_norm


def OHE(X_train, X_test):
    n_train = len(X_train)
    X = pd.concat([X_train, X_test], ignore_index=True)
    cols_to_ohe = []
    for col in X:
        if 2 < X[col].nunique() < 31:
            cols_to_ohe.append(col)
    X = pd.get_dummies(X, columns=cols_to_ohe)
    X_train_norm, X_test_norm = X[:n_train], X[n_train:]
    return X_train_norm, X_test_norm