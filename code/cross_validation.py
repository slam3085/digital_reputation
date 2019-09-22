from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
import numpy as np


def CV_metrics(model, X_train, y_train):
    scoring = ['roc_auc']
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_validate(model, X_train, y_train, cv=cv_strat, scoring=scoring)
    for s in scoring:
        print(s, ' avg: ', '%.3f' % np.mean(cv['test_' + s]), ' ', ['%.3f' % item for item in cv['test_' + s]])
    return np.mean(cv['test_roc_auc'])


def CV_multilabel(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    roc_aucs = {target: [] for target in y.columns}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        for y_proba, target in zip(y_pred, y.columns):
            roc_aucs[target].append(roc_auc_score(y_test[target].values, y_proba[:, 1]))
    roc_auc_avgs = []
    for target in y.columns:
        roc_auc_avg = np.mean(roc_aucs[target])
        roc_auc_avgs.append(roc_auc_avg)
        print(f'{target}: {roc_auc_avg:.3f}, {roc_aucs[target]}')
    print(f'{np.mean(roc_auc_avgs)}')