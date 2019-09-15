from sklearn.model_selection import cross_validate, StratifiedKFold
import numpy as np


def CV_metrics(model, X_train, y_train):
    scoring = ['roc_auc']
    cv_strat = StratifiedKFold(shuffle=True, random_state=42)
    cv = cross_validate(model, X_train, y_train, cv=cv_strat, scoring=scoring, n_jobs=2)
    for s in scoring:
        print(s, ' avg: ', '%.3f' % np.mean(cv['test_' + s]), ' ', ['%.3f' % item for item in cv['test_' + s]])
    return np.mean(cv['test_roc_auc'])