import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, \
                            f1_score, confusion_matrix


def feature_importance_plot(columns, feature_importances_, feat_count=50):
    feature_imp = pd.DataFrame(sorted(zip(feature_importances_, columns)),
                               columns=['Value', 'Feature'])
    feature_imp['Value'] = feature_imp['Value'] / feature_imp['Value'].sum() * 100
    plt.figure(figsize=(20, 10))
    data = feature_imp.sort_values(by="Value", ascending=False).head(feat_count)
    sns.barplot(x="Value", y="Feature", data=data, color='#1f77b4')
    plt.tight_layout()
    plt.show()