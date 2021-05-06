import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score, roc_curve, mean_squared_error, auc
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, RFE, RFECV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.neighbors import KNeighborsClassifier




def plot_cf(model, x_train_data, y_train_data, x_test_data, y_test_data):
    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(10,6))
    plot_confusion_matrix(model, x_train_data, y_train_data, cmap=plt.cm.magma, ax=ax1)
    plot_confusion_matrix(model, x_test_data, y_test_data, cmap=plt.cm.magma, ax=ax2)
    ax1.set_title('Training Data', size=20)
    ax2.set_title('Testing Data', size=20)
    return plt.show()

def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1,2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        f1 = f1_score(y_test, preds)
        if f1 > best_score:
            best_k = k
            best_score = f1
    
    print('Best value for k: {}'.format(best_k))
    print('F1 Score: {}'.format(best_score))