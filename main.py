'''
pip install xgboost
pip install tensorflow
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
'''

import os

os.makedirs('Data visualization', exist_ok=True)
os.makedirs('Saved Data', exist_ok=True)
os.makedirs('Results', exist_ok=True)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datagen import datagen
from save_load import *
from detection import RF, KNN, DTree, SVM, MLRK, xgboost, logisticRegression
from plot_result import plotres
import matplotlib.pyplot as plt


def full_analysis():
    datagen()

    # 70 training, 30 testing

    x_train_70 = load('x_train_70')
    x_test_70 = load('x_test_70')
    y_train_70 = load('y_train_70')
    y_test_70 = load('y_test_70')

    # 80 training, 20 testing

    x_train_80 = load('x_train_80')
    x_test_80 = load('x_test_80')
    y_train_80 = load('y_train_80')
    y_test_80 = load('y_test_80')

    training_data = [(x_train_70, y_train_70, x_test_70, y_test_70), (x_train_80, y_train_80, x_test_80, y_test_80)]

    i = 70

    for train_data in training_data:
        X_train, y_train, X_test, y_test = train_data

        pred, met = MLRK(X_train, y_train, X_test, y_test)
        save(f'proposed_{i}', met)
        save(f'predicted_{i}', pred)

        pred, met = SVM(X_train, y_train, X_test, y_test)
        save(f'svm_{i}', met)

        pred, met = RF(X_train, y_train, X_test, y_test)
        save(f'rf_{i}', met)

        pred, met = DTree(X_train, y_train, X_test, y_test)
        save(f'dtree_{i}', met)

        pred, met = KNN(X_train, y_train, X_test, y_test)
        save(f'knn_{i}', met)

        pred, met = xgboost(X_train, y_train, X_test, y_test)
        save(f'xgboost_{i}', met)

        pred, met = logisticRegression(X_train, y_train, X_test, y_test)
        save(f'logisticRegression_{i}', met)

        i += 10


a = 0
if a == 1:
    full_analysis()

plotres()
plt.show()