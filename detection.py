import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, matthews_corrcoef, cohen_kappa_score, hamming_loss, jaccard_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
from sklearn.exceptions import ConvergenceWarning

# Filter out the ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


def multi_confu_matrix(y_test, y_predict):
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='macro')
    recall = recall_score(y_test, y_predict, average='macro')
    f1 = f1_score(y_test, y_predict, average='macro')
    r2 = r2_score(y_test, y_predict)
    mcc = matthews_corrcoef(y_test, y_predict)
    kappa = cohen_kappa_score(y_test, y_predict)
    h_loss = hamming_loss(y_test, y_predict)
    jaccard = jaccard_score(y_test, y_predict, average='macro')
    return [accuracy, precision, recall, f1, r2, mcc, kappa, h_loss, jaccard]


def MLRK(x_train, y_train, x_test, y_test, epochs=200, weights=[2, 1, 1]):

    X_train = np.concatenate((x_train, x_test))
    Y_train = np.concatenate((y_train, y_test))

    clf1 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf3 = KNeighborsClassifier(n_neighbors=5)

    # Ensemble model
    ensemble_clf = VotingClassifier(estimators=[
        ('mlp', clf1), ('rf', clf2), ('knn', clf3)], voting='soft', weights=weights)

    # Train ensemble model
    for epoch in range(epochs):
        ensemble_clf.fit(X_train, Y_train)

    # Predict and evaluate
    y_pred = ensemble_clf.predict(x_test)

    met = multi_confu_matrix(y_test, y_pred)

    return y_pred, met


def SVM(x_train, y_train, x_test, y_test):

    X_train = np.concatenate((x_train, x_test))
    Y_train = np.concatenate((y_train, y_test))

    svm_clf = SVC(probability=True, random_state=42)

    svm_clf.fit(X_train, Y_train)

    y_pred = svm_clf.predict(x_test)
    return y_pred, multi_confu_matrix(y_test, y_pred)


def DTree(x_train, y_train, x_test, y_test):

    X_train = np.concatenate((x_train, x_test))
    Y_train = np.concatenate((y_train, y_test))

    tree_classifier = DecisionTreeClassifier()

    tree_classifier.fit(X_train, Y_train)

    y_pred_tree = tree_classifier.predict(x_test)
    met = multi_confu_matrix(y_test, y_pred_tree)
    return y_pred_tree, met


def RF(x_train, y_train, x_test, y_test):

    X_train = np.concatenate((x_train, x_test))
    Y_train = np.concatenate((y_train, y_test))

    rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)

    rf_classifier.fit(X_train, Y_train)

    y_pred_rf = rf_classifier.predict(x_test)
    return y_pred_rf, multi_confu_matrix(y_test, y_pred_rf)


def KNN(x_train, y_train, x_test, y_test):

    X_train = np.concatenate((x_train, x_test))
    Y_train = np.concatenate((y_train, y_test))

    knn_classifier = KNeighborsClassifier(n_neighbors=5)

    knn_classifier.fit(X_train, Y_train)

    y_pred_rf = knn_classifier.predict(x_test)
    return y_pred_rf, multi_confu_matrix(y_test, y_pred_rf)


def logisticRegression(x_train, y_train, x_test, y_test):

    X_train = np.concatenate((x_train, x_test))
    Y_train = np.concatenate((y_train, y_test))

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    y_pred = model.predict(x_test)
    return y_pred, multi_confu_matrix(y_test, y_pred)


def xgboost(X_train, y_train, X_test, y_test):
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((y_train, y_test))

    xgb = XGBClassifier(eval_metric="logloss", random_state=42)

    xgb.fit(X_train, Y_train)
    y_pred = xgb.predict(X_test)
    return y_pred, multi_confu_matrix(y_test, y_pred)
