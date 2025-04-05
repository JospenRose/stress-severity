import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from save_load import load
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


def classfi_report(y_test, predicted, k):

    # Classification report
    class_report = classification_report(y_test, predicted, output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()

    # Plot the DataFrame
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')

    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.title('Classification Report', fontweight='bold', fontname='Serif')

    plt.savefig(f'Results/Classification Report learning rate - {k}.png')
    plt.show()


def bar_plot(label, data1, data2, metric):

    # create data
    df = pd.DataFrame([data1, data2],
                      columns=label)
    df1 = pd.DataFrame()
    df1['Learn Rate(%)'] = [70, 80]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Learn Rate(%)',
            kind='bar',
            stacked=False)

    plt.ylabel(metric, fontweight='bold', fontname='Serif')
    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.xlabel('Learn Rate %', fontweight='bold', fontname='Serif')
    plt.legend(loc='center', prop={'weight': 'bold', 'family': 'Serif', 'size': 7})
    plt.title(metric, fontweight='bold', fontname='Serif')

    plt.savefig('./Results/'+metric+'.png', dpi=400)
    plt.show(block=False)


def densityplot(actual, predicted, learning_rate):

    plt.figure(figsize=(8, 6))
    sns.kdeplot(actual, color='orange', label='Actual',  fill=True)
    sns.kdeplot(predicted, color='blue', label='Predicted',  fill=True)

    plt.ylabel('Density', fontweight='bold', fontname='Serif')
    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.xlabel('Value', fontweight='bold', fontname='Serif')
    plt.legend(loc='upper right', prop={'weight': 'bold', 'family': 'Serif'})
    plt.title("Density plot of Actual vs Predicted values", fontweight='bold', fontname='Serif')

    plt.savefig(f'Results/Density Plot Learning rate-{learning_rate}.png')
    plt.show()


def confu_plot(y_test, y_pred, k):
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Purples, values_format='.0f', ax=ax)

    plt.ylabel('True Labels', fontweight='bold', fontname='Serif')
    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.xlabel('Predicted Labels', fontweight='bold', fontname='Serif')
    plt.title('Confusion Matrix', fontweight='bold', fontname='Serif')
    plt.tight_layout()
    plt.savefig(f'Results/Confusion Matrix Learning rate-{k}.png')
    plt.show()


def precision_recall_plot(y_test, y_pred, k):
    # Binarize the output labels
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    # Compute Precision-Recall curve and plot
    precision = dict()
    recall = dict()
    n_classes = len(classes)

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_bin[:, i])

    # Plot Precision-Recall curve for each class
    plt.figure(figsize=(10, 7))
    colors = cycle(
        ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'green', 'purple', 'brown', 'pink'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Class {0} (area = {1:0.2f})'.format(i, auc(recall[i], precision[i])))

    plt.ylabel('Precision', fontweight='bold', fontname='Serif')
    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.xlabel('Recall', fontweight='bold', fontname='Serif')
    plt.legend(loc="lower left", prop={'weight': 'bold', 'family': 'Serif'})
    plt.title('Precision-Recall curve for multi-class data', fontweight='bold', fontname='Serif')
    plt.tight_layout()
    plt.savefig(f'Results/Precision Recall Curve - learning rate - {k}.png')
    plt.show()


def plotres():

    # learning rate -  70  and 30

    svm_70 = load('svm_70')
    xgboost_70 = load('xgboost_70')
    dtree_70 = load('dtree_70')
    rf_70 = load('rf_70')
    knn_70 = load('knn_70')
    logisticRegression_70 = load('logisticRegression_70')
    proposed_70 = load('proposed_70')

    data = {
        'SVM': svm_70,
        'XGBoost': xgboost_70,
        'DTree': dtree_70,
        'RF': rf_70,
        'KNN': knn_70,
        'Logistic Regression': logisticRegression_70,
        'PROPOSED': proposed_70
    }

    ind = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score', 'MCC', 'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']
    table = pd.DataFrame(data, index=ind)
    print('---------- Metrics for 70 training 30 testing ----------')
    print(table)

    table.to_excel('./Results/table_70.xlsx')

    val1 = np.array(table)

    # learning rate -  80  and 20

    svm_80 = load('svm_80')
    xgboost_80 = load('xgboost_80')
    dtree_80 = load('dtree_80')
    rf_80 = load('rf_80')
    knn_80 = load('knn_80')
    logisticRegression_80 = load('logisticRegression_80')
    proposed_80 = load('proposed_80')

    data1 = {
        'SVM': svm_80,
        'XGBoost': xgboost_80,
        'DTree': dtree_80,
        'RF': rf_80,
        'KNN': knn_80,
        'Logistic Regression': logisticRegression_80,
        'PROPOSED': proposed_80
    }

    ind =['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score', 'MCC', 'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']
    table1 = pd.DataFrame(data1, index=ind)
    print('---------- Metrics for 80 training 20 testing ----------')
    print(table1)

    val2 = np.array(table1)
    table1.to_excel('./Results/table_80.xlsx')

    metrices = [val1, val2]

    mthod = ['SVM', 'XGBoost', 'DTree', 'RF', 'KNN', 'Logistic Regression', 'Proposed']
    metrices_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score', 'MCC', 'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']

    # Bar plot
    for i in range(len(metrices_plot)):
        bar_plot(mthod, metrices[0][i, :], metrices[1][i, :], metrices_plot[i])

    learn_data = [70, 80]
    for k in learn_data:
        y_test = load(f'y_test_{k}')
        y_pred = load(f'predicted_{k}')
        densityplot(y_test, y_pred, k)
        classfi_report(y_test, y_pred, k)

        confu_plot(y_test, y_pred, k)

        precision_recall_plot(y_test, y_pred, k)

