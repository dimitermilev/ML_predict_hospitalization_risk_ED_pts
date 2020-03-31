from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

def graph_roc(data, features, optimal_feature_num, model, title):
    '''Build a ROC AUC curve graph with matplotlib'''
    X_train, X_train_sc, X_test, X_test_sc, y_train, y_test = split_data(data, features, optimal_feature_num)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr,lw=2)
    plt.plot([0,1],[0,1],c='violet',ls='--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve');
    print(f"{title} ROC AUC score = ", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

def make_confusion_matrix(data, features, optimal_feature_num, model, threshold=0.5):
    '''Build confusion matrix with seaborn. Leverage the predict_proba function of models to adjust threshold'''
    X_train, X_train_sc, X_test, X_test_sc, y_train, y_test = split_data(data, features, optimal_feature_num)
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    hosp_confusion = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=80)
    sns.heatmap(hosp_confusion, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels=['Discharged', 'Hospitalized'],
           yticklabels=['Discharged', 'Hospitalized']);
    plt.xlabel('prediction')
    plt.ylabel('actual')

def make_precision_recall_curves(data, features, optimal_feature_num, model):
    '''Plot precision and recall curves for desired model'''
    X_train, X_train_sc, X_test, X_test_sc, y_train, y_test = split_data(data, features, optimal_feature_num)
    precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1] )
    plt.figure(dpi=80)
    plt.plot(threshold_curve, precision_curve[1:],label='precision')
    plt.plot(threshold_curve, recall_curve[1:], label='recall')
    plt.legend(loc='lower left')
    plt.xlabel('Threshold (above this probability, label as hospitalization)');
    plt.title('Precision and Recall Curves');

