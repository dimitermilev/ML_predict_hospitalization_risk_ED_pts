from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, make_scorer
import operator

def get_feature_importance(data, maxfeat):
    '''Use Random Forest to assess information gain (Gini index) for each of the ~1,000 dataset features'''
    X_train, X_train_sc, X_test, X_test_sc, y_train, y_test = split_data(data,'','')
    '''Get all feature labels'''
    feat_labels = list(X_train.columns)
    '''Initiate and train Random Forest classifier'''
    clf = RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, n_jobs=-1)
    clf.fit(X_train, y_train)
    '''Create a list of feature importances and append the Gini scores of each feature to the list'''
    feat_select = []
    for feature in zip(feat_labels, clf.feature_importances_ ):
        feat_select.append(feature)
    df_feat = pd.DataFrame(feat_select, columns=["feature", "gini"]).sort_values(by='gini', ascending=False)
    return df_feat[0:maxfeat]

def hyper_parameter_tune_svm(data, features, features_num):    
    '''Tune hyperparameters of an SVM Model for optimal performance on test data'''
    X_train, X_train_sc, X_test, X_test_sc, y_train, y_test = split_data(data, '','')
    '''Use cross validation to assess performance by hyer parameter'''
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=47)
    '''Grid search of SVM hyper parameters'''
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    '''Scoring methods for hyperparamaters'''
    scoringmethods = ['f1','accuracy','precision', 'recall','roc_auc']
    for score in scoringmethods:
        print(f"Hyper-parameter tuning for best {score}")
        svmclf = GridSearchCV(SVC(C=1), parameters, cv=kf, scoring=score, n_jobs=-1, verbose = 1)
        svmclf.fit(X_train_sc, y_train)        
    print("Classification report:")
    y_pred = svmclf.predict(X_test_sc)
    print(classification_report(y_test, y_pred))
    print("Best model:")
    print(svmclf.best_estimator_)
    return svmclf.best_estimator_

def optimal_feature_count(feat_count_scores):
    '''Finds the optimal number of features to include based on three models' aggregate performance'''
    feat_count_scores_pv = feat_count_scores.pivot(index='features', columns='model', values='score')
    feat_count_scores_pv.plot()
    feat_count_scores_pv['model_agg'] = (feat_count_scores_pv['Logistic']+feat_count_scores_pv['RF']+feat_count_scores_pv['SVC'])/3
    optimal_count_index, optimal_count_performance = max(enumerate(scores_pv['model_agg'].tolist()), key=operator.itemgetter(1))
    return optimal_count_index

