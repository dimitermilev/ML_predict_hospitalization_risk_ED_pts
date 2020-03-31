from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer, roc_auc_score, roc_curve
import pickle 

def run_classifier_models(model_names, data, features, optimal_feature_num):
    '''Create and train three classifier model candidates, using hyperparameter tuned SVM and optimal number of features'''
    lr = LogisticRegression(max_iter=1000)
    svm = svc
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=.25, n_jobs=-1)
    X_train, X_train_sc, X_test, X_test_sc, y_train, y_test = split_data(data, features, optimal_feature_num)
    '''Train and pickle the models'''
    trained_models = []
    for model_name in model_names:
        curr_model = eval(model_name)
        if model_name!="svc":
            curr_model.fit(X_train, y_train)
            print(f'{model_name} accuracy score: {curr_model.score(X_test, y_test)}')
            print('{} auc score: {}'.format(model_name, roc_auc_score(y_test, curr_model.predict_proba(X_test)[:,1])))
        else:
            curr_model.fit(X_train_sc, y_train)
            print(f'{model_name} accuracy score: {curr_model.score(X_test_sc, y_test)}')
            print('{} auc score: {}'.format(model_name, roc_auc_score(y_test, curr_model.predict_proba(X_test_sc)[:,1])))
    '''Save models to '''
        with open(f"models/{model_name}.pickle", "wb") as pfile:
            pickle.dump(curr_model, pfile)    
    return

    def ensemble_tuned_trained_models(model_names, data, features, optimal_feature_num):
    '''Load pre-trained, tuned, pickled models and implement them in a voting classifier'''
    X_train, X_train_sc, X_test, X_test_sc, y_train, y_test = split_data(data, features, optimal_feature_num)
    '''Model loads from pickled files'''
    activated_models = []
    for model_name in model_names:
        with open(f"models/{model_name}.pickle", "rb") as pfile:
            exec(f"{model_name} = pickle.load(pfile)")
            print(model_name)
            print(eval(model_name).score(X_train, y_train))
            activated_models.append(eval(model_name))
    model_list = list(zip(model_names, activated_models))
    print(model_list)
    '''Initiate and train voting classifier model'''
    voting_model = VotingClassifier(estimators=model_list,
                                    voting='soft', 
                                    n_jobs=-1,
                                    weights = [4,5,3]
                                    )    
    voting_model.fit(X_train, y_train)
    '''Model score: get final model accuracy and auc for voting model'''
    y_pred = voting_model.predict(X_test)
    accuracy_score(y_test, y_pred)
    print(f'Voting Model accuracy score: {accuracy_score(y_test, y_pred)}')
    print('Voting Model auc score: {}'.format(roc_auc_score(y_test, voting_model.predict_proba(X_test)[:,1])))
    return voting_model

