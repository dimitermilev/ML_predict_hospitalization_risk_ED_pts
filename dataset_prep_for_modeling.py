import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def one_hot_encode_db(data):
    '''Transform categorical fields into one hot encoded dummy variables'''
    data_one_hot_cats = pd.get_dummies(data, drop_first=True)
    return data_one_hot_cats

def impute_missing_vals(data): 
    '''Impute missing values with column means'''
    data_mean_impute = data.fillna(data.mean())
    return data_mean_impute

def split_data(data, features, feat_num):
    '''Function to split dataset intro train and test variables, and to scale the dataset features'''
    if len(features)>0:
        X = data[list(features[:feat_num]['feature'])]
    else:
        X = data.drop(['disposition_int'], axis=1)
    y = data['disposition_int']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    X_train_sc, X_test_sc = preprocessing.scale(X_train), preprocessing.scale(X_test)
    return X_train, X_train_sc, X_test, X_test_sc, y_train, y_test