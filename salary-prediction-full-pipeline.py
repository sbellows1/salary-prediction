##This is an end to end script that loads and cleans the data,
##builds 5 models and compares them, and then chooses the best model
##to make predictions and save to a PostgreSQL database.

__author__ = 'Sam Bellows'
__email__ = 'sbellows1@gmail.com'
__website__ = 'www.github.com/sbellows1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clean_helperfunctions import *
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import psycopg2
from sqlalchemy import create_engine
from logincreds import grab_creds

def load_data(filepath):
    '''loads a csv from the given filepath'''
    return pd.read_csv(filepath)

def merge_dfs_delete_originals(df1, df2, key = None, left_index = False, right_index = False):
    '''Merges the two input dataframes and then deletes them. Merges on the
    given key. left_index and right_index show which index to keep if any.'''
    merge_df = df1.merge(df2, on = key, left_index = left_index, right_index = right_index)
    del df1
    del df2
    return merge_df

def RMSE(actual, predictions):
    '''Calculates RMSE for the given values and predictions'''
    return np.sqrt(((actual - predictions)**2).mean())

def split_features_target(df, target):
    '''Takes a dataframe and a target column, returns a dataframe of features
    and a Series containing the target'''
    target_series = df.pop(target)
    return df, target_series

def one_hot_encode(df, cat_cols):
    '''Takes a dataframe and a list of categorical columns. Drops the
    categorical columns and replaces them with one hot encoded versions of
    said columns'''
    df = pd.concat([df, pd.get_dummies(df[cat_cols])], axis = 1)
    df.drop(cat_cols, axis = 1, inplace = True)
    return df

def make_baseline(df, feature, target):
    '''Takes a dataframe, single feature, and target feature and returns a
    baseline RMSE for the linear model based on the single feature'''
    lm = LinearRegression()
    lm.fit(np.array(df[feature]).reshape(-1, 1), target)
    preds = lm.predict(np.array(df[feature]).reshape(-1, 1))
    return RMSE(target, preds)

def train_model(model, features, targets, rmse_dict, cv=5, n_jobs = -1, scorer = None):
    '''Takes a model, feature and target dataframes, dictionaries to record
    mean and standard deviation of RMSE, the number of folds, and the
    scorer. Trains the model with cross_validation, then records the mean and
    standard deviation for the RMSE'''
    cv_score = cross_val_score(model, features, targets, cv = cv, n_jobs = n_jobs, scoring = scorer)
    rmse_dict[model] = -1.0*cv_score.mean()

def create_feature_importance_df(model, features):
    '''Takes a model and returns a dataframe displaying the feature importances
    of said model.'''

    if hasattr(final_model, 'feature_importances_'):
        feature_importances = final_model.feature_importances_
    else:
        feature_importances = [0] * len(features.columns)

    feature_importances = np.array(feature_importances)
    feature_importances = pd.DataFrame({'features': features.columns,
                                        'importances': feature_importances})
    feature_importances.set_index('features', inplace = True)
    feature_importances.sort_values('importances', ascending = False,
                                    inplace = True)
    return feature_importances

def make_predictions(model, test_data):
    '''Takes a model, test data, and a cleaner of class ImputeEncode and
    transforms the test data before predicting on the test data.'''
    predictions = model.predict(test_data)
    predictions = pd.Series(predictions)
    return predictions

def display_results(model, predictions, feature_importances):
    '''Displays the final model, the first few predictions, and the feature
    importances given by said model'''
    print(str(model))
    print(predictions.head(10))
    print(feature_importances)

def save_results(model, predictions, feature_importances, rmse_df):
    '''Saves the model into a txt file and the predictions and feature
    importances into a postgreSQL database. Also saves a plot of the
    feature importances and RMSE values.'''
    with open ('final_model.txt', 'w') as f:
        f.write(str(model))
    db_user, db_password, db_server, db_name = grab_creds()
    conn_string = 'postgresql://'+db_user+':'+db_password+'@'+db_server+':5432/'+db_name
    engine = create_engine(conn_string)
    predictions.to_sql('predictions', engine, if_exists = 'replace')
    feature_importances.to_sql('feature_importances', engine, if_exists = 'replace', method = 'multi')
    rmse_df.to_sql('rmse', engine, if_exists = 'replace', method = 'multi')
    plot = rmse_df.plot.bar()
    fig = plot.get_figure()
    fig.savefig('model_RMSE.png')
    plot = feature_importances.iloc[0:20].plot.bar()
    fig = plot.get_figure()
    fig.savefig('feature_importances.png')

class Normalizer(BaseEstimator, TransformerMixin):
    '''Normalizes columns in a feature dataframe'''
    def __init__(self):
        pass
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X.apply(lambda x: (x - min(x))/(max(x) - min(x)))

class ImputeEncode(BaseEstimator, TransformerMixin):
    '''Goes through all the cleaning processes for this particular analysis'''
    def __init__(self, numeric_cols, cat_cols, outlier_low=None,
                 outlier_high=None):
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.outlier_low = outlier_low
        self.outlier_high = outlier_high
    def fit(self, X, y= None):
        return self
    def transform(self, X, y=None):
        X = set_numeric(X, self.numeric_cols)
        X = set_categorical(X, self.cat_cols)
        X = impute_mean(X, self.numeric_cols)
        X = impute_mode(X, self.cat_cols)
        X = one_hot_encode(X, self.cat_cols)
        return X

if __name__ == '__main__':

    ##Load the data
    train_features = load_data('data/train_features.csv')
    test_features = load_data('data/test_features.csv')
    train_target = load_data('data/train_salaries.csv')

    ##ID the columns
    drop_col_names = ['jobId', 'companyId']
    outlier_col = 'salary'
    numeric_cols = ['yearsExperience', 'milesFromMetropolis']
    cat_cols = ['jobType', 'degree', 'major', 'industry']

    ##Merge Train and train_target
    train_full = merge_dfs_delete_originals(train_features, train_target)

    ##Clean train data
    train_full_clean = remove_outliers(train_full, outlier_col, low = 0)
    train_full_clean = drop_cols(train_full, drop_col_names)

    ##Split out target into its own series
    features, target = split_features_target(train_full_clean, 'salary')

    #Make baseline
    baseline_RMSE = make_baseline(features, 'yearsExperience', target)

    ##Create scoring object
    RMSE_scorer = make_scorer(RMSE, greater_is_better = False)

    ##Initialize ImputeEncode instance to transform training data
    cleaner = ImputeEncode(numeric_cols, cat_cols)
    features_transform = cleaner.transform(features)

    ##Initialize models
    ##Hyperparameters previously tuned with a gridsearch
    enet = ElasticNet()

    pca_enet_pipe = Pipeline([('scale', StandardScaler()),
                              ('pca', PCA()),
                              ('en', ElasticNet())])

    knn_pipe = Pipeline([('norm', Normalizer()),
                       ('knn', KNeighborsRegressor(n_neighbors = 25,
                                                   weights = 'uniform'))])

    rf = RandomForestRegressor(n_estimators = 150, max_depth = 10,
                               max_features = 0.5)
    gb = GradientBoostingRegressor(n_estimators = 150, learning_rate = 0.1,
                                   max_depth = 5, max_features = 0.75)

    model_list = []
    RMSE_dict = {}

    model_list.extend([enet, pca_enet_pipe, knn_pipe, rf, gb])
    model_list.extend([gb])
    ##Train models
    for model in model_list:
        train_model(model, features_transform, target, RMSE_dict,
                    n_jobs = 4, scorer = RMSE_scorer)
    ##Pick model with lowest RMSE
    final_model = min(RMSE_dict, key = RMSE_dict.get)
    print(RMSE_dict)
    ##Refit model on entire training set
    final_model.fit(features_transform, target)

    ##prepare test set
    test_features = drop_cols(test_features, drop_col_names)
    test_transform = cleaner.transform(test_features)

    #Make predictions and generate feature importances
    predictions = make_predictions(final_model, test_transform)
    predictions = pd.Series(np.array(predictions))
    feature_importances = create_feature_importance_df(final_model, features_transform)

    #Create dictionary pairing models with RMSE
    models = ['baseline', 'linear', 'linear with PCA', 'random forest',
              'gradient boosting', 'knn']
    RMSE_vals = [baseline_RMSE, RMSE_dict[enet], RMSE_dict[pca_enet_pipe],
                 RMSE_dict[rf], RMSE_dict[gb], RMSE_dict[knn_pipe]]
    RMSE_df = pd.DataFrame({'model':models, 'RMSE':RMSE_vals}).set_index('model').sort_values('RMSE', ascending = False)

    #Display and save results
    display_results(final_model, predictions, feature_importances)
    save_results(final_model, predictions, feature_importances, RMSE_df)
