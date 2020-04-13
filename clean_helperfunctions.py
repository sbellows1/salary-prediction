import numpy as np
import pandas as pd


def find_outlier_bounds(df, col):
    '''Takes a dataframe and a column and returns the bounds of the Tukey rule
    for outliers (25% - 1.5IQR and 75% + 1.5IQR).'''
    descript = df[col].describe()
    IQR = descript['75%'] - descript['25%']
    high = descript['75%'] + 1.5 * IQR
    low = descript['25%'] - 1.5 * IQR
    return low, high

def drop_cols(df, cols):
    '''Takes a dataframe and a list of columns to drop, drops those columns
    from the dataframe.'''

    for col in cols:
        df.drop(col, axis = 1, inplace = True)
    return df

def remove_outliers(df, outlier_col, low = None, high = None):
    '''Takes a dataframe, the column to remove outliers from, and the high and
    low bounds for outliers. Removes any values below the low value and above
    the  high value'''

    if low != None:
        df = df.loc[df['salary'] > low]
    if high != None:
        df = df.loc[df['salary'] > high]
    return df

def set_numeric(df, numeric_cols):
    '''Takes a dataframe and a list of columns that should be numeric and
    makes those columns numeric'''

    for col in numeric_cols:
        df.loc[:,col] = pd.to_numeric(df.loc[:,col])
    return df

def set_categorical(df, cat_cols):
    '''Takes a dataframe and a list of columns that should be categorical and
    makes those columns categorical'''

    for col in cat_cols:
        df.loc[:,col] = df.loc[:,col].astype('category')
    return df

def impute_mean(df, numeric_cols):
    '''Takes a dataframe and a list of numeric columns, fills NA in the
    numeric columns with the mean value'''

    df.loc[:,numeric_cols].fillna(df.loc[:,numeric_cols].mean(), inplace = True)
    return df

def impute_mode(df, cat_cols):
    '''Takes a dataframe and a list of categorical columns, fills NA in the
    categorical columns with the mode value'''

    df.loc[:,cat_cols].fillna(df.loc[:,cat_cols].mode(), inplace = True)
    return df
