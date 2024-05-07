from datetime import datetime
import numpy as np
import pandas as pd
import ast  # Module for literal_eval
import urllib.request
from PIL import Image
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree  import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import string
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def data_report(df:pd.DataFrame):
    """
    Function that returns a DataFrame of the information of column names, type of variable, missings and unique values

    Arguments:

        `df` (DataFrame): Variable dataframe de Pandas.


    Returns:
        concatenado: DataFrame
    """
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)


    return concatenado.T

def is_english(text:str):
    """
    Function returns True or False if contains characters in English

    Arguments:

        `text` (str): Variable dataframe de Pandas.


    Returns:
        True/False: Bool
    """
    # Define the English character set (including punctuation and spaces)
    english_chars = string.ascii_letters + string.digits + string.punctuation + ' '
    
    # Use a regular expression to find non-English characters
    if re.search('[^' + re.escape(english_chars) + ']', text):
        return False
    else:
        return True


def function_template(values):
    """
    Description

    Arguments:

        `values` (type): Variable dataframe de Pandas.


    Returns:
        variable: Type
    """
    pass

def convert_to_list(value:str):
    """
    Custom converter function to convert string representation of list to actual list

    Arguments:

        `value` (str): Text to convert to a list.


    Returns:
        value: List
    """
    # Custom converter function to convert string representation of list to actual list
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def onehotencoder(df:pd.DataFrame,features_cat:list):
    """
    Turns categorical columns in binary columns

    Arguments:

        `df` (pd.DataFrame): Dataframe de Pandas.
        `features_cat` (list): List of categorical columns.

    Returns:
        df: pd.DataFrame
    """
    #Turns categorical columns in binary columns
    onehot = OneHotEncoder(sparse_output=False, drop='first') 
    data = onehot.fit_transform(df[features_cat])
    new_features = onehot.get_feature_names_out()
    df[new_features] = data
    df.drop(columns= features_cat, axis = 1, inplace = True)
    return df

def scaler_of_x_train_and_x_test(X_train:pd.DataFrame,X_test:pd.DataFrame,minmax:bool=True):
        
        """
        Función que escala valores de un dataframe con MinMaxScaler o StandardScaler

        Args:
                `df` (pandas.DataFrame): DataFrame que contiene los datos.
                `minmax` (bool): Aplica MinMaxScaler por defecto, de lo contrario aplica StandardScaler
        
        Devuelve:
                `df` (pandas.DataFrame) normalizado
        """
        if minmax:
                minmax=MinMaxScaler()
                X_train_scaled=minmax.fit_transform(X_train)
                X_test_scaled=minmax.transform(X_test)
        else:
                standardscaler=StandardScaler()
                X_train_scaled=standardscaler.fit_transform(X_train)
                X_test_scaled=standardscaler.transform(X_test)
        return X_train_scaled,X_test_scaled

def to_lowercase_and_remove_blanks_on_columns_df(df:pd.DataFrame):
    """
    Change to lowercase and remove blanks from column names

    Arguments:

        `df` (pd.DataFrame): Dataframe de Pandas.


    Returns:
        df: pd.DataFrame
    """
    #Change to lowercase and remove blanks from column names
    keys=df.columns.to_list()
    new_columns={}
    for column in keys:
        value=column.lower()
        value=value.replace(" ","_")
        new_columns[column]=value
    #print(new_columns)
    df.rename(columns=new_columns,inplace=True)
    return df

def eda1_cat(df:pd.DataFrame,col_target:str,numeric_cols:list):

    
    # Assuming df is your DataFrame and 'target' is a categorical column in df

    # Histograms for all numeric features with target as hue

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df, x=col, hue=col_target, element='step', stat='density', common_norm=False)
        plt.title(f'Histogram of {col} by Target')
        plt.show()
    return plt
    
def eda2_cat(df:pd.DataFrame,col_target:str,numeric_cols:list):

    # Box plots for all numeric features with target as hue
    plt.figure(figsize=(12, len(numeric_cols) * 4))
    for i, col in enumerate(numeric_cols):
        plt.subplot(len(numeric_cols), 1, i + 1)
        sns.boxplot(x=col, y=col_target, data=df, orient='h')
        plt.title(f'Box Plot of {col} by Target')
    plt.tight_layout()
    plt.show()
    return plt

def eda3_cat(df:pd.DataFrame,col_target:str,numeric_cols:list):

    # Scatter plots for first few numeric features (example: first 3 columns) with target as hue
    if len(numeric_cols) > 1:
        sns.pairplot(data=df, vars=numeric_cols[:3], hue=col_target)
        plt.suptitle('Pair Plot of Numeric Features by Target')
        plt.show()


    return plt

def eda4_cat(df:pd.DataFrame,col_target:str,numeric_cols:list):

    # Correlation Matrix Heatmap of Numeric Features
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='viridis')
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()
    return plt

def eda1_num(df:pd.DataFrame,col_target:str,numeric_cols:list):


    # Assuming df is your DataFrame with a numeric 'target' column

    # Histograms and Box Plots for all numeric features
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.show()

        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col}')
        plt.show()
    return plt






def eda2_num(df:pd.DataFrame,col_target:str,numeric_cols:list):

        # Scatter Plots for numeric features against the target
    for col in numeric_cols:
        if col != col_target:
            plt.figure(figsize=(8, 4))
            sns.scatterplot(x=df[col], y=df[col_target])
            plt.title(f'Scatter Plot of {col} vs including Target column {col_target}')
            plt.show()
    return plt


def eda3_num(df:pd.DataFrame,col_target:str,numeric_cols:list):

        # Correlation Matrix Heatmap including the target
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='viridis')
    plt.title(f'Correlation Matrix of Numeric Features including Target column {col_target}')
    plt.show()
    return plt

def eda4_num(df:pd.DataFrame,col_target:str,numeric_cols:list):

        # Pair Plot for the first few numeric features (including target)
    if len(numeric_cols) > 1:
        sns.pairplot(data=df, vars=numeric_cols[:3].insert(0, col_target))
        plt.title(f'Pair Plot of Numeric Features including Target column {col_target}')
        plt.show()
    return plt

def remove_outliers(df:pd.DataFrame,columns:list, range):
    """
    Filter by percentiles to remove or to minimize outliers

    Arguments:

        `df` (pd.DataFrame): Pandas DataFrame
        `columns` (list): List of columns of the Dataframe

    Returns:
        df: pd.DataFrame
    """
    percentile_thresholds={}
    for column in columns:
        percentile_thresholds[column]=range
    
    # Filter outliers based on percentiles
    for col, (lower_percentile, upper_percentile) in percentile_thresholds.items():
            
            lower_bound = np.percentile(df[col], lower_percentile)
            upper_bound = np.percentile(df[col], upper_percentile)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

