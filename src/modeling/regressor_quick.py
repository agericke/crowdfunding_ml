import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import os, sys
import datetime
from datetime import date
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
#import graphviz
from sklearn import tree
from sklearn.tree import export_graphviz
import pickle

def initial_setup():
    """
    Create Initial setup of directories variables, and dataframe vars to use.
    Returns:
      A tuple containing:
          - datadir:   Absolute Path to the data directory of the project.
          - dirname:   Absolute Path of directory that contains this file.
          - imagesdir: Absolute path of directory that contains the images.
          - colnames: A list containing the initial colnames of the dataframe.
    """
    # Initial directories set up
    dirname = os.path.dirname(os.path.abspath('__file__'))
    datadir =  os.path.join(os.path.abspath(os.path.join(os.path.join(dirname, os.pardir), os.pardir)), 'data')
    imagesdir =  os.path.join(os.path.abspath(os.path.join(dirname, os.pardir)), 'images')
    initial_colnames = sorted(['backers_count', 'blurb', 'category', 'country', 'created_at', 'state_changed_at', 'currency', 'deadline', 'goal', 'id', 'launched_at', 'location', 'name', 'pledged', 'state', 'static_usd_rate', 'usd_pledged'])
    return dirname, datadir, imagesdir, initial_colnames

def read_from_disk(filename):
    """
    Read a dataframe from a filename in disk.
    Params:
        filename....Path to the file.
    Returns:
        A pandas dataframe.
    """
    return pickle.load(open(filename, 'rb'))

def store_model(model, filename):
    """
    Store the model using pickle.
    Params:
        model...trained ML model to store.
        filename....Path to the file to store the model in.
    Returns:
        Nothing.
    """
    pickle.dump(model, open(filename, 'wb'))

def prepare_data_for_ML(df):
    """
    Prepare data for ML analysis and algorithms
    Params:
        dataframe...Pandas dataframe.
    Returns:
        A tuple containing:
            - A pandas dataframe ready for ML analysis.
            - X: x variables centered and normalized.
            - y: objective variable.
        
    """
#     Create the target variable, in this case 'state' of a project
    target = 'success_rate'
    
#     Drop unnecessary variables or future variables
    to_drop = ['id', 'backers_count', 'usd_pledged', 'pledge_per_backer', 'state']
    df = df.drop(to_drop, axis=1)
    x = df.drop(target, axis=1)
    y = df[target]
    
#     One-hot encode the categorical data
    x = pd.get_dummies(x)
    
    return df, x, y

def main():
    print("Step 0: Initial directories and colnames set up")
    dirname, datadir, imagesdir, initial_colnames = initial_setup()
        
    # 1 - Load from disk the complete Merged Dataframe.
    print("\n\n\nStep 1: Load from disk the complete Merged and Cleaned Dataframe.")
    filename = os.path.join(datadir, 'formatting_intermediate_data2.pkl')
    print("Complete Dataframe is read from file %s" % filename)
    data = read_from_disk(filename)
    # Print summary of dataframe
    print("Dataframe contains %d projects and %d columns for each project\n" % (data.shape[0], data.shape[1]))
    
    # 2 - Process data frame for machine learning
    df, x, y = prepare_data_for_ML(data)
#     x = x.loc[170000:185000,:]
#     y = y.loc[170000:185000]
    
    # 3 - Train/ test split
    print("\n\n\nStep 2: Process data + Train/Test split")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22333)
    
    # 4 - Decision Tree Regressor
    print("\n\n\nStep 3: Decision Tree Regressor")
    reg_tree = DecisionTreeRegressor() # Default decision tree
    reg_tree.fit(x_train, y_train)
    y_pred = reg_tree.predict(x_test)
    
    ev = explained_variance_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("Explained variance score : %s" %ev)
    print("Root mean squared error score : %s" %rmse)
    print("R^2 score : %s" %r2)
    
    # 5 - Random Forest Regressor
    print("\n\n\nStep 4: Random Forest Regressor")
    rf_reg = RandomForestRegressor(n_estimators = 500)
    rf_reg.fit(x_train, y_train)
    y_pred = rf_reg.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    
    print("Explained variance score : %s" %ev)
    print("Root mean squared error score : %s" %rmse)
    print("R^2 score : %s" %r2)
    
    # 6 - Save different models obtained.
    print("\n\n\nStep 5: Save trained models")
    filename = os.path.join(datadir, 'decision_tree_regressor.pkl')
    store_model(reg_tree, filename)
    print("Model Decision Tree succesfully saved to %s" % filename)
    filename = os.path.join(datadir, 'random_forest_regressor.pkl')
    store_model(rf_reg, filename)
    print("Model Random Forest succesfully saved to %s" % filename)
    
if __name__ == "__main__":
    main()