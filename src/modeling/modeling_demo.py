#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:49:36 2019

@author: agericke
"""

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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import graphviz
from sklearn import tree
from sklearn.tree import export_graphviz
import pickle
from collections import Counter
import json

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


def read_models(datadir):
    filename = os.path.join(datadir, 'logistic_regression_classifier.pkl')
    #print("Logistic Model is read from is read from file %s" % filename)
    log_reg_model = read_from_disk(filename)
    
    filename = os.path.join(datadir, 'naive_bayes_classifier.pkl')
    #print("Naive Bayes Model read from read from file %s" % filename)
    nb_model = read_from_disk(filename)
    
#    filename = os.path.join(datadir, 'random_forest_classifier.pkl')
#    print("Random Forest Model read from read from file %s" % filename)
#    rf_model = read_from_disk(filename)
    
    filename = os.path.join(datadir, 'decision_tree_classifier.pkl')
    #print("Decision Tree Model read from read from file %s" % filename)
    dt_model = read_from_disk(filename)
    
    return log_reg_model, nb_model, dt_model
    

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


def prepare_data(df):
    
    #     Create the target variable, in this case 'state' of a project
    target = 'state'
    
    #     Drop unnecessary variables or future variables
    to_drop = ['id', 'backers_count', 'usd_pledged', 'pledge_per_backer', 'success_rate']
    df = df.drop(to_drop, axis=1)
    #df = df.iloc[0,:]
    x_no_dumm = df.drop(target, axis=1)
    y = df[target]
    
    #     One-hot encode the categorical data
    x = pd.get_dummies(x_no_dumm)
    
    return df, x, y, x_no_dumm
    
    
def create_vector_x(df, x_values):
    target = 'state'
    #df.append(x_values)
    x_no_dumm = df.drop(target, axis=1)
    x_no_dumm.append(x_values)
    #     One-hot encode the categorical data
    x = pd.get_dummies(x_no_dumm)
    
    return x.iloc[[-1]]


def read_demo_values(filename):
    with open(filename, 'r') as f:
        values = json.load(f)
    print("Values are:")
    for key, value in values.items():
        print("%s: %s" % (key, value))
    return pd.Series(values).to_frame().T
    

def main():
    
    print("Step 0: Initial directories and colnames set up")
    dirname, datadir, imagesdir, initial_colnames = initial_setup()
    
    filename = os.path.join(datadir, 'formatting_intermediate_data2.pkl')
    data = read_from_disk(filename)
    df, x, y, x_no_dumm = prepare_data(data)
    
    print("\n\n\n")
    demo = read_demo_values('demodata.json')
    
    demo_x = create_vector_x(df, demo)
    
    
    # 1 - Load from disk the complete Merged Dataframe.
    print("\n\n\nStep 1: Load from disk the classifiers.")
    log_reg_model, nb_model, dt_model = read_models(datadir)
    # Print summary of dataframe
    
    # 3 - Naive Bayes Classifier
    y_pred = nb_model.predict(demo_x)
    print("Naive Bayes Classifier predicts: ")
    print(y_pred[0])
    
    # 3 - Naive Bayes Classifier
    y_pred = log_reg_model.predict(demo_x)
    print("Logistic Regression Classifier predicts: ")
    print(y_pred[0])
    
     # 3 - Naive Bayes Classifier
    y_pred = dt_model.predict(demo_x)
    print("Decision Tree Classifier predicts: ")
    print(y_pred[0])
    
    
    
if __name__ == "__main__":
    main()    
