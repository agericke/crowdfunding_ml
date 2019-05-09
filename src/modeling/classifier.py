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
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import graphviz
from sklearn import tree
from sklearn.tree import export_graphviz
import pickle
import configparser

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
    return dirname, datadir, imagesdir
    
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
    
    
def read_data_for_demo(x, demo_file, colnames, floatcols, intcols, strcols, catcols):
    config = configparser.ConfigParser()
    config.read(demo_file)
    demo_values = []
    for col in colnames:    
        value = config.get('demodata', col)
        demo_values.append(value)
        if col == 'usd_goal':
            value = float(value)
        elif col == '
        print("Column %s has values %s" %(col, str(value)))
        print("Type is : %s" % type(value))
    
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
    target = 'state'
    
#     Drop unnecessary variables or future variables
    to_drop = ['id', 'backers_count', 'usd_pledged', 'pledge_per_backer', 'success_rate']
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
#     x = x.loc[50000:70000,:]
#     y = y.loc[50000:70000]
    
    # Train/ test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22333)
    
    # No Information Rate in the data
    nir = max(y_train.mean(), 1-y_train.mean())
    print("No Information Rate : %s" %nir)
    
    # 3 - Naive Bayes Classifier
    print("\n\n\nStep 2: Naive Bayes Classifier.")
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
    print("Confusion matrix : %s" %cm )
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score : %s" %acc )
    prec = precision_score(y_test, y_pred)
    print("Precision score : %s" %prec )
    rec = recall_score(y_test, y_pred)
    print("Recall score : %s" %rec )
    
    # 4 - Logistic Regression Classifier
    print("\n\n\nStep 3: Logistic Regression Classifier.")
    log_reg = LogisticRegression(random_state=0)
    log_reg.fit(x_train, y_train)
    y_pred = log_reg.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
    print("Confusion matrix : %s" %cm )
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score : %s" %acc )
    prec = precision_score(y_test, y_pred)
    print("Precision score : %s" %prec )
    rec = recall_score(y_test, y_pred)
    print("Recall score : %s" %rec )
    
    # 5 - Decision Tree Classifier
    print("\n\n\nStep 4: Decision Tree Classifier.")
    tree_model = DecisionTreeClassifier() # Default decision tree
    tree_model.fit(x_train, y_train)
    y_pred = tree_model.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
    print("Confusion matrix : %s" %cm )
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score : %s" %acc )
    prec = precision_score(y_test, y_pred)
    print("Precision score : %s" %prec )
    rec = recall_score(y_test, y_pred)
    print("Recall score : %s" %rec )
    
    # Visualize the decision tree
    class_names = ['Failure', 'Success']
    np.asarray(class_names)
    dot_data = tree.export_graphviz(tree_model, out_file=None,
                                   feature_names=list(x),
                                   max_depth = 7,
                                   class_names=class_names,
                                   filled=True, rounded=True,  
                                   special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render("Decision Tree") 
    
    # 6 - Grid Search CV to find best decision tree
    print("\n\n\nStep 5: Best Decision Tree Classifier using GridSearchCV")
    param_grid = {'max_depth': [None, 5, 10, 15, 20], 'min_samples_leaf': [1, 2, 3, 5], 'min_samples_split': [2, 5, 10]}
    tree_classifier = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    tree_classifier.fit(x_train, y_train)
    y_pred = tree_classifier.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
    print("Confusion matrix : %s" %cm )
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score : %s" %acc )
    prec = precision_score(y_test, y_pred)
    print("Precision score : %s" %prec )
    rec = recall_score(y_test, y_pred)
    print("Recall score : %s" %rec )
    
    # Plotting feature importance
    print("Plotting the top 20 Feature importances of the fitted decision tree: " )
    feat_importances = pd.Series(tree_classifier.feature_importances_, index=x_train.columns)
    feat_importances.nlargest(20).plot(kind='barh')
    plt.show()
    
    # Visualize the decision tree
    dot_data = tree.export_graphviz(tree_classifier.best_estimator_, out_file=None, 
                      feature_names=list(x_train),  
                      class_names=class_names,
                      max_depth = 7,
                      filled=True, rounded=True,  
                      special_characters=True)  
    graph2 = graphviz.Source(dot_data)
    graph2.render("Best Decision Tree")
    
    # 7 - Random Forest Classifier
    print("\n\n\nStep 6: Random Forest Classifier.")
    rf_default = RandomForestClassifier()
    rf_default.fit(x_train, y_train)
    y_pred = rf_default.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
    print("Confusion matrix : %s" %cm )
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score : %s" %acc )
    prec = precision_score(y_test, y_pred)
    print("Precision score : %s" %prec )
    rec = recall_score(y_test, y_pred)
    print("Recall score : %s" %rec )
    
    print("Plotting top 20 Feature importances of the fitted Random Forest: " )
    feat_importances = pd.Series(rf_default.feature_importances_, index=x_train.columns)
    feat_importances.nlargest(20).plot(kind='barh')
    plt.show()
    
    # 8 - Grid Search CV to do hyperparameter tuning for Random Forest
    print("\n\n\nStep 7: Best Random Forest Classifier using GridSearchCV")
    param_grid_rf = {'n_estimators': [10, 50, 100, 500], 'max_depth': [None, 5, 10, 15, 20], 
                     'min_samples_leaf': [1, 2, 3, 5], 'min_samples_split': [2, 5, 10]}
    rf_best = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5) # default: 3-fold cross validation
    rf_best.fit(x_train, y_train)
    y_pred = rf_best.predict(x_test)
    
    print("The best settings for Random Forest is: %s" %rf_best.best_params_)
    
    cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
    print("Confusion matrix : %s" %cm )
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score : %s" %acc )
    prec = precision_score(y_test, y_pred)
    print("Precision score : %s" %prec )
    rec = recall_score(y_test, y_pred)
    print("Recall score : %s" %rec )
    
    # 9 - Save different models obtained.
    print("\n\n\nStep 8: Save different models obtained.")
    filename = os.path.join(datadir, 'naive_bayes_classifier.pkl')
    store_model(gnb, filename)
    print("Model Naive Bayes succesfully saved to %s" % filename)
    filename = os.path.join(datadir, 'logistic_regression_classifier.pkl')
    store_model(log_reg, filename)
    print("Model Logistc Regression succesfully saved to %s" % filename)
    filename = os.path.join(datadir, 'decision_tree_classifier.pkl')
    store_model(tree_classifier.best_estimator_, filename)
    print("Model Decision Tree succesfully saved to %s" % filename)
    filename = os.path.join(datadir, 'random_forest_classifier.pkl')
    store_model(rf_best.best_estimator_, filename)
    print("Model Random Forest succesfully saved to %s" % filename)
    
if __name__ == "__main__":
    main()    