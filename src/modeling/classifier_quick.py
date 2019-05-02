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
    target = 'state'
    
#     Drop unnecessary variables or future variables
    to_drop = ['id', 'backers_count', 'usd_pledged', 'pledge_per_backer', 'success_rate']
    df = df.drop(to_drop, axis=1)
    x = df.drop(target, axis=1)
    y = df[target]
    
#     One-hot encode the categorical data
    x = pd.get_dummies(x)
    
    return df, x, y

def plot_feature_importance(tree_model, x_train, filename):
    """
    Plot top 20 features that has the largest importance in the model.
    Params:
        tree_model: The fitted decision tree model.
        x_train: the independant variables X.
        filename: Path to the file where the image will be saved.
    Returns:
        Save image to filename.
    """
    fig, ax = plt.subplots()
    feat_importances = pd.Series(tree_model.feature_importances_, index=x_train.columns)
    feat_importances.nlargest(20).plot(kind='barh', ax=ax)
    ax.set_title('Top 20 Feature Importance')
    ax.set_xlabel('Feature Importance')
    fig.subplots_adjust(left=0.36, right = 0.97)
    plt.show()
    fig.savefig(filename, dpi=fig.dpi)

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
#     x = x.loc[150000:170000,:]
#     y = y.loc[150000:170000]
    
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
    print("Visualize the fitted decision tree: " )
    filename = os.path.join(imagesdir,'Decision_Tree')
    class_names = ['Failure', 'Success']
    np.asarray(class_names)
    dot_data = tree.export_graphviz(tree_model, out_file=None,
                                   feature_names=list(x),
                                   max_depth = 6,
                                   class_names=class_names,
                                   filled=True, rounded=True,  
                                   special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render(filename, format='png')
    print("Decision tree plot succesfully saved to file %s" % filename)
    
    # Plotting feature importance
    print("Plotting the top 20 Feature importances of the fitted decision tree: " )
    filename = os.path.join(imagesdir,'tree_feature_importance.png')
    plot_feature_importance(tree_model, x_train, filename)
    print("Feature importance plot succesfully saved to file %s" % filename)
    
    # 6 - Random Forest Classifier
    print("\n\n\nStep 5: Random Forest Classifier.")
    rf_default = RandomForestClassifier(n_estimators = 500)
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
    filename = os.path.join(imagesdir,'rf_feature_importance.png')
    plot_feature_importance(rf_default, x_train, filename)
    print("Feature importance plot succesfully saved to file %s" % filename)
    
    # 7 - Save different models obtained.
    print("\n\n\nStep 6: Save different models obtained.")
    filename = os.path.join(datadir, 'naive_bayes_classifier.pkl')
    store_model(gnb, filename)
    print("Model Naive Bayes succesfully saved to %s" % filename)
    filename = os.path.join(datadir, 'logistic_regression_classifier.pkl')
    store_model(log_reg, filename)
    print("Model Logistc Regression succesfully saved to %s" % filename)
    filename = os.path.join(datadir, 'decision_tree_classifier.pkl')
    store_model(tree_model, filename)
    print("Model Decision Tree succesfully saved to %s" % filename)
    filename = os.path.join(datadir, 'random_forest_classifier.pkl')
    store_model(rf_default, filename)
    print("Model Random Forest succesfully saved to %s" % filename)
    
if __name__ == "__main__":
    main()    
