#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 01:54:00 2019

@author: agericke
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import os, sys
import datetime
from datetime import date
import time
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from ggplot import *
#%matplotlib inline


plt.rcParams.update({'font.size': 22})

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


def store_dataframe(dataframe, filename):
    """
    Store the dataframe using pickle.
    Params:
        dataframe...pandas dataframe to store.
        filename....Path to the file to store the datafram in.
    Returns:
        Nothing.
    """
    pickle.dump(dataframe, open(filename, 'wb'))
    

def check_missing_values_and_drop(data, drop=False):
    """
    Check the number of missing values that we have. Notice that this function
    will count 
    Params:
        data....Dataframe to check the missing values.
        drop....Boolean to indicate if we want to drop missing values or not.
    Returns:
        Prints a summary of the number and % of missing values.
        The dataframe with no missing values
    """
    #dataMis=data[data.isnull().any(axis=1)]
    #print(dataMis.groupby('country').country.count())
    total_rows = data.shape[0]
    na_col_counts = data.isna().sum()
    for idx in na_col_counts.index:
        na_number = na_col_counts[idx]
        print("Total number of NA values in column %s is %d" % (str(idx), na_number))
        pct_number = (na_number/total_rows) * 100
        print("Percentage of Na in column %s is %.4f %%\n" % (str(idx), pct_number))
    if drop:
        data = data.dropna()
    return data
    # TODO: See if we can check the missing indexes for each column and run a study on them.
    # TODO: Run experiments to try to identify is the missing values are mainly because of a reason or one type of project, or specific to one period of time (see if they are missing at random, missing not at random...)
    

def create_new_vars(data):
    """
    Create a new columns from actual columns, this new columns are:
        - usd_goal: Goal of the project in USD.
        - duration: Contains the dyas between the launching date and the deadline.
        - duration_until_lanched: Tracks the # of days between the created day and the launched date.
    Params:
        data.....Dataframe to create the usd_goal column from.
    Returns:
        A new dataframe with the usd_goal columns and the goal column erased.
    """
    #Create usd_goal, where we convert the goal into usd
    data["usd_goal"] = data["goal"]*data["static_usd_rate"]
    #Drop the original goal feature
    data.drop("goal", inplace=True, axis=1)
    
    # Create duration column. We get a var in seconds.
    data["duration"] = data["deadline"]-data["launched_at"]
    #Transform the variable to days
    data["duration"] = data["duration"]/(60*60*24)
    #We get a variable in seconds
    data["days_until_launched"] = data["launched_at"]-data["created_at"]
    #Transform the variable to days
    data["days_until_launched"] = data["days_until_launched"]/(60*60*24)
    print("Succesfully created usd_goal, duration and days_until launched columns.\n")
    return data


def create_new_date_cols(data, col_from, col_format, new_column, axis=1):
    """
    Create a new column variable in date format from a specific column, and the output specified
    by the format (i.e %Y for complete year).
    Params:
        data...........Dataframe to create the new column and obtain the columns to create the new one.
        col_from.......Specific column to obtain the data to convert to date type.
        col_format.....The format the the date type column will have.
        new_column.....Name of the new column.
        axis...........To specify if we want to apply function to rows or columns. (By default rows (1))
    Returns:
        A new dataframe with the new column added.
    """
    data[new_column] = data.apply(lambda row: time.strftime(col_format, time.localtime(row[col_from])), axis=axis)
    print("Succesfully created %s column, from column %s with format %s\n" % (new_column, col_from, col_format))
    return data


def convert_to_date(data, col, axis=1):
    """
    Convert an existing column to timestamp date format.
    Params:
        data...........Dataframe to create the new column and obtain the columns to create the new one.
        col............Name of the column we want to convert to date.
        axis...........To specify if we want to apply function to rows or columns. (By default rows (1))
    Returns:
        The dataframe with the specified column converted.
    """
    data[col] = data.apply(lambda row: datetime.datetime.fromtimestamp(row[col]), axis=axis)
    print("Succesfully converted to date type column %s\n" % col)
    return data


def plot_proyects_per_year(data, filename1, filename2):
    """
    Plot amount of proyects grouped by year and proyect state per year.
    Params:
        data.......Dataframe to plot the data from.
        filename...Path to the file where the image will be saved.
    Returns:
        Nothing. Saves image to filename.
    """    
    #fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(25,12))
    fig = plt.figure( figsize=(13,12))
    ax1 = plt.gca()
    data.groupby('year_launched').year_launched.count().plot(kind='bar', ax=ax1, color='green')
    plt.setp( ax1.xaxis.get_majorticklabels(), rotation=45 )
    ax1.set_title('Number of Projects per Year')
    ax1.set_xlabel('Year')
    fig.savefig(filename1, dpi=fig.dpi)
    
    fig = plt.figure( figsize=(13,12))
    ax2 = plt.gca()
    stateDistYear1 = pd.get_dummies(data.set_index('year_launched').state).groupby('year_launched').sum()
    #stateDistYear1.columns = ['failed', 'successful']
    stateDistYear1.plot(kind='bar', ax=ax2)
    plt.setp( ax2.xaxis.get_majorticklabels(), rotation=45 )
    ax2.set_title('State distribution of projects per year')
    ax2.set_xlabel('Year')
    fig.savefig(filename2, dpi=fig.dpi)


def remove_cols(dataframe, cols_to_remove):
    """
    Remove all the columns specified by the list form dataframe
    Params:
        cols_to_remove....List of columns we want to remove
        dataframe.........The dataframe to remove the columns from.
    Returns:
        A dataframe with only the columns we want.
    """
    dataframe.drop(cols_to_remove, inplace=True, axis=1)
    print("Succesfully removed columns %s" % cols_to_remove)
    return dataframe


def len_str_col(data, col, new_col):
    """
    Obtain length of a specific string column.
    Params:
        data......Dataframe.
        col.......Column we want to obatin the length from.
        new_col...Name of new column where to store result.
    Returns:
        Dataframe with the new column added.
    """
    data[new_col] = data[col].str.split().str.len()
    print("Succesfully created new column %s that contains length of column %s" % (new_col, col))
    return data


def print_data_summary(data):
    """
    Print summary of data.
    Params:
        data....Dataframe to print the summary from.
    Returns:
        A tuple containg:
            - unique values accross columns.
            - Summary of data per column. Prints in console summary.
    """
    #Lets look at the data. Printing summary of the kickstarter data not discarded
    print("Dataframe contains %d projects and %d columns for each project\n" % (data.shape[0], data.shape[1]))
    #This are the unique values across all columns. To determine which can be categorical variables.
    unique=data.nunique()
    print("\nUnique values summary:")
    print(unique)
    #Find out the columns data types
    print("\nDatatypes summary:")
    print(data.info())
    #Summary of the information of each column
    print("\nColumns summary:")
    summary=data.describe()
    print(summary)
    return unique, summary


def obtain_cat_and_subcat_for_row(row):
    """
    Obtain category and subcategory of a row. notice that if subcategory not present,
    this will set it to none.
    Params:
        row....The dataframe row.
    Returns:
        A tuple containing:
            - category: Value for obtained category.
            - subcategory: Value for obtained subcategory, and None if not present.
    """
    # Convert string to dict
    cat_dict = json.loads(row['category'])
    cat_split = cat_dict['slug'].split("/")
    # Save category and subcategory
    category = cat_split[0]
    subcategory = ""
    if len(cat_split)>1:
        # There is a sucategory
        subcategory = cat_split[1]
    return (category, subcategory)
    

def create_cat_and_subcat(data):
    """
    Save in a new column the category and subcategory extracted from the category column.
    Params:
        data.....Dataframe.
    Returns:
        A dataframe with category and subcategory vars created.¡ for each row.
    """
    cat_subcat = data.apply(obtain_cat_and_subcat_for_row, axis=1)
    data["main_category"] = [c[0].lower().replace(" ","") for c in cat_subcat]
    data["sub_category"] = [c[1].lower().replace(" ","") for c in cat_subcat]
    data.drop("category", inplace=True, axis=1)
    print("Succesfully created columns category and subcategory")
    return data
    

def create_category_2(data):
    """
    Create category column with data from the dataframe.
    Params:
        data....The dataframe.
    Returns:
        A dataframe containing the main_category column and with category column removed.
    """
    data['main_category'] = data['category'].apply(lambda x: x.split("slug",1)[-1].split("\"")[2].split("/")[0])
    #data['sub_category'] = data['category'].apply(lambda x: x.split("slug",1)[-1].split("\"")[2].split("/")[-1])
    # We drop the category variable since we don't need it anymore.
    data.drop("category", inplace=True, axis=1)


def obtain_location_vars_for_row(row):
    """
    Obtain country, state and type of location.
    Params:
        row....The dataframe row.
    Returns:
        A tuple containing:
            - country: Country oobtained.
            - state: State obtained.
            - loc_type: Type of location obtained.
    """
    # Convert string to dict
    loc_dict = json.loads(row['location'])
    country = loc_dict['country']
    state = loc_dict['state']
    loc_type =  loc_dict['type']
    return (country, state, loc_type)
    

def create_location_vars(data):
    """
    Save in a new column the country, state and location type extracted from the location column.
    Params:
        data.....Dataframe.
    Returns:
        A dataframe with country, state and location vars created for each row.
    """
    location_vars = data.apply(obtain_location_vars_for_row, axis=1)
    data["country2"] = [c[0] for c in location_vars]
    data["region_state"] = [c[1] for c in location_vars]
    data["type"] = [c[2] for c in location_vars]
    data.drop("location", inplace=True, axis=1)
    print("Succesfully created columns country, region_state and type")
    return data


def obtain_success_by_goal_range(data, ranges, range_values):
    """
    Obtain the success rates of the proyects classifying them by goal ranges.
    Params:
        data............Dataframe
        ranges..........A list of the ranges we are going to use.
        range_values....List of values each range will have.
    Returns:
        New dataframe.
    """
    for i in range(len(range_values)):
        if i == 0:
            data.loc[data['usd_goal'] < ranges[i],'goal_range'] = range_values[i]
        elif i == (len(ranges)) :
            data.loc[data['usd_goal'] >= ranges[i-1],'goal_range'] = range_values[i]
        else:
            data.loc[(data['usd_goal'] >= ranges[i-1])&(data['usd_goal'] < ranges[i]),'goal_range'] = range_values[i]
    data['goal_cat_division'] =  data.groupby(['main_category'])['usd_goal'].transform(
                     lambda x: pd.qcut(x, [0, .25, .50, .75, 1.0], labels =['A','B','C','D']))
    
    print("Succesfully created success rate var by ranges")
    return data


def run_competitors_evaluation(data):
    """
    Run a competitors analysis on projects of the dataset.
    Params:
        data....The dataframe.
    Retuns:
        dataframe.
    """
    competitors=data.groupby(['main_category','year_launched','month_launched','goal_range']).count()
    competitors=competitors[['name']]
    competitors.reset_index(inplace=True)
    
    #renaming columns of the derived table
    colmuns_month=['main_category', 'year_launched', 'month_launched', 'goal_range', 'competitors']
    competitors.columns=colmuns_month
    
    #merging the particpants column into the base table
    data = pd.merge(data, competitors, on = ['main_category', 'year_launched', 'month_launched','goal_range'], how = 'left')
    
    #We create a variable to evaluate the proportion of succesful projects depending
    #on the competitors range
    # We can not observe optimal values in the regression plot below. Commented because it takes a lot of time
    #sns.lmplot(x="duration", y="state", data=dataTry,
    #           logistic=True, y_jitter=.05, height=15, aspect=1);
    # 
    #sns.lmplot(x="competitors", y="state", data=dataTry,
    #           logistic=True, y_jitter=.05, height=15, aspect=1); 
    data.loc[data['competitors'] < 10,'comp_range'] = 'A'
    data.loc[(data['competitors'] >= 10)&(data['competitors'] < 30),'comp_range'] = 'B'
    data.loc[(data['competitors'] >= 30)&(data['competitors'] < 60),'comp_range'] = 'C'
    data.loc[(data['competitors'] >= 60)&(data['competitors'] < 100),'comp_range'] = 'D'
    data.loc[(data['competitors'] >= 100)&(data['competitors'] < 150),'comp_range'] = 'E'
    data.loc[(data['competitors'] >= 150)&(data['competitors'] < 200),'comp_range'] = 'F'
    data.loc[data['competitors'] >= 200,'comp_range'] = 'G'
    
    #We also calculate the percentiles of competitors per main_category and give a value.
    data['competitors_cat_division'] =  data.groupby(['main_category'])['competitors'].transform(
                     lambda x: pd.qcut(x, [0, .25, .50, .75, 1.0], labels =['A','B','C','D']))
    return data


def refractor_country_projects(dataframe):
    """
    Count the number of projects from each country and change the country of those that have less than 16,
    since it is a low amount to predict correctly, to OTHER
    Params:
        data....The dataframe.
    Retuns:
        dataframe.
    """
    # TODO: Taking into account 16 threshold for other bucket, or 21?
    countryCount=dataframe.groupby('country2').country2.count()
    countryCount=countryCount.sort_values()
    countries=countryCount[countryCount < 21]
    countries=list(countries.index.values)
    dataframe.loc[dataframe['country2'].isin(countries),'country2'] = 'OTHER'
    
    return dataframe


def us_projects_df(data2, initial_data):
    """
    Build a dataset that conatins only Us projects.
    Params:
        data2...........Pandas dataframe.
        initial_data....Initial Pandas dataframe.
    Returns:
        A tuple containing:
            - A pandas dataframe that contains only US projects.
            - A summary of projects per state.
    """
    dataUS=data2[initial_data['country2'] == 'US']
    #Percentage of projects that are from the US
    USprojectPer=len(dataUS.index)/len(data2.index)*100
    print("%% of projects that are ony form US is %.2f" % USprojectPer)
    
    #State analysis. There is a small percentage with a wrong classification. Classify as OTHER. Delete?
    stateCount=dataUS.groupby('region_state').region_state.count()
    dataUS.loc[dataUS['region_state']=='Canton of Basel-Country','region_state'] = 'OTHER'
    dataUS.loc[dataUS['region_state']=='location','region_state'] = 'OTHER'
    dataUS.loc[dataUS['region_state']=='short_name','region_state'] = 'OTHER'
    dataUS.loc[dataUS['region_state']=='slug','region_state'] = 'OTHER'
    
    return dataUS, stateCount


def plot_figures_about_states(data2, filename):
    """
    Plot and save several figures about the variables.
    Params:
        data2.....Initial Pandas dataframe.
        filename..Path to file to store state related figure.
    Returns:
        Nothing. Saves to disk figure
    """
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(15,15))
    color=['red', 'green']
    
    print("\nPlots regarding states")
    data2.groupby('state').state.count().sort_values().plot(kind='bar', ax=ax1, color=color)
    plt.setp(ax1.get_xticklabels(), rotation=0)
    ax1.set_title('Number of Projects \nper State', fontsize=18)
    ax1.set_xlabel('')
    
    data2.groupby('state').usd_goal.median().plot(kind='bar', ax=ax2, color=color)
    plt.setp(ax2.get_xticklabels(), rotation=0)
    ax2.set_title('Median project goal ($)', fontsize=18)
    ax2.set_xlabel('')
    
    data2.groupby('state').usd_pledged.median().plot(kind='bar', ax=ax3, color=color)
    plt.setp(ax3.get_xticklabels(), rotation=0)
    ax3.set_title('Median project pledged ($)', fontsize=18)
    ax3.set_xlabel('')
    
    data2.groupby('state').backers_count.median().plot(kind='bar', ax=ax4, color=color)
    plt.setp(ax4.get_xticklabels(), rotation=0)
    ax4.set_title('Median project backers', fontsize=18)
    ax4.set_xlabel('')
    
    data2.groupby('state').duration.mean().plot(kind='bar', ax=ax5, color=color)
    plt.setp(ax5.get_xticklabels(), rotation=0)
    ax5.set_title('Mean project duration \nfrom launch to deadline', fontsize=18)
    ax5.set_xlabel('')
    
    data2.groupby('state').name_length.mean().plot(kind='bar', ax=ax6, color=color)
    plt.setp(ax6.get_xticklabels(), rotation=0)
    ax6.set_title('Mean name length of project', fontsize=18)
    ax6.set_xlabel('')
    
    data2.groupby('state').competitors.mean().plot(kind='bar', ax=ax7, color=color)
    plt.setp(ax7.get_xticklabels(), rotation=0)
    ax7.set_title('Median number \nof competitors', fontsize=18)
    ax7.set_xlabel('')
    
    data2.groupby('state').description_length.mean().plot(kind='bar', ax=ax8, color=color)
    plt.setp(ax8.get_xticklabels(), rotation=0)
    ax8.set_title('Mean description \nlength of project', fontsize=18)
    ax8.set_xlabel('')
    
    data2.groupby('state').days_until_launched.mean().plot(kind='bar', ax=ax9, color=color)
    plt.setp(ax9.get_xticklabels(), rotation=0)
    ax9.set_title('Mean project duration \nuntil launched', fontsize=18)
    ax9.set_xlabel('')
    
    fig.subplots_adjust(hspace=0.6)
    fig.tight_layout()
    # TODO: Comment if running from console.
    #plt.show()
    fig.savefig(filename, dpi=fig.dpi)
    
    
def plot_figures_about_main_category(data2, stateDistCat, filename):
    """
    Plot and save several figures with respect to main category.
    Params:
        data2.....Initial Pandas dataframe.
        filename..Path to file to store state related figure.
    Returns:
        Nothing. Saves to disk figure
    """ 
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7,ax8)) = plt.subplots(4, 2, figsize=(35,35))
    #color2 = cm.CMRmap(np.linspace(0.1,0.9,data2.main_category.nunique()))
    color2 = 'blue'
    
    data2.groupby('main_category').main_category.count().sort_values(ascending=False).plot(kind='barh', ax=ax1, color=color2)
    plt.setp(ax1.get_xticklabels(), fontsize=28)
    plt.setp(ax1.get_yticklabels(), fontsize=30, fontweight='bold')
    ax1.set_title('Number of \nprojects', fontsize=32)
    ax1.set_ylabel('')
    
    data2.groupby('main_category').usd_goal.median().sort_values(ascending=False).plot(kind='barh', ax=ax2, color=color2)
    plt.setp(ax2.get_xticklabels(), rotation=0, fontsize=28)
    plt.setp(ax2.get_yticklabels(), fontsize=30, fontweight='bold')
    ax2.set_title('Median project goal ($)', fontsize=32)
    ax2.set_ylabel('')
    
    data2.groupby('main_category').usd_pledged.median().sort_values(ascending=False).plot(kind='barh', ax=ax3, color=color2)
    plt.setp(ax3.get_xticklabels(), rotation=0, fontsize=28)
    plt.setp(ax3.get_yticklabels(), fontsize=30, fontweight='bold')
    ax3.set_title('Median pledged \nper project ($)', fontsize=32)
    ax3.set_ylabel('')
    
    stateDistCat.div(stateDistCat.sum(axis=1), axis=0).successful.sort_values(ascending=False).plot(kind='barh', ax=ax4, color=color2)
    plt.setp(ax4.get_xticklabels(), rotation=0, fontsize=28)
    plt.setp(ax4.get_yticklabels(), fontsize=30, fontweight='bold')
    ax4.set_title('Proportion of \nsuccessful projects', fontsize=32)
    vals = ax4.get_xticks()
    ax4.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    ax4.set_ylabel('')
    
    data2.groupby('main_category').backers_count.median().sort_values(ascending=False).plot(kind='barh', ax=ax5, color=color2)
    plt.setp(ax5.get_xticklabels(), rotation=0, fontsize=28)
    plt.setp(ax5.get_yticklabels(), fontsize=30, fontweight='bold')
    ax5.set_title('Median backers \nper project', fontsize=32)
    ax5.set_ylabel('')
    
    data2.groupby('main_category').pledge_per_backer.median().sort_values(ascending=False).plot(kind='barh', ax=ax6, color=color2)
    plt.setp(ax6.get_xticklabels(), rotation=0, fontsize=28)
    plt.setp(ax6.get_yticklabels(), fontsize=30, fontweight='bold')
    ax6.set_title('Median pledged \nper backer ($)', fontsize=32)
    ax6.set_ylabel('')
    
    data2.groupby('main_category').competitors.median().sort_values(ascending=False).plot(kind='barh', ax=ax7, color=color2)
    plt.setp(ax7.get_xticklabels(), rotation=0, fontsize=28)
    plt.setp(ax7.get_yticklabels(), fontsize=30, fontweight='bold')
    ax7.set_title('Median number \nof competitors', fontsize=32)
    ax7.set_ylabel('')
    
    stateDistComp = pd.get_dummies(data2.set_index('comp_range').state).groupby('comp_range').sum()
    stateDistComp.columns = ['failed', 'successful']
    
    stateDistComp.div(stateDistComp.sum(axis=1), axis=0).successful.sort_values(ascending=False).plot(kind='barh', ax=ax8, color=color2)
    plt.setp(ax8.get_xticklabels(), rotation=0, fontsize=28)
    plt.setp(ax8.get_yticklabels(), fontsize=30, fontweight='bold')
    vals = ax8.get_xticks()
    ax8.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    ax8.set_title('Proportion of \nsuccessful projects', fontsize=32)
    ax8.set_ylabel('Competitors Range', fontsize=28)
    
    fig.subplots_adjust(hspace=0.6)
    fig.tight_layout()
    # TODO: Comment if running from console.
    #plt.show()
    fig.savefig(filename, dpi=fig.dpi)
    

def plot_other_figures(data2, dataUS, filename, filename1,filename2, filename3, filename4, filename5, filename6, filename7, filename8, filename9):
    """
    Plot and save several figure.
    Params:
        data2.....Initial Pandas dataframe.
        dataUS....Dataframe about US only projects.
        filename..Path to file to store state related figure.
    Returns:
        Nothing. Saves to disk figure
    """ 
    #color2 = cm.CMRmap(np.linspace(0.1,0.9,data2.main_category.nunique()))
    color2 = 'blue'
    
    stateDistCountry = pd.get_dummies(data2.set_index('country').state).groupby('country').sum()
    stateDistCountry.columns = ['failed', 'successful']
    fig1, (ax1) = plt.subplots(1,1, figsize=(15,8))
    stateDistCountry.div(stateDistCountry.sum(axis=1), axis=0).successful.sort_values(ascending=False).plot(kind='bar', ax=ax1, color=color2)
    plt.setp(ax1.get_xticklabels(), rotation=70, fontsize=28, fontweight='bold')
    #plt.setp(ax1.get_yticklabels(), fontsize=30)
    ax1.set_title('Proportion of successful projects', fontsize=28)
    ax1.set_xlabel('Country', fontsize=30)
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    fig1.tight_layout()
    #plt.show()
    # TODO: Comment if running from console.
    fig1.savefig(filename, dpi=fig1.dpi)
    
    stateDistMonth = pd.get_dummies(data2.set_index('month_launched').state).groupby('month_launched').sum()
    stateDistMonth.columns = ['failed', 'successful']
    stateDistMonth.index = stateDistMonth.index.str.strip()
    stateDistMonth = stateDistMonth.reindex(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    fig2, (ax2) = plt.subplots(1,1, figsize=(15,8))
    stateDistMonth.div(stateDistMonth.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax2, color=color2)
    plt.setp(ax2.get_xticklabels(), rotation=70, fontsize=28, fontweight='bold')
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax2.set_title('Proportion of successful projects')
    ax2.set_xlabel('Month')
    fig2.tight_layout()
    #plt.show()
    # TODO: Comment if running from console.
    fig2.savefig(filename1, dpi=fig2.dpi)
    
    stateDistYear = pd.get_dummies(data2.set_index('year_launched').state).groupby('year_launched').sum()
    stateDistYear.columns = ['failed', 'successful']
    fig3, (ax3) = plt.subplots(1,1, figsize=(15,8))
    stateDistYear.div(stateDistYear.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax3, color=color2)
    plt.setp(ax3.get_xticklabels(), rotation=0, fontsize=23, fontweight='bold')
    vals = ax3.get_yticks()
    ax3.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax3.set_title('Proportion of successful projects')
    ax3.set_xlabel('Year')
    fig3.tight_layout()
    #plt.show()
    # TODO: Comment if running from console.
    fig3.savefig(filename2, dpi=fig3.dpi)
    
    fig4, (ax4) = plt.subplots(1,1, figsize=(15,8))
    stateDistYear.plot(kind='bar', ax=ax4, color=['red', 'green'])
    plt.setp(ax4.get_xticklabels(), rotation=30, fontsize=23, fontweight='bold')
    ax4.set_title('Number of failed and successful projects')
    ax4.set_xlabel('Year')
    fig4.tight_layout()
    #plt.show()
    # TODO: Comment if running from console.
    fig4.savefig(filename3, dpi=fig4.dpi)
    
    stateDistCurr = pd.get_dummies(data2.set_index('currency').state).groupby('currency').sum()
    stateDistCurr.columns = ['failed', 'successful']
    fig5, (ax5) = plt.subplots(1,1, figsize=(15,8))
    stateDistCurr.div(stateDistCurr.sum(axis=1), axis=0).successful.sort_values(ascending=False).plot(kind='bar', ax=ax5, color=color2)
    plt.setp(ax5.get_xticklabels(), rotation=30, fontsize=20, fontweight='bold')
    vals = ax5.get_yticks()
    ax5.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax5.set_title('Proportion of successful projects')
    ax5.set_xlabel('Currency')
    fig5.tight_layout()
    #plt.show()
    # TODO: Comment if running from console.
    fig5.savefig(filename4, dpi=fig5.dpi)
    
    stateDistWeekday = pd.get_dummies(data2.set_index('weekday_launched').state).groupby('weekday_launched').sum()
    stateDistWeekday.columns = ['failed', 'successful']
    stateDistWeekday.index = stateDistWeekday.index.str.strip()
    stateDistWeekday = stateDistWeekday.reindex(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    fig6, (ax6) = plt.subplots(1,1, figsize=(12,8))
    stateDistWeekday.div(stateDistWeekday.sum(axis=1), axis=0).successful.sort_values(ascending=False).plot(kind='bar', ax=ax6, color=color2)
    plt.setp(ax6.get_xticklabels(), rotation=30, fontsize=20, fontweight='bold')
    vals = ax6.get_yticks()
    ax6.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax6.set_title('Proportion of successful projects')
    ax6.set_xlabel('Weekday')
    fig6.tight_layout()
    #plt.show()
    # TODO: Comment if running from console.
    fig6.savefig(filename5, dpi=fig6.dpi)
    
    stateDistUS = pd.get_dummies(dataUS.set_index('region_state').state).groupby('region_state').sum()
    stateDistUS.columns = ['failed', 'successful']
    fig7 = plt.figure(figsize=(12,18))
    ax7 = plt.gca()
    stateDistUS.div(stateDistUS.sum(axis=1), axis=0).successful.sort_values(ascending=False).plot(kind='barh', ax=ax7, color=color2)
    plt.setp(ax7.get_yticklabels(), rotation=0, fontsize=18)
    vals = ax7.get_xticks()
    ax7.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    ax7.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False,
                     bottom=True, top=True, left=True, right=False)
    ax7.set_title('Proportion of successful projects in US', y = 1.03)
    ax7.set_ylabel('US State')
    fig7.tight_layout()
    #plt.show()
    # TODO: Comment if running from console.
    fig7.savefig(filename6, dpi=fig7.dpi)
    
    stateDistCountry2 = pd.get_dummies(data2.set_index('country2').state).groupby('country2').sum()
    stateDistCountry2.columns = ['failed', 'successful']
    fig8, (ax8) = plt.subplots(1,1, figsize=(12,20))
    stateDistCountry2.div(stateDistCountry2.sum(axis=1), axis=0).successful.sort_values(ascending=False).plot(kind='barh', ax=ax8, color=color2)
    vals = ax8.get_xticks()
    ax8.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.setp(ax8.get_yticklabels(), rotation=0, fontsize=15)
    ax8.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False,
                     bottom=True, top=True, left=True, right=False)
    ax8.set_title('Proportion of successful projects', y=1.02)
    ax8.set_ylabel('Country True')
    fig8.tight_layout()
    #plt.show()
    # TODO: Comment if running from console.
    fig8.savefig(filename7, dpi=fig8.dpi)
    
    stateDistGoal = pd.get_dummies(data2.set_index('goal_range').state).groupby('goal_range').sum()
    stateDistGoal.columns = ['failed', 'successful']
    fig9, (ax9) = plt.subplots(1,1, figsize=(12,8))
    stateDistGoal.div(stateDistGoal.sum(axis=1), axis=0).successful.sort_values(ascending=False).plot(kind='bar', ax=ax9, color=color2)
    vals = ax9.get_yticks()
    ax9.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.setp(ax9.get_xticklabels(), rotation=0, fontsize=20, fontweight='bold')
    ax9.set_title('Proportion of successful projects')
    ax9.set_xlabel('Goal Range')
    fig9.tight_layout()
    #plt.show()
    # TODO: Comment if running from console.
    fig9.savefig(filename8, dpi=fig9.dpi)
    
    stateDistType = pd.get_dummies(data2.set_index('type').state).groupby('type').sum()
    stateDistType.columns = ['failed', 'successful']
    fig10, (ax10) = plt.subplots(1,1, figsize=(12,8))
    stateDistType.div(stateDistType.sum(axis=1), axis=0).successful.sort_values(ascending=False).plot(kind='barh', ax=ax10, color=color2)
    plt.setp(ax10.get_yticklabels(), rotation=0, fontsize=20, fontweight='bold')
    vals = ax10.get_xticks()
    ax10.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    ax10.set_title('Proportion of successful projects')
    ax10.set_ylabel('Type of location')
    fig10.tight_layout()
    #plt.show()
    # TODO: Comment if running from console.
    fig10.savefig(filename9, dpi=fig10.dpi)

    
def prepare_data_for_ML(dataframe):
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
    #First of, we need to create dummy variables for the categorical variables
    data3 = pd.get_dummies(dataframe)
    
    #We then need to seperate into the input data X variables and the target y variables
    X1 = data3.drop('state', axis=1)
    y = data3.state
    
    #Then, Transform the inputs in X so that they are on the same scale
    scale = StandardScaler()
    X = pd.DataFrame(scale.fit_transform(X1), columns=list(X1.columns))
    
    return data3, X, y
    
    

def main():
    # 0 - Initial directories and colnames set up
    print("Step 0: Initial directories and colnames set up")
    dirname, datadir, imagesdir, initial_colnames = initial_setup()
    print("Directory of this file is %s" % dirname)
    print("Data directory is %s" % datadir)
    print("Images directory is %s" % imagesdir)
    print("Initial columns for our model are: \n%s" % initial_colnames)
    
    
    # 1 - Load from disk the complete Merged Dataframe.
    print("\n\n\nStep 1: Load from disk the complete Merged Dataframe.")
    filename = os.path.join(datadir, 'dataframe_total.pkl')
    print("Completed Dataframe read from file %s" % filename)
    data = read_from_disk(filename)
    # Print summary of dataframe
    print("Dataframe contains %d projects and %d columns for each project\n" % (data.shape[0], data.shape[1]))
    
    
    # 2 - Look for missing values for every row and print summary.
    print("\n\n\nStep 2: Look for missing values for every row and print summary.")
    data = check_missing_values_and_drop(data, True)
    print("As we can see, we have very low percentage of missing values,the highest column with missing values is location column with only a 0.34 %, so we decided to drop the missing values")
    print("Also, studying the missing data, we discover that out of 1091 rows with missing data: \n")
    print("United States     1087\nGreat Britain     2\nDenmark           1\nAustria           1\n")
    print("The distribution of the missing values across the main_category variable is:\n")
    print("art             118\ncomics           14\ncrafts            9\ndance            18\ndesign           12\nfashion           6\nfilm & video    279\nfood            10\ngames            49\njournalism       55\nmusic           258\nphotography      49\npublishing      141\ntechnology       51\ntheater          22")
    # TODO: NEED TO CHECK OTHER TYPES OF EMPTY VALUES ("empty strings for example") They have already been checked right?
    
    
    # 3 - Create new variables from present columns. The new columns to create are:
    #       - usd_goal: Goal of the project in USD.
    #       - duration: Contains the dyas between the launching date and the deadline.
    #       - duration_until_lanched: Tracks the # of days between the created day and the launched date. 
    print("\n\n\nStep 3: Create new columns usd_goal, duration and duration_until_launched from present columns.")
    data = create_new_vars(data)
    
    
    # 4 - Create year, month and week vars from the launched_at var, and convert to 
    #    date type launched_at and deadline vars   
    print("\n\n\nStep 4: Create year, month and week vars from the launched_at var, and convert to date type launched_at and deadline vars.")
    data = create_new_date_cols(data, 'launched_at', '%Y', 'year_launched', 1)
    data = create_new_date_cols(data, 'launched_at', '%b', 'month_launched', 1)
    data = create_new_date_cols(data, 'launched_at', '%a', 'weekday_launched', 1)
    data = convert_to_date(data, 'launched_at', 1)
    data = convert_to_date(data, 'deadline', 1)
    
    
    # 5 - Plots per year.
    print("\n\n\nStep 5: Plots per year.")
    filename1 = os.path.join(imagesdir,'number_proyects_per_year.png')
    filename2 = os.path.join(imagesdir,'number_proyects_per_year_per_state.png')
    plot_proyects_per_year(data, filename1, filename2)
    print("Plots per year succesfully saved to file %s and file %s" % (filename1, filename2))
    
    
    # 6 - Find length of name and blurb, and drop vars.
    print("\n\n\nStep 6: Name and description length. Drop not necessary vars.")
    data = len_str_col(data, 'name', 'name_length')
    data = len_str_col(data, 'blurb', 'description_length')
    # Drop variables that we don't need anymore. Because they have been used to engineer new ones.
    # TODO: Maybe we want to maintain blurb for running tokenize analysis further if possible.
    to_drop = ['blurb','state_changed_at','pledged','deadline','launched_at','created_at','static_usd_rate']
    data = remove_cols(data, to_drop)
    
    
    # 7 - Summary of data
    print("\n\n\nStep 7: Print data summary")
    unique, summary = print_data_summary(data)
    
    
    # 8 - Create new columns main_category and sub_category
    print("\n\n\nStep 8: Create category and subcategory columns")
    data = create_cat_and_subcat(data)
    
    
    # 9 - Create location vars.
    print("\n\n\nStep 9: Create country, state and type columns")
    data = create_location_vars(data)
    
    
    # 10 - Calculate the pledge per backer for each project.
    print("\n\n\nStep 10: Calculate the pledge per backer for each project.")
    data['pledge_per_backer']=data['usd_pledged']/data['backers_count']
    
    
    # 11 - Calculate the percentage of "success" by dividing the amount of money pledge by the goal
    print("\n\n\nStep 11: Calculate the percentage of success")
    # TODO: Do not understand what this variable tries to obtain
    data['success_rate']=data['usd_pledged']/data['usd_goal']*100
    
    
    # 12 - Recheck missing values
    print("\n\n\nStep 12: Recheck missing values")
    #There are some missing values in the data frame like the pledge_per_backer
    #variable. It is nan when both usd_pledged and backers are 0.
    check_missing_values_and_drop(data, drop=False)
    print('The missing values in the region_state variable per country, are as follows:\n')
    print('AQ  Antartica  23\nNZ  New Zealand  23\nMK  Macedonia  15\nAW  Aruba  1\nCW  Curacau   3\nGI  Gibraltar   4\nKI  Kiribati  1\nMO  Macao   1\nPN Pitcairn    1\nSX Sint Maarten  3\nVA  Vatican City  1\nXK Kosovo    7')
    print('As we can see, most of the regions that are missing are either from a small country, an island or Antartica\n')
    print('The missing values in the region_state variable per main_category, are as follows:\n')
    print('art            10\ncrafts          1\ndesign          1\nfashion         1\nfilm&video     13\nfood            5\ngames          18\njournalism      3\nmusic           4\nphotography    11\npublishing      9\ntechnology      6\ntheater         1\n')
    print("As we can see the missing values in the region_state variable have more to do with the country than with the category.\n")
    print("For the region_state variables that don't have a state we change the empty box to None.\n")
    data.loc[data['region_state'].isnull(),'region_state'] = 'None'
    #We print the amount of sub_categories per main_category
    dataSub=data[data['sub_category']=='']
    dataSub1=dataSub.groupby('main_category').sub_category.count()
    print("The amount of missing sub_category variables per main_category is:\n")
    print(dataSub1)
    dataSub2=round((dataSub.groupby('main_category').sub_category.count())/data["main_category"].value_counts()*100,2)
    print("The percentage of missing sub_category variables per main_category is:\n")
    print(dataSub2)
    print("For the empty strings found in sub_category we change the empty string to None.\n")
    data.loc[data['sub_category'] == '','sub_category'] = 'None'

    # We fill all nan with zero
    data.fillna(0,inplace=True)
    check_missing_values_and_drop(data, drop=False)
    print("In the case of pledge_per_backer, there are missing values, because some of the projects have 0 usd_pledged and 0 backers, and so the division becomes nan.")

    # 13 - Create a variable to evaluate the proportion of successful projects depending
    # on the goal money range
    print("\n\n\nStep 13: Calculate the percentage of success")
    print("To create the goal_range variable we define the ranges based on Kickstarters.\n")
    ranges = [1000, 3000, 6000, 10000, 20000, 100000, 1000000]
    range_values = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    data = obtain_success_by_goal_range(data, ranges, range_values)
    
    
    # 14 - Calculate the number of competitors in the same category, with the same goal range and in a time perios
    # same year and month
    print("\n\n\nStep 14: Calculate the number of competitors in the same category, with the same goal range and in a time period same year and month")
    data = run_competitors_evaluation(data)
    
    
    # 15 - We find out the distribution of data across state. 
    print("\n\n\nStep 15: We find out the distribution of data across state.")
    percentage_per_state = round(data["state"].value_counts() / len(data["state"]) * 100,2)
    print("State Percent: ")
    print(percentage_per_state)
    #The higher percentage belong to succesful and failed state, so we can get rid of the rest of the projects that have another category
    #We only keep those projects that have values either successful or failed
    data2 = data[(data['state'] == 'failed') | (data['state'] == 'successful')]
    
    
    # 16 - Count the number of projects from each country and change the country of those that have less than 21,
    # since it is a low amount to predict correctly, to OTHER.
    print("\n\n\nStep 16: Count the number of projects from each country and change the country of those that have less than 21, since it is a low amount to predict correctly, to OTHER.")
    data2 = refractor_country_projects(data2)
    
    
    # 17 - Lets get a dataframe only with the projects in the US.
    print("\n\n\nStep 17: Analysis on US only projects")
    dataUS, stateCount = us_projects_df(data2, data)
    
    
    # 18 - Distribution of projects across main categories.
    print("\n\n\nStep 18: Distribution of projects across main categories.")
    stateDistCat = pd.get_dummies(data2.set_index('main_category').state).groupby('main_category').sum()
    stateDistCat.columns = ['failed', 'successful']
    
    
    # 19 - Finding the correlation of continuous variables with the dependent variable.
    print("\n\n\nStep 19: Finding the correlation of continuous variables with the dependent variable.")
    corr=data2[['backers_count','usd_pledged','usd_goal','duration','name_length','days_until_launched','pledge_per_backer','state']].corr()
    print(corr)
    # Print correlation as a figure
    fig = plt.figure()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(corr, mask=mask, vmax=.3, square=True)
    filename = os.path.join(imagesdir,'correlations.png')
    print(filename)
    fig.savefig(filename, dpi = fig.dpi)
    
    # 20 - Per state plots.
    print("\n\n\nStep 20: Per state plots.")
    filename = os.path.join(imagesdir,'by_state_plots.png')
    plot_figures_about_states(data2, filename)
    print("Plots per state succesfully saved to file %s" % filename)
    
    
    # 21 - Per main_category plots.
    print("\n\n\nStep 21: Per main_category plots.")
    filename = os.path.join(imagesdir,'by_main_category_plots.png')
    plot_figures_about_main_category(data2, stateDistCat, filename)
    print("Plots per main_category succesfully saved to file %s" % filename)
    
    
    # 22 - Other plots.
    print("\n\n\nStep 22: Other plots.")
    filename = os.path.join(imagesdir,'plot_per_country.png')
    filename1 = os.path.join(imagesdir,'plot_per_month.png')
    filename2 = os.path.join(imagesdir,'plot_per_year.png')
    filename3 = os.path.join(imagesdir,'plot_per_year_dist.png')
    filename4 = os.path.join(imagesdir,'plot_per_currency.png')
    filename5 = os.path.join(imagesdir,'plot_per_weekday.png')
    filename6 = os.path.join(imagesdir,'plot_per_state_US.png')
    filename7 = os.path.join(imagesdir,'plot_per_countryTrue.png')
    filename8 = os.path.join(imagesdir,'plot_per_goal_range.png')
    filename9 = os.path.join(imagesdir,'plot_per_location.png')

    plot_other_figures(data2, dataUS, filename,filename1, filename2, filename3, filename4, filename5, filename6, filename7, filename8, filename9)
    print("Other plots succesfully saved to file %s" % filename)
    

    # 23 - Convert the variables 'successful' state to 1 and failed to 0, to have our logical target variable
    print("\n\n\nStep 23: Convert the variables 'successful' state to 1 and failed to 0, to have our logical target variable.")
    data2['state'] = (data2['state'] =='successful').astype(int)
    
    
    # 24 - Drop variables that we dont need anymore. Because they have been used to engineer new ones.
    print("\n\n\nStep 24: Drop variables that we dont need anymore. Because they have been used to engineer new ones.")
    to_drop4 = ['name','country','region_state']
    data2 = remove_cols(data2, to_drop4)
    
    
    # 25 - Preparing the data for Machine Learning
    print("\n\n\nStep 25: Preparing the data for Machine Learning.")
    data3, X, y = prepare_data_for_ML(data2)
    
    
    # 26 - Data stratified sampling.
    print("\n\n\nStep 26: Data sampling into training and test dataset.")
    #Finally, the data is separated into training and testing set
    # TODO: Are we using stratified sampling as seen in class or a sampling type that allows having equal distribution of every possible value of the target var in each set?
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=22333)
    
    
    # 27 - Save different datasets obtained.
    print("\n\n\nStep 27: Save different datasets obtained.")
    filename = os.path.join(datadir, 'formatting_initial_data.pkl')
    store_dataframe(data, filename)
    print("Initial dataframe 'data' succesfully saved to %s" % filename)
    filename = os.path.join(datadir, 'formatting_intermediate_data2.pkl')
    store_dataframe(data2, filename)
    print("Intermediate dataframe 'data2' succesfully saved to %s" % filename)
    filename = os.path.join(datadir, 'formatting_ML_data3.pkl')
    store_dataframe(data3, filename)
    print("Machine Learning dataframe 'data3' succesfully saved to %s" % filename)
    filename = os.path.join(datadir, 'formatting_ML_X.pkl')
    store_dataframe(X, filename)
    print("Machine Learning 'X' dataframe succesfully saved to %s" % filename)
    filename = os.path.join(datadir, 'formatting_ML_y.pkl')
    store_dataframe(y, filename)
    print("Machine Learning 'y' array succesfully saved to %s" % filename)
    
    
if __name__ == "__main__":
    main()