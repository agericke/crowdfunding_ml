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
#%matplotlib inline


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


def plot_proyects_per_year(data, filename):
    """
    Plot amount of proyects grouped by year and proyect state per year.
    Params:
        data.......Dataframe to plot the data from.
        filename...Path to the file where the image will be saved.
    Returns:
        Nothing. Saves image to filename.
    """    
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(25,12))
    data.groupby('year_launched').year_launched.count().plot(kind='bar', ax=ax1, color='green')
    ax1.set_title('Number of Projects per Year')
    ax1.set_xlabel('Year')
    stateDistYear1 = pd.get_dummies(data.set_index('year_launched').state).groupby('year_launched').sum()
    #stateDistYear1.columns = ['failed', 'successful']
    stateDistYear1.plot(kind='bar', ax=ax2)
    ax2.set_title('State distribution of projects per year')
    ax2.set_xlabel('Year')
    fig.savefig(filename, dpi=fig.dpi)


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
    Create category column with data form the dataframe.
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
    # TODO: REALLY IMPORTANT: DEFINE A WAY TO OBTAIN OPTIMAL RANGES SEPARATORS (JUST AS DID IN CLASS ONCE)
    # TODO: PROFESSOR DID IT BY SCATTER PLOTTING VARIABLE AGAINST VAR OBJECTIVE (SUCCESS RATE IN OUR CASE) IF I REMEMBER WELL.
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
    # TODO: Refractor this part.
    # TODO: Again see what are optimal values to set ranges.
    data.loc[data['competitors'] < 10,'comp_range'] = 'A'
    data.loc[(data['competitors'] >= 10)&(data['competitors'] < 30),'comp_range'] = 'B'
    data.loc[(data['competitors'] >= 30)&(data['competitors'] < 60),'comp_range'] = 'C'
    data.loc[(data['competitors'] >= 60)&(data['competitors'] < 100),'comp_range'] = 'D'
    data.loc[(data['competitors'] >= 100)&(data['competitors'] < 150),'comp_range'] = 'E'
    data.loc[(data['competitors'] >= 150)&(data['competitors'] < 200),'comp_range'] = 'F'
    data.loc[data['competitors'] >= 200,'comp_range'] = 'G'
    
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
    # TODO: Taking into account 16 threshold for other bucket, or 51?
    countryCount=dataframe.groupby('country2').country2.count()
    countryCount=countryCount.sort_values()
    countries=countryCount[countryCount < 51]
    countries=list(countries.index.values)
    dataframe.loc[dataframe['country2'].isin(countries),'country2'] = 'OTHER'
    
    # TODO: Can we erase this lines?
    #data2 = data2[~data2['country2'].isin(countries)]
    countryCount2=dataframe.groupby('country2').country2.count()
    countryCount2=countryCount2.sort_values()
    
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
    
    #State analysis. There is a small percentage with a wrong classification. Classify as OTHER
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
    color=['orange', 'blue', 'pink', 'red', 'green', 'yellow', 'cyan']
    
    print("\nPlots regarding states")
    data2.groupby('state').state.count().plot(kind='bar', ax=ax1, color=color)
    ax1.set_title('Number of Projects per State')
    ax1.set_xlabel('')
    
    data2.groupby('state').usd_goal.median().plot(kind='bar', ax=ax2, color=color)
    ax2.set_title('Median project goal ($)')
    ax2.set_xlabel('')
    
    data2.groupby('state').usd_pledged.median().plot(kind='bar', ax=ax3, color=color)
    ax3.set_title('Median project pledged ($)')
    ax3.set_xlabel('')
    
    data2.groupby('state').backers_count.median().plot(kind='bar', ax=ax4, color=color)
    ax4.set_title('Median project backers')
    ax4.set_xlabel('')
    
    data2.groupby('state').duration.mean().plot(kind='bar', ax=ax5, color=color)
    ax5.set_title('Mean project duration from launch to deadline')
    ax5.set_xlabel('')
    
    data2.groupby('state').name_length.mean().plot(kind='bar', ax=ax6, color=color)
    ax6.set_title('Mean name length of project')
    ax6.set_xlabel('')
    
    
    # TODO: Can we erase this lines?
    #staffPickDistr = pd.get_dummies(data2.set_index('state').staff_pick).groupby('state').sum()
    #staffPickDistr.columns = ['false', 'true']
    #staffPickDistr.div(staffPickDistr.sum(axis=1), axis=0).true.plot(kind='bar', ax=ax7)
    #ax7.set_title('Proportion that are staff picks')
    #ax7.set_xlabel('')
    
    data2.groupby('state').competitors.mean().plot(kind='bar', ax=ax7, color=color)
    ax7.set_title('Median number of competitors')
    ax7.set_xlabel('')
    
    data2.groupby('state').description_length.mean().plot(kind='bar', ax=ax8, color=color)
    ax8.set_title('Mean description length of project')
    ax8.set_xlabel('')
    
    data2.groupby('state').days_until_launched.mean().plot(kind='bar', ax=ax9, color=color)
    ax9.set_title('Mean project duration until launched')
    ax9.set_xlabel('')
    
    fig.subplots_adjust(hspace=0.6)
    # TODO: Comment if running from console.
    plt.show()
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
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7,ax8)) = plt.subplots(4, 2, figsize=(20,20))
    color2 = cm.CMRmap(np.linspace(0.1,0.9,data2.main_category.nunique()))
    
    data2.groupby('main_category').main_category.count().plot(kind='bar', ax=ax1, color=color2)
    ax1.set_title('Number of projects')
    ax1.set_xlabel('')
    
    data2.groupby('main_category').usd_goal.median().plot(kind='bar', ax=ax2, color=color2)
    ax2.set_title('Median project goal ($)')
    ax2.set_xlabel('')
    
    data2.groupby('main_category').usd_pledged.median().plot(kind='bar', ax=ax3, color=color2)
    ax3.set_title('Median pledged per project ($)')
    ax3.set_xlabel('')
    
    stateDistCat.div(stateDistCat.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax4, color=color2)
    ax4.set_title('Proportion of successful projects')
    ax4.set_xlabel('')
    
    data2.groupby('main_category').backers_count.median().plot(kind='bar', ax=ax5, color=color2)
    ax5.set_title('Median backers per project')
    ax5.set_xlabel('')
    
    data2.groupby('main_category').pledge_per_backer.median().plot(kind='bar', ax=ax6, color=color2)
    ax6.set_title('Median pledged per backer ($)')
    ax6.set_xlabel('')
    
    data2.groupby('main_category').competitors.median().plot(kind='bar', ax=ax7, color=color2)
    ax7.set_title('Median number of competitors')
    ax7.set_xlabel('')
    
    stateDistComp = pd.get_dummies(data2.set_index('comp_range').state).groupby('comp_range').sum()
    stateDistComp.columns = ['failed', 'successful']
    
    stateDistComp.div(stateDistComp.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax8, color=color2)
    ax8.set_title('Proportion of successful projects')
    ax8.set_xlabel('Competitors Range')
    
    fig.subplots_adjust(hspace=0.6)
    # TODO: Comment if running from console.
    plt.show()
    fig.savefig(filename, dpi=fig.dpi)
    

def plot_other_figures(data2, dataUS, filename):
    """
    Plot and save several figure.
    Params:
        data2.....Initial Pandas dataframe.
        dataUS....Dataframe about US only projects.
        filename..Path to file to store state related figure.
    Returns:
        Nothing. Saves to disk figure
    """ 
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2, figsize=(30,40))
    color2 = cm.CMRmap(np.linspace(0.1,0.9,data2.main_category.nunique()))
    
    stateDistCountry = pd.get_dummies(data2.set_index('country').state).groupby('country').sum()
    stateDistCountry.columns = ['failed', 'successful']
    
    stateDistCountry.div(stateDistCountry.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax1, color=color2)
    ax1.set_title('Proportion of successful projects')
    ax1.set_xlabel('Country')
    
    stateDistMonth = pd.get_dummies(data2.set_index('month_launched').state).groupby('month_launched').sum()
    stateDistMonth.columns = ['failed', 'successful']
    stateDistMonth.index = stateDistMonth.index.str.strip()
    stateDistMonth = stateDistMonth.reindex(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    
    stateDistMonth.div(stateDistMonth.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax2, color=color2)
    ax2.set_title('Proportion of successful projects')
    ax2.set_xlabel('Month')
    
    stateDistYear = pd.get_dummies(data2.set_index('year_launched').state).groupby('year_launched').sum()
    stateDistYear.columns = ['failed', 'successful']
    
    stateDistYear.div(stateDistYear.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax3, color='green')
    ax3.set_title('Proportion of successful projects')
    ax3.set_xlabel('Year')
    
    stateDistYear.plot(kind='bar', ax=ax4)
    ax4.set_title('Number of failed and successful projects')
    ax4.set_xlabel('Year')
    
    stateDistCurr = pd.get_dummies(data2.set_index('currency').state).groupby('currency').sum()
    stateDistCurr.columns = ['failed', 'successful']
    
    stateDistCurr.div(stateDistCurr.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax5, color=color2)
    ax5.set_title('Proportion of successful projects')
    ax5.set_xlabel('Currency')
    
    stateDistWeekday = pd.get_dummies(data2.set_index('weekday_launched').state).groupby('weekday_launched').sum()
    stateDistWeekday.columns = ['failed', 'successful']
    stateDistWeekday.index = stateDistWeekday.index.str.strip()
    stateDistWeekday = stateDistWeekday.reindex(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    
    stateDistWeekday.div(stateDistWeekday.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax6, color=color2)
    ax6.set_title('Proportion of successful projects')
    ax6.set_xlabel('Weekday')
    
    stateDistUS = pd.get_dummies(dataUS.set_index('region_state').state).groupby('region_state').sum()
    stateDistUS.columns = ['failed', 'successful']
    
    stateDistUS.div(stateDistUS.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax7, color=color2)
    ax7.set_title('Proportion of successful projects in US')
    ax7.set_xlabel('US State')
    
    stateDistCountry2 = pd.get_dummies(data2.set_index('country2').state).groupby('country2').sum()
    stateDistCountry2.columns = ['failed', 'successful']
    
    stateDistCountry2.div(stateDistCountry2.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax8, color=color2)
    ax8.set_title('Proportion of successful projects')
    ax8.set_xlabel('Country True')
    
    stateDistGoal = pd.get_dummies(data2.set_index('goal_range').state).groupby('goal_range').sum()
    stateDistGoal.columns = ['failed', 'successful']
    
    stateDistGoal.div(stateDistGoal.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax9, color=color2)
    ax9.set_title('Proportion of successful projects')
    ax9.set_xlabel('Goal Range')
    
    stateDistType = pd.get_dummies(data2.set_index('type').state).groupby('type').sum()
    stateDistType.columns = ['failed', 'successful']
    
    stateDistType.div(stateDistType.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax10, color=color2)
    ax10.set_title('Proportion of successful projects')
    ax10.set_xlabel('Type of location')
    
    fig.subplots_adjust(hspace=0.6)
    # TODO: Comment if running from console.
    plt.show()
    fig.savefig(filename, dpi=fig.dpi)
    
    
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
    print("Complete Dataframe is read from file %s" % filename)
    data = read_from_disk(filename)
    # Print summary of dataframe
    print("Dataframe contains %d projects and %d columns for each project\n" % (data.shape[0], data.shape[1]))
    
    # TODO: Erase?
    #data.set_index('id',inplace=True)
    
    
    # 2 - Look for missing values for every row and print summary.
    print("\n\n\nStep 2: Look for missing values for every row and print summary.")
    data = check_missing_values_and_drop(data, True)
    print("As we can see, we have very low percentage of missing values,the highest column with missing values is location column with only a 0.34 %, so we decided to drop the missing values")
    # TODO: NEED TO CHECK OTHER TYPES OF EMPTY VALUES ("empty strings for example")
    
    
    # 3 - Create new variables from present columns. The new columns to create are:
    #       - usd_goal: Goal of the project in USD.
    #       - duration: Contains the dyas between the launching date and the deadline.
    #       - duration_until_lanched: Tracks the # of days between the created day and the launched date. 
    print("\n\n\nStep 3: Create new columns usd_goal, duration and duration_until_launched from present columns.")
    data = create_new_vars(data)
    
    
    # 4 - Create year, month and week vars from the launched_at var, and convert to 
    #    date type launched_at and deadline vars   
    # TODO: Can we erase this line?
    #day=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1347517370))
    print("\n\n\nStep 4: Create year, month and week vars from the launched_at var, and convert to date type launched_at and deadline vars.")
    data = create_new_date_cols(data, 'launched_at', '%Y', 'year_launched', 1)
    data = create_new_date_cols(data, 'launched_at', '%b', 'month_launched', 1)
    data = create_new_date_cols(data, 'launched_at', '%a', 'weekday_launched', 1)
    data = convert_to_date(data, 'launched_at', 1)
    data = convert_to_date(data, 'deadline', 1)
    
    
    # 5 - Plots per year.
    print("\n\n\nStep 5: Plots per year.")
    filename = os.path.join(imagesdir,'proyects_per_year.png')
    plot_proyects_per_year(data, filename)
    print("Plots per year succesfully saved to file %s" % filename)
    
    
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
    # TODO: Need to run missing values identification in the new vars, and plot some graphs for the new vars.
    # TODO: for running the missing values identification is not sufficient to run only .isna() function. Also
    # TODO: necessary to identify empty strings, or other values that may be introduced as missing values just as studied in class.
    
    
    # 10 - Calculate the pledge per backer for each project.
    print("\n\n\nStep 10: Calculate the pledge per backer for each project.")
    data['pledge_per_backer']=data['usd_pledged']/data['backers_count']
    
    
    # 11 - Calculate the percentage of "success" by dividing the amount of money pledge by the goal
    print("\n\n\nStep 11: Calculate the percentage of success")
    # TODO: Do not understand what this variable tries to obtain
    data['success_rate']=data['usd_pledged']/data['usd_goal']*100
    
    
    # 12 - Recheck missing values
    print("\n\n\nStep 12: Recheck missing values")
    #There are no missing values in the data frame except for the pledge_per_backer
    #variable. It is nan when both usd_pledged and backers are 0.
    check_missing_values_and_drop(data, drop=False)
    # We fill all nan with zero
    data.fillna(0,inplace=True)
    check_missing_values_and_drop(data, drop=False)
    
    
    # 13 - Create a variable to evaluate the proportion of succesful projects depending
    # on the goal money range
    print("\n\n\nStep 13: Calculate the percentage of success")
    # TODO: Really important, redefine ranges following a criteria.
    ranges = [250, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    range_values = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    data = obtain_success_by_goal_range(data, ranges, range_values)
    
    
    # 14 - Calculate the number of competitors in the same category, with the same goal range and in a time perios
    # same year and month
    print("\n\n\nStep 14: Calculate the number of competitors in the same category, with the same goal range and in a time perio dsame year and month")
    data = run_competitors_evaluation(data)
    
    
    # 15 - We find out the distribution of data across state. 
    print("\n\n\nStep 15: We find out the distribution of data across state.")
    percentage_per_state = round(data["state"].value_counts() / len(data["state"]) * 100,2)
    print("State Percent: ")
    print(percentage_per_state)
    #The higher percentage belong to succesful and failed state, so we can get rid of the rest of the projects that have another category
    #We only keep those projects that have values either successful or failed
    # TODO: Re-check conclusions. Probably professor will ask the reason for droping the vars.
    data2 = data[(data['state'] == 'failed') | (data['state'] == 'successful')]
    
    
    # 16 - Count the number of projects from each country and change the country of those that have less than 16,
    # since it is a low amount to predict correctly, to OTHER.
    print("\n\n\nStep 16: Count the number of projects from each country and change the country of those that have less than 16, since it is a low amount to predict correctly, to OTHER.")
    data2 = refractor_country_projects(data2)
    
    #Check if the data frame is in appropriate format:
    data2.head()
    # TODO: Can we erase this line?
    #Finally, we drop the id variable. We dont need it for the models
    #data2.drop("id", inplace=True, axis=1)
    
    
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
    filename = os.path.join(imagesdir,'other_plots.png')
    plot_other_figures(data2, dataUS, filename)
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
    store_dataframe(data, filename)
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
    
    # TODO: Maybe it is better having one figure per plot instead of plotting several figures in one plot. This will make it easier
    # for including the plots in the presentation afterwards as well as in the report that we will need to generate.

    
if __name__ == "__main__":
    main()    
