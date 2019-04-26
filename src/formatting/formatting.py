#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 01:54:00 2019

@author: agericke
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import datetime
import time
#%matplotlib inline

# Initial directories set up
dirname = os.path.dirname(os.path.abspath('__file__'))
datadir = os.path.join(os.path.abspath(os.path.join(os.path.join(dirname, os.pardir), os.pardir)), 'data/')

"""
Need to create a big dataframe. For each csv file we should add the data to the dataframe,
check if the column names coincide and expand the dataframe.

Loop through all folders, and add a column to each project that represents the year
and the season of the year.
"""
data = pd.read_csv(os.path.join(datadir, "2016-05-15T020446/Kickstarter001.csv"))

#First lets drop the columns that have no values
to_drop = ['friends','is_starred','is_backing','permissions']
data.drop(to_drop, inplace=True, axis=1)
#Now we will drop the columns/variables that we have determine have no value to our study
to_drop2 = ['photo','currency','blurb','currency_symbol','currency_trailing_code','state_changed_at',
            'slug','created_at','disable_communication','creator','spotlight','profile','urls','source_url']
data.drop(to_drop2, inplace=True, axis=1)

# Drop missing rows
data = data.dropna()
data['city'] = data['location'].apply(lambda x: x.split(":")[-8].strip("\"").split("\"")[0])
data['country_state'] = data['location'].apply(lambda x: x.split(":")[-7].strip("\"").split(",")[1].strip(" ").strip("\""))

#Extract the categories: main_category and the category
#category=data['category']
#for i in data.iterrows(): 
 #   cat=category[i]
  #  category[i]=cat.get('slug')
  
#Create usd_goal, where we convert the goal into usd
data["usd_goal"] = data["goal"]*data["static_usd_rate"]
#Drop the original goal feature
data.drop("goal", inplace=True, axis=1)
   
#Create variable duration in days. We get a variable in seconds
data["duration"] = data["deadline"]-data["launched_at"]
#Transform the variable to days
data["duration"] = data["duration"]/(60*60*24)

#Create the variables Month and Year in which the project was launched
#time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1347517370))   

data['year_launched'] = data.apply(lambda row: time.strftime('%Y', time.localtime(row.launched_at)), axis=1)
data['month_launched'] = data.apply(lambda row: time.strftime('%m', time.localtime(row.launched_at)), axis=1)
data['launched_at'] = data.apply(lambda row: datetime.datetime.fromtimestamp(row.launched_at), axis=1)
data['deadline'] = data.apply(lambda row: datetime.datetime.fromtimestamp(row.deadline), axis=1)

#Find out the length of the name
data['name_length'] = data['name'].str.len() 
#Should we get the length of the blurb variable too? It is the description   


#Drop variables that we dont need anymore
to_drop3 = ['name','pledged','deadline','launched_at','static_usd_rate']
data.drop(to_drop3, inplace=True, axis=1)

#Lets look at the data. Printing summary of the kickstarter data not discarded
print(data.shape)
#This are the unique values across all columns. To determine which can be categorical variables.
print(data.nunique())
unique=data.nunique()
#Find out the columns data types
print(data.info())
datatypes=data.info()
#Summary of the information of each column
print(data.describe())
summary=data.describe()

# Create new columns main_category and sub_category on the new Data Frame data2
data['main_category'] = data['category'].apply(lambda x: x.split(":")[-1].strip("}").strip("\"").split("/")[0])
data['sub_category'] = data['category'].apply(lambda x: x.split(":")[-1].strip("}").strip("\"").split("/")[1])
data.drop("category", inplace=True, axis=1)

#We calculate the pledge per backer for each project
data['pledge_per_backer']=data['usd_pledged']/data['backers_count']

#We find out the distribution of data across state. 
percentage_per_state = round(data["state"].value_counts() / len(data["state"]) * 100,2)
print("State Percent: ")
print(percentage_per_state)
#The higher percentage belong to succesful and failed state, so we can get rid of the rest of the projects that have another category
#We only keep those projects that have values either successful or failed
data2 = data[(data['state'] == 'failed') | (data['state'] == 'successful')]


#There are no missing values in the data frame except for the pledge_per_backer
#variable. It is nan when both usd_pledged and backers are 0.
print(data2.isna().sum()) 
#We fill all nan with zero
data2.fillna(0,inplace=True)
print(data2.isna().sum()) #Now there are no missing values

# Check if the data frame is in appropriate format:
data2.head()

#Finally, we drop the id variable. We dont need it for the models
data2.drop("id", inplace=True, axis=1)


# Plot some bar plots to visualize categorical variables
# def bar_plot(df, col):
#     h = df[col].value_counts()
#     x = pd.DataFrame(df[col].value_counts()).transpose()
#     plt.bar(list(x),h)

# bar_plot(data2, 'country')
# bar_plot(data2, 'year_launched')
# bar_plot(data2, 'month_launched')
# bar_plot(data2, 'main_category')
# bar_plot(data2, 'sub_category')


#Calculating the distribution of projects across the main categories
stateDistCat = pd.get_dummies(data2.set_index('main_category').state).groupby('main_category').sum()
stateDistCat.columns = ['failed', 'successful']
#Finding the correlation of continuous variables with the dependent variable
corr=data2[['backers_count','usd_pledged','usd_goal','duration','name_length','state','pledge_per_backer']].corr()

# Plotting

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(12,12))

data2.groupby('state').state.count().plot(kind='bar', ax=ax1)
ax1.set_title('Number of Projects per State')
ax1.set_xlabel('')

data2.groupby('state').usd_goal.median().plot(kind='bar', ax=ax2)
ax2.set_title('Median project goal ($)')
ax2.set_xlabel('')

data2.groupby('state').usd_pledged.median().plot(kind='bar', ax=ax3)
ax3.set_title('Median project pledged ($)')
ax3.set_xlabel('')

data2.groupby('state').backers_count.median().plot(kind='bar', ax=ax4)
ax4.set_title('Median project backers')
ax4.set_xlabel('')

data2.groupby('state').duration.mean().plot(kind='bar', ax=ax5)
ax5.set_title('Mean project duration')
ax5.set_xlabel('')

data2.groupby('state').name_length.mean().plot(kind='bar', ax=ax6)
ax6.set_title('Mean name length of project')
ax6.set_xlabel('')

staffPickDistr = pd.get_dummies(data2.set_index('state').staff_pick).groupby('state').sum()
staffPickDistr.columns = ['false', 'true']
staffPickDistr.div(staffPickDistr.sum(axis=1), axis=0).true.plot(kind='bar', ax=ax7)
ax7.set_title('Proportion that are staff picks')
ax7.set_xlabel('')

fig.subplots_adjust(hspace=0.6)
plt.show()


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(12,12))

data2.groupby('main_category').main_category.count().plot(kind='bar', ax=ax1)
ax1.set_title('Number of projects')
ax1.set_xlabel('')

data2.groupby('main_category').usd_goal.median().plot(kind='bar', ax=ax2)
ax2.set_title('Median project goal ($)')
ax2.set_xlabel('')

data2.groupby('main_category').usd_pledged.median().plot(kind='bar', ax=ax3)
ax3.set_title('Median pledged per project ($)')
ax3.set_xlabel('')

stateDistCat.div(stateDistCat.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax4)
ax4.set_title('Proportion of successful projects')
ax4.set_xlabel('')

stateDistCat.div(stateDistCat.sum(axis=1), axis=0).failed.plot(kind='bar', ax=ax6)
ax6.set_title('Proportion of failed projects')
ax6.set_xlabel('')

data2.groupby('main_category').backers_count.median().plot(kind='bar', ax=ax5)
ax5.set_title('Median backers per project')
ax5.set_xlabel('')

data2.groupby('main_category').pledge_per_backer.median().plot(kind='bar', ax=ax7)
ax7.set_title('Median pledged per backer ($)')
ax7.set_xlabel('')

fig.subplots_adjust(hspace=0.6)
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,12))

stateDistCountry = pd.get_dummies(data2.set_index('country').state).groupby('country').sum()
stateDistCountry.columns = ['failed', 'successful']

stateDistCountry.div(stateDistCountry.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax1)
ax1.set_title('Proportion of successful projects')
ax1.set_xlabel('')

stateDistMonth = pd.get_dummies(data2.set_index('month_launched').state).groupby('month_launched').sum()
stateDistMonth.columns = ['failed', 'successful']

stateDistMonth.div(stateDistMonth.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax2)
ax2.set_title('Proportion of successful projects')
ax2.set_xlabel('')

stateDistYear = pd.get_dummies(data2.set_index('year_launched').state).groupby('year_launched').sum()
stateDistYear.columns = ['failed', 'successful']

stateDistYear.div(stateDistYear.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax3, color='green')
ax3.set_title('Proportion of successful projects')
ax3.set_xlabel('')

stateDistYear.plot(kind='bar', ax=ax4)
ax4.set_title('Number of failed and successful projects')
ax4.set_xlabel('')

fig.subplots_adjust(hspace=0.6)
plt.show()

#We convert the variables 'successful' state to 1 and failed to 0, to have our logical target variable
data2['state'] = (data2['state'] =='successful').astype(int)
#We convert the variables 'True' state to 1 and 'False' to 0, to have a logical variable staff_pick
data2['staff_pick'] = (data2['staff_pick']).astype(int)

#Since there are too many sub_categories we will drop this column and work with the main_category
data2.drop("sub_category", inplace=True, axis=1)


#Preparing the data for Machine Learning
#First of, we need to create dummy variables for the categorical variables
data3 = pd.get_dummies(data2)

#We then need to seperate into the input data X variables and the target y variables
X1 = data3.drop('state', axis=1)
y = data3.state

#Then, Transform the inputs in X so that they are on the same scale
scale = StandardScaler()
X = pd.DataFrame(scale.fit_transform(X1), columns=list(X1.columns))

#Finally, the data is separated into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=22333)

