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
            'slug','created_at','creator','location','spotlight','profile','urls','source_url']
data.drop(to_drop2, inplace=True, axis=1)

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

#We find out the distribution of data across state. 
percentage_per_state = round(data["state"].value_counts() / len(data["state"]) * 100,2)
print("State Percent: ")
print(percentage_per_state)
#The higher percentage belong to succesful and failed state, so we can get rid of the rest of the projects that have another category
#We only keep those projects that have values either successful or failed
data2 = data[(data['state'] == 'failed') | (data['state'] == 'successful')]
#We convert the variables 'successful' state to 1 and failed to 0, to have our logical target variable
data2['state'] = (data2['state'] =='successful').astype(int)
