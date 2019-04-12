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
to_drop2 = ['photo','name','blurb','currency','currency_symbol','currency_trailing_code','state_changed_at',
            'slug','created_at','creator','location','spotlight','static_usd_rate','staff_pick','profile','urls','source_url']
data.drop(to_drop2, inplace=True, axis=1)

#Extract the categories: main_category and the category
#category=data['category']
#for i in data.iterrows(): 
 #   cat=category[i]
  #  category[i]=cat.get('slug')
   
#Create variable duration
data["duration"] = data["deadline"]-data["launched_at"]
#Transform the variable to days


#Create the variable Month


  
#Lets look at the data. Printing summary of the kickstarter data not discarded
print(data.shape)
#This are the unique values across all columns. To determine which can be categorical variables.
print(data.nunique())
#Find out the columns data types
print(data.info())
#Summary of the information of each column
print(data.describe())


#We find out the distribution of data across state. 
percentage_per_state = round(data["state"].value_counts() / len(data["state"]) * 100,2)
print("State Percent: ")
print(percentage_per_state)
#The higher percentage belong to succesful and failed state, so we can get rid of the rest of the projects that have another category

#We only keep those projects that have values either successful or failed
data2 = data[(data['state'] == 'failed') | (data['state'] == 'successful')]
#We convert the variables 'successful' state to 1 and failed to 0, to have our logical target variable
data2['state'] = (data2['state'] =='successful').astype(int)
