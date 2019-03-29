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
dirname = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.abspath(os.path.join(os.path.join(dirname, os.pardir), os.pardir)), 'data/')

"""
Need to create a big dataframe. For each csv file we should add the data to the dataframe,
check if the column names coincide and expand the dataframe.

Loop through all folders, and add a column to each project that represents the year
and the season of the year.
"""
data = pd.read_csv(os.path.join(datadir, "2016-05-15T020446/Kickstarter001.csv"))