#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 01:19:10 2019

@author: agericke
"""

# First import the necessary modules
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import re
import os, sys

# %matplotlib inline

from urllib.request import urlopen
from bs4 import BeautifulSoup
from io import BytesIO
from zipfile import ZipFile

sys.stdout.write("Python executing correctly \n")

# Initial directories set up
dirname = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.abspath(os.path.join(os.path.join(dirname, os.pardir), os.pardir)), 'data/')

sys.stdout.write("Directory name is %s \n" % dirname)
sys.stdout.write("Datadir is %s \n" % datadir)
# Data URL from webroots
url = "https://webrobots.io/kickstarter-datasets"
html = urlopen(url)

soup = BeautifulSoup(html, 'lxml')

# Find all links for downloading the data
all_links = list(soup.find_all('a'))
all_hrefs = set()
"""
Pick only the zips containing 2015 or 2016 datasets by default. In case we pass
an argument when running, that argument must be a year and we will pick the zips
of that year.
"""
for link in all_links:
    if type(link.get('href')) == str:
        #  By default only pick csv files from 2016 and 2017
        exp = "https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_201[6-7]"
        if len(sys.argv) > 1:
            # Create condition for verifying that the argument consists of a year.
            exp = str("https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_%s" % sys.argv[1])
        if re.search(exp, link.get('href')):
            # Pick only the csv files
            if link.get('href').__contains__("zip"):
                all_hrefs.add(link.get('href'))

all_hrefs = sorted(all_hrefs)

"""
For every zip obtained, check if directory with extracted data from that zip exists,
and if not create the directory and extract all csv in it.
"""
for href in all_hrefs:
    zipfile = urlopen(href)
    zipfile = ZipFile(BytesIO(zipfile.read()))
    #Check if directory exists
    namepath = re.sub("_", "", re.search('_[\w,-]+_', href).group())
    zipdir = os.path.join(datadir, namepath)
    if not os.path.isdir(zipdir):
        os.makedirs(zipdir)
        zipfile.extractall(zipdir)
