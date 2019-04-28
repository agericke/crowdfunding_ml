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

def initial_setup():
    """
    Create Initial setup of directories variables.
    Returns:
      A tuple containing:
          - datadir: Absolute Path to the data directory of the project.
          - dirname: Absolute Path of directory that conatins this file.
          - url: The url from where we are going to extract the data.
    """    
    # Initial directories set up
    dirname = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(os.path.abspath(os.path.join(os.path.join(dirname, os.pardir), os.pardir)), 'data/')
    # Data URL from webroots
    url = "https://webrobots.io/kickstarter-datasets"
    return dirname, datadir, url

def open_html(url):
    """
    Open html and generate the BeautifulSoup Object.

    Params:
        url...The url from where we want to obtain the html
    Returns:
        The BeautifulSoup Object containing the html as python Objects.
    """
    html = urlopen(url)
    return BeautifulSoup(html, 'lxml')


def obtain_hrefs(soupObject):
    """
    Obtain all hrefs from the list of links.

    Params:
        soupObject...The BeautifulSoup Object from which obtain the links.
    Returns:
        A list containing all the hrefs from the links sorted.
    """
    # Find all links for downloading the data
    all_links = list(soupObject.find_all('a'))
    all_hrefs = set()
    # =============================================================================
    #     Pick only the zips containing 2015 or 2016 datasets by default. In case we pass
    #     an argument when running, that argument must be a year and we will pick the zips
    #     of that year.
    # =============================================================================
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
    return sorted(all_hrefs)


def download_and_persist(hrefs, datadir):
    """
    Download and persist the data in the data directory.
    
    Params:
        hrefs...List of all the hrefs we want to download the data from.
        datadir...The directory where we want to store the data.
    """
    
    # =============================================================================
    #     For every zip obtained, check if directory with extracted data from that zip exists,
    #     and if not create the directory and extract all csv in it.
    # =============================================================================
    for href in hrefs:
        zipfile = urlopen(href)
        zipfile = ZipFile(BytesIO(zipfile.read()))
        #Check if directory exists
        namepath = re.sub("_", "", re.search('_[\w,-]+_', href).group())
        sys.stdout.write("Downloading data from %s.\n" % namepath)
        zipdir = os.path.join(datadir, namepath)
        if not os.path.isdir(zipdir):
            sys.stdout.write("Creating directory %s and storing files under that that directory.\n" % zipdir)
            os.makedirs(zipdir)
            zipfile.extractall(zipdir)
            
    
def main():
    if len(sys.argv) > 1:
        sys.stdout.write("Running scrapping for storing Kickstarter data from year %s.\n" % str(sys.argv[1]))
    else:
        sys.stdout.write("Running scrapping for storing Kickstarter data from years 2016 and 2017.\n")
    dirname, datadir, url = initial_setup()
    sys.stdout.write("Directory name is %s \n" % dirname)
    sys.stdout.write("Datadir is %s \n" % datadir)
    sys.stdout.write("URL: is %s \n" % url)
    
    # Obtain the soupObject form the html
    soup = open_html(url)
    # Obtain the hrefs for downloading the data from.
    all_hrefs = obtain_hrefs(soup)
    # Download and persist the data
    download_and_persist(all_hrefs, datadir)
    sys.stdout.write("Scrapping completed.\n")
    
    
if __name__ == '__main__':
    main()
