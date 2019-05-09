#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: agericke
"""

import pandas as pd
import numpy as np
import os, sys
import time
import pickle
from collections import defaultdict
from collections import Counter


def initial_setup():
    """
    Create Initial setup of directories variables, and dataframe vars to use.
    Returns:
      A tuple containing:
          - datadir: Absolute Path to the data directory of the project.
          - dirname: Absolute Path of directory that conatins this file.
          - colnames: A list containing the colnames we will use in every file.
    """    
    # Initial directories set up
    dirname = os.path.dirname(os.path.abspath('__file__'))
    datadir =  os.path.join(os.path.abspath(os.path.join(os.path.join(dirname, os.pardir), os.pardir)), 'data')
    colnames = sorted(['backers_count', 'blurb', 'category', 'country', 'created_at', 'state_changed_at', 'currency', 'deadline', 'goal', 'id', 'launched_at', 'location', 'name', 'pledged', 'state', 'static_usd_rate', 'usd_pledged'])
    return dirname, datadir, colnames


def obtain_all_directories(path):
    """
    Obtain all directories present in a path.
    Params:
        path....Tha abolute path.
    Returns
        A sorted list of all directories.
    """
    return sorted([os.path.join(path, o) for o in os.listdir(path) 
                    if os.path.isdir(os.path.join(path,o))])

    
def obtain_files_in_directory(path):
    """
    Obtain all files inside a directory.
    Params:
        path....The directory path from where to obtain all the files.
    Returns:
        A sorted list of paths to the files.
    """
    files = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return sorted(files)


def obtain_all_files(dirs_list):
    """
    Obatain all files for each of the directories from a list of a directories.
    Params:
        dirs_list....A list of directories paths.
    Returns:
        A list of all files' path inside each of the directories.
    """
    files = []
    for directory in dirs_list:
        files.extend(obtain_files_in_directory(directory))
    return sorted(files)
    

def read_dataframe(path):
    """
    Read data from the file provided and return it as a pandas dataframe.
    Params:
        path.....The absolute path to the dataframe.
    Returns:
        A pandas dataframe.
    """
    #pd.read_csv(os.path.join(datadir, "2016-05-15T020446/Kickstarter001.csv"))
    return pd.read_csv(path, enconding='ISO-8859-1')


def sort_dataframe_by_columns(dataframe):
    """
    Pick the colnames for a dataframe, and returned the dataframe sorted by the colnames.
    Params:
        dataframe.....The pandas Dataframe.
    Returns:
        A pandas dataframe ordered by column name.
    """
    return dataframe.reindex(sorted(dataframe.columns), axis=1)


def check_if_colnames_present(dataframe, fix_colnames, path):
    """
    Check if a dataframe has all colnames necessary for our job.
    Params:
        dataframe.....The pandas Dataframe.
        fix_colnames..The list of colnames we need to check for.
        path..........Path of the file we formed the dataframe from.
    Returns:
        A tuple than contains:
            - A logical indicating if it contains all the columns or not.
            - A list of the non present columns.
    """
    df_cols = sorted(dataframe.columns)
    not_present_columns = []
    for col in fix_colnames:
        if col not in df_cols:
            not_present_columns.append(col)
    if len(not_present_columns) > 0:
        return False, not_present_columns
        print("File %s missing columns %s" % (path, not_present_columns))
    else:
        print("File %s is OK." % path)
        return True, not_present_columns
        

def check_all_files(file_list, fix_columns):
    """
    For every file, convert to dataframe and check if cols present.
    Params:
        file_list.....List of all files path.
        fix_colnames..List of colnames we need to check for.
    Returns:
        A list of tuples with each tuple containing:
            - A logical indicating if it contains all the columns or not.
            - A list of the non present columns.
            - The path to the file.
    """
    results = []
    for file in file_list:
        df = sort_dataframe_by_columns(read_dataframe(file))
        all_cols, not_present_columns = check_if_colnames_present(df,fix_columns, file)
        result = tuple([all_cols, not_present_columns, file])
        results.append(result)
    for result in results:
        if result[0] == False:
            print("File %s missing columns %s" % (result[2], result[1]))
    return results   


def remove_cols(fix_columns, dataframe):
    """
    Pick only the columns we want.
    Params:
        fix_colnames..List of colnames we will pick.
        dataframe.....The dataframe to remove the columns from.
    Returns:
        A dataframe with only the columns we want.
    """
    return dataframe[fix_columns]


def merge_dataframes_by_rows(df1, df2):
    """
    Merge two dataframes by rows
    Params:
        df1....The first dataframe
        df2....The second dataframe
    Returns:
        A single dataframe that results from the merge of the other two
    """
    frames = [df1, df2]
    return pd.concat(frames)


def merge_and_remove_duplicates(df1, df2):
    """
    This function will merge two dataframes and will return a single dataframe as a
    result of the merge, but it will remove the duplicates, considering only the id
    column for looking for duplicates. We will maintain the last appearance of the 
    duplicates as the unique one, and remove the others.
    Params:
        df1....The first dataframe
        df2....The second dataframe
    Returns:
        A single dataframe with unique id values.
    """
    df_merge = merge_dataframes_by_rows(df1, df2)
    # Obtain indexes of duplicates.
    duplicated_rows = df_merge.duplicated("id", keep='last')
    # Pick only the non-duplicated rows
    df_merge = df_merge[~duplicated_rows]
    return df_merge


def create_full_dataframe(all_files, colnames):
    """
    Function for creating the full dataframe from all the files. It will call the necessary
    defined functions for merging dataframes, removing the unsued columns, sorting the dataframes
    and removing the duplicates keeping the last apppearance of a project as the good one.
    
    It is required that the files are stored in the list sorted by date, being the initial
    ones the oldest, o that when merging, the last rows are the most recent.
    Params:
        all_files.....The list of files for creating the general dataframe from.
        colnames......The lis of columns that we will use for the project.
    Returns:
        A Dataframe that contains all the projects from the files and with the duplicated rows removed.
    """
    # Create the initial dataframe from the first project.
    first_file = all_files[0]
    all_files = all_files[1:] # Remove first file from all the files.
    df = remove_cols(colnames, sort_dataframe_by_columns(read_dataframe(first_file)))
    # Iterate over all the remaining files.
    for file in all_files:
        df_toadd = remove_cols(colnames, sort_dataframe_by_columns(read_dataframe(file)))
        df = merge_and_remove_duplicates(df, df_toadd)
    return df
    

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
    

def read_from_disk(filename):
    """
    Read a dataframe from a filename in disk.
    Params:
        filename....Path to the file.
    Returns:
        A pandas dataframe.
    """
    return pickle.load(open(filename, 'rb'))


def main():
    # 0 - Initial directories set up
    dirname, datadir, colnames = initial_setup()
    print("Directory of this file is %s" % dirname)
    print("Data directory is %s" % datadir)
    print("Columns for our model are: \n%s" % colnames)
    
    # 1 - Obtain all directories with data from project folders.
    data_dirs = obtain_all_directories(datadir)
    print("Directories with csv files are:")
    for directory in data_dirs:
        print(directory)
    
    # 2 - Obtain all files paths within directories.
    all_files = obtain_all_files(data_dirs)
    
    # # 3 - Check that all files have the required columns for our project.
    # results_col_checks = check_all_files(all_files, colnames)
    # # Print summary of results.
    # bad_files = 0
    # good_files = 0
    # for result in results_col_checks:
    #     if result[0] == False:
    #         bad_files += 1
    #     if result[0] == True:
    #         good_files += 1
    # print("We have %d bad files" % bad_files)
    # print("We have %d good files" % good_files)
            
    # 4 - Create full dataframe from all files, removing duplicates
    df_total = create_full_dataframe(all_files, colnames)
    
    # 5 - Store dataframe in disk
    filename = os.path.join(datadir, 'dataframe_total.pkl')
    print("Dataframe total is going to be saved to %s" % filename)
    store_dataframe(df_total, filename)


if __name__ == "__main__":
    main()