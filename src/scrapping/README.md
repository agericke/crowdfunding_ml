# SCRAPPING

This folder contains a python script that will gather all the zip files from webroots that contain csv files (not the JSON ones), create a directory for each of the zip files encountered and extract all csv files under that directory.

By default, it picks only csv files of 2016 and 2017. But it is possible to determine a specific year for acquiring the csv files of projects from a specific year.

Beware that the script will store the data and directories under the **data/** directory.


For running the script just open a terminal and execute:

`python scrapping.py`

This will run the script with the default mode (gather data from 2016 and 2017). In order to specify a year (2018 for example) you should execute the following:

`python scrapping.py 2018`

Actual things still to be done:

* Check that the argument introduced has an appropiate format.
* In case that format is incorrect, inform user or just run default mode.
* Create shell script to run the python script.