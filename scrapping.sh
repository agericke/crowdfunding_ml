#!/bin/bash
#.-*-.ENCONDING:-UTF-8.-*-

# Script for downloading all the data

NUMARG=$#

if [ $NUMARG -gt "0" ]
then
	python ./src/scrapping/scrapping.py sys.argv[1]
else 
	python ./src/scrapping/scrapping.py
fi