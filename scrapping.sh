#!/bin/bash
#.-*-.ENCONDING:-UTF-8.-*-

# Script for downloading all the data

NUMARG=$#

if [ $NUMARG -gt "0" ]
then
	echo "More than one argument"
	echo $1
	python ./src/scrapping/scrapping.py $1
else 
	python ./src/scrapping/scrapping.py
fi