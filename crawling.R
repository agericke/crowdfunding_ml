# Obtaining data from web sources
# Webscraping (very basic introduction)
install.packages('rvest') # Webscraping package
install.packages('stringr') # Package that makes manipulating strings (text) easier
install.packages('tidyr') # Package for making data manipulation easier

library(rvest)
library(stringr)
library(tidyr)

#Set years limit to download data. In our case we will work with data from 2016 to 2018
ini_year <- 2016
end_year <- 2018

for (i in ini_year:end_year) {
  
}