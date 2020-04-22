# CLEANING PROCESS & RESULTS

We will follow a thorough process during the cleaning. This process will contain the following phases:

1 - First of all we study the missing values for every row in the data.
2 - Then we will create the correct data structure for obtaining all the necessary columns that will be used during the process and removing those that will not be used.
3 - Afterwards, we will perform per variable analysis. We will study all the different aspects for each variable and try to identify possible knowledge propmpted from analysing independently each variable.
4 - Then, we will study possible relations within variables and compute useful transformations that we may find during the analysis.
5 - Finally we will prepare the data for being able to used it in our models.

TODO: As we will need to obtain data from our source that still has not been seen by our models, we will need to compute a python script for reading the raw data, generating the dataframe, clean it and prepare it for our model being able to process it.


## 1 - MISSING VALUES

We found that we have nearly non NA values along the rows. Only a total of 1090 rows, and most of them (1072) along the location column. Remember that location column contained a JSON string from which we could extract location related values such as country, state or city. This corresponds to only 0.31% of all of our data with missing values.

After grouping the missing values by country, we obtained 4 different countries (AU, GB, DE, US). Except for US which had a total of 1087 rows from the 1090, the others had only 1 missing value.

We proceed to create the data structure that we want.

## DATA STRUCTURE CREATION

### CURRENCY COLUMNS
First of all we convert all currency values to USD values for working always with the same currency values. Note that we have the columns `[goal, pledged, usd_pledged, static_usd_rate]` are all columns referred to currency values. We consider here that goal is stated in local currency value instead of USD as well as pledged amount. We use the static_usd_rate values to convert tose into usd values. As a matter of checking our results, we will check the value obtained against the usd_pledged value to see if there are any incongruity.

We observed that the max value of the difference is 9.31e-10, which we see it is totally .

### DATE COLUMNS
We convert date realted columns to datetime using pd.to_datetime(data, unit='s') function. It is important to notice that this type allows us to perform multiple operations on the date, such as extract the year, month, day or day of the week. The columns that are converted are `[launched_at, created_at, deadline]`.

From these columns we will create additional columns.

### LOCATION COLUMNS


## TABLEAU INTEGRATION

For being able to open the dataset in Tableau and generate superior charts, we created a function for extracting the dataset into a readable .hyper file in Tableau.

It is important to know that Tableau can connect to Excel and many other types of files, but for the purpose of assuring that we correctly open the dataset in Tableau we decided to generate by ourselves directly from the pandas dataframe a structure and file with extension .hyper, which is the one that uses Tableau.

We used the Extract API 2.0 from Tableau for completing this work.