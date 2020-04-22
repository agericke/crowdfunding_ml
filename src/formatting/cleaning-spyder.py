import json
import math
import pandas as pd
import numpy as np
import os, sys
import datetime
from datetime import date
import time
import pickle

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
import seaborn as sns
#from ggplot import *

from tableausdk import *
from tableausdk.HyperExtract import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#%matplotlib inline
sns.set()

def initial_setup():
    """
    Create Initial setup of directories variables, and dataframe vars to use.
    Returns:
      A tuple containing:
          - datadir:   Absolute Path to the data directory of the project.
          - dirname:   Absolute Path of directory that contains this file.
          - imagesdir: Absolute path of directory that contains the images.
          - colnames: A list containing the initial colnames of the dataframe.
    """
    # Initial directories set up
    dirname = os.path.dirname(os.path.abspath('__file__'))
    datadir =  os.path.join(os.path.abspath(os.path.join(os.path.join(dirname, os.pardir), os.pardir)), 'data')
    imagesdir =  os.path.join(os.path.abspath(os.path.join(dirname, os.pardir)), 'images')
    initial_colnames = sorted(['backers_count', 'blurb', 'category', 'country', 'created_at', 'currency', 'deadline', 'goal', 'id', 'launched_at', 'location', 'pledged', 'slug', 'spotlight', 'staff_pick', 'state', 'static_usd_rate', 'usd_pledged'])
    return dirname, datadir, imagesdir, initial_colnames


def read_from_disk(filename):
    """
    Read a dataframe from a filename in disk.
    Params:
        filename....Path to the file.
    Returns:
        A pandas dataframe.
    """
    return pickle.load(open(filename, 'rb'))


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


def remove_cols(dataframe, cols_to_remove):
    """
    Remove all the columns specified by the list from dataframe
    Params:
        cols_to_remove....List of columns we want to remove
        dataframe.........The dataframe to remove the columns from.
    Returns:
        A dataframe with only the columns we want.
    """
    dataframe.drop(cols_to_remove, inplace=True, axis=1)
    print("Succesfully removed columns {}".format(cols_to_remove))
    return dataframe


def categorical_with_per_count(dataframe, feature):
    """
    Calculate frequency of the categorical feature with % and count base.
    Sorted on the descending order.

    Params:
        dataframe.....Pandas dataframe from where to pick the data.
        feature.......Column for which we want to calculate the data for.
    """
    
    # calculate frequency on % and value
    freq_merged = pd.concat([dataframe[feature].value_counts(normalize=True) * 100,
                             dataframe[feature].value_counts(normalize=False)], axis=1)
    # rename columns
    freq_merged.columns = [feature + '_%', feature + '_count']
    return freq_merged


def check_missing_values_and_drop(data, drop=False):
    """
    Check the number of missing values that we have. Notice that this function
    will count 
    Params:
        data....Dataframe to check the missing values.
        drop....Boolean to indicate if we want to drop missing values or not.
    Returns:
        Prints a summary of the number and % of missing values.
        The dataframe with no missing values
    """
    #dataMis=data[data.isnull().any(axis=1)]
    #print(dataMis.groupby('country').country.count())
    total_rows = data.shape[0]
    #isna() sets True NA values and numpy.NaN. Empty strings or infinites are not set as True.
    na_col_counts = data.isna().sum().sort_values(ascending = False)
    freq_merged = pd.concat([na_col_counts, (na_col_counts/total_rows)*100], axis=1)
    freq_merged.columns = ['Total_count', '%_count']

    if drop:
        data = data.dropna()
    
    return data, freq_merged
    # TODO: See if we can check the missing indexes for each column and run a study on them.
    # TODO: Run experiments to try to identify is the missing values are mainly because of a reason or one type of project, or specific to one period of time (see if they are missing at random, missing not at random...)
    
    
def check_na_column(data, column, value, print_cols):
    """
    Function to print info regarding specific coolumn missing values.
    
    Params:
        data.........Pandas dataframe
        column.......Column code to check
        value........Column value to fiter data
        print_cols...Array of columns to print
    Returns:
        Nothing.
    """
    print(data[data[column] == value][print_cols])
    print('\n')

    
def obtain_cat_and_subcat_for_row(row):
    """
    Obtain category and subcategory of a row. notice that if subcategory not present,
    this will set it to none.
    Params:
        row....The dataframe row.
    Returns:
        A tuple containing:
            - category: Value for obtained category.
            - subcategory: Value for obtained subcategory, and None if not present.
    """
    # Convert string to dict
    cat_split = json.loads(row['category'])['slug'].split('/')
    # Save category and subcategory
    category = cat_split[0]
    subcategory = ""
    
    if len(cat_split)>1:
        # There is a sucategory
        subcategory = cat_split[1]
    
    return (category, subcategory)
    

def create_cat_and_subcat(data):
    """
    Save in a new column the category and subcategory extracted from the category column.
    Params:
        data.....Dataframe.
    Returns:
        A dataframe with category and subcategory vars created.¡ for each row.
    """
    cat_subcat = data.apply(obtain_cat_and_subcat_for_row, axis=1)
    data['main_category'] = [c[0].lower().strip().replace(' ','-') for c in cat_subcat]
    data['sub_category'] = [c[1].lower().strip().replace(' ','-') for c in cat_subcat]
    data.drop('category', inplace=True, axis=1)
    print('Succesfully created columns category and subcategory')
    
    return data


def obtain_location_vars_for_row(row):
    """
    Obtain country, state and city.
    Params:
        row....The dataframe row.
    Returns:
        A tuple containing:
            - country: Country oobtained.
            - state: State obtained.
            - loc_type: Type of location obtained.
    """
    # Convert string to dict
    loc_dict = json.loads(row['location'])
    country = loc_dict['country']
    state = loc_dict['state']
    city =  loc_dict['name']
    
    return (country, state, city)
    

def create_location_vars(data):
    """
    Save in a new column the country, state and location type extracted from the location column.
    Params:
        data.....Dataframe.
    Returns:
        A dataframe with country, state and location vars created for each row.
    """
    location_vars = data.apply(obtain_location_vars_for_row, axis=1)
    data['country2'] = [c[0] for c in location_vars]
    data['state'] = [c[1] for c in location_vars]
    data['city'] = [c[2] for c in location_vars]
    data.drop("location", inplace=True, axis=1)
    print("Succesfully created columns country, region_state and type")
    
    return data


def barplot_date_xaxis(data, y_format, title, filename, use_x_formatter=False, x_rotation=45):
    """
    Create a bar plot with the specified parameters. Data indexes must be of type dateIndex.
    
    Params:
        data..............pandas dataframe to pick the values from.
        y_format..........format of the y axis ticks.
        title.............Plot title
        filename..........File name and pathwhere to store the figure.
        use_x_formatter...Bool to indicate whether to use a fixed fomratter or not.
        x_rotation........X ticks rotation.
    
    Returns:
        Nothing.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if use_x_formatter:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    ax.bar(year_projs.index, year_projs.values)
    vals = ax.get_yticks()
    ax.set_yticklabels([y_format.format(y/1000) for y in vals])
    ax.xaxis.set_tick_params(rotation=x_rotation)
    
    fig.tight_layout()
    fig.savefig(filename, dpi=fig.dpi)


# You cannot delete or append to a .tde file if it is opened in Tableau
def to_tde(dataframe, filename, tb_name='Kickstarter'):
    """
    Function for creating an hyper dataset for Tableua from the pandas dataframe.
    
    Params:
        - dataframe.....Pandas dataframe to create the extract from.
        - filename......Path to the file where we will store the .hyper extract.
        - tb_name.......Name of the table to be created. (Deafault: Kickstarter)
        
    Returns:
        Nothing
    """    
    # 0 - Initialize extract API
    ExtractAPI.initialize()

    # Step 1: Create the Extract File
    dataExtract = Extract(filename)

    if dataExtract.hasTable(tb_name):
        return print("tde already exist use another name")

    # Step 2: Create the table definition
    if (not dataExtract.hasTable(tb_name)):
        dataSchema = TableDefinition()
        dataSchema.addColumn('backers_count', Type.INTEGER)
        dataSchema.addColumn('blurb', Type.UNICODE_STRING)
        dataSchema.addColumn('category', Type.CHAR_STRING)
        dataSchema.addColumn('country', Type.CHAR_STRING)
        dataSchema.addColumn('created_at', Type.DATETIME)
        dataSchema.addColumn('currency', Type.CHAR_STRING)
        dataSchema.addColumn('deadline', Type.DATETIME)
        dataSchema.addColumn('id', Type.INTEGER)
        dataSchema.addColumn('launched_at', Type.DATETIME)
        dataSchema.addColumn('location', Type.UNICODE_STRING)
        dataSchema.addColumn('slug', Type.UNICODE_STRING)
        dataSchema.addColumn('spotlight', Type.BOOLEAN)
        dataSchema.addColumn('staff_pick', Type.BOOLEAN)
        dataSchema.addColumn('state', Type.CHAR_STRING)
        dataSchema.addColumn('goal_usd', Type.DOUBLE)
        dataSchema.addColumn('pledged_usd', Type.DOUBLE)

        # Step 3: Create a table in the image of the table definition
        table = dataExtract.addTable(tb_name, dataSchema)

    # Step 4: Create rows and insert them one by one
    newRow = Row(dataSchema)
    for i in range(0, len(dataframe)):
        newRow.setInteger(0, dataframe['backers_count'].iloc[i])
        newRow.setString(1, dataframe['blurb'].iloc[i])
        newRow.setCharString(2, dataframe['category'].iloc[i])
        newRow.setCharString(3, dataframe['country'].iloc[i])
        newRow.setDateTime(4, dataframe['created_at'].iloc[i].year, dataframe['created_at'].iloc[i].month, dataframe['created_at'].iloc[i].day, dataframe['created_at'].iloc[i].hour, dataframe['created_at'].iloc[i].minute, dataframe['created_at'].iloc[i].second, dataframe['created_at'].iloc[i].microsecond)
        newRow.setCharString(5, dataframe['currency'].iloc[i])
        newRow.setDateTime(6, dataframe['deadline'].iloc[i].year, dataframe['deadline'].iloc[i].month, dataframe['deadline'].iloc[i].day, dataframe['deadline'].iloc[i].hour, dataframe['deadline'].iloc[i].minute, dataframe['deadline'].iloc[i].second, dataframe['deadline'].iloc[i].microsecond)
        newRow.setInteger(7, dataframe['id'].iloc[i])
        newRow.setDateTime(8, dataframe['launched_at'].iloc[i].year, dataframe['launched_at'].iloc[i].month, dataframe['launched_at'].iloc[i].day, dataframe['launched_at'].iloc[i].hour, dataframe['launched_at'].iloc[i].minute, dataframe['launched_at'].iloc[i].second, dataframe['launched_at'].iloc[i].microsecond)
        newRow.setString(9, dataframe['location'].iloc[i])
        newRow.setString(10, dataframe['slug'].iloc[i])
        newRow.setBoolean(11, dataframe['spotlight'].iloc[i])
        newRow.setBoolean(12, dataframe['staff_pick'].iloc[i])
        newRow.setCharString(13, dataframe['state'].iloc[i])
        newRow.setDouble(14, dataframe['goal_usd'].iloc[i])
        newRow.setDouble(15, dataframe['pledged_usd'].iloc[i])
        
        table.insert(newRow)

    # Step 5: Close the tde
    dataExtract.close()
    
    # Step 6: Close the Tableau Extract API
    ExtractAPI.cleanup()


# 0 - Initial directories and colnames set up
print("Step 0: Initial directories and colnames set up")
dirname, datadir, imagesdir, initial_colnames = initial_setup()
print("Directory of this file is {}".format(dirname))
print("Data directory is {}".format(datadir))
print("Images directory is {}".format(imagesdir))
print("Initial columns for our model are: \n{}".format(initial_colnames))


# 1 - Load from disk the complete Merged Dataframe.
print("\n\n\nStep 1: Load from disk the complete Merged Dataframe.")
filename = os.path.join(datadir, 'dataframe_total.pkl')
print("Completed Dataframe read from file {}".format(filename))
data = read_from_disk(filename)
# Print summary of dataframe
print("Dataframe contains {} projects and {} columns for each project\n".format(data.shape[0], data.shape[1]))


# 2 - Take a look at the data, data types and data info.
print(data.head(5))
print(data.describe())
print(data.info())
print(data.get_dtype_counts())


# 3 - Look for missing values for every row and print summary.
print("\n\n\nStep 2: Look for missing values for every row and print summary.")
data, na_freq = check_missing_values_and_drop(data, False)
print("As we can see, we have very low percentage of missing values, with a total of {} rows\
that correspond to {:.2f}%".format(int(na_freq.sum()[0]), na_freq.sum()[1]))
print('The columns with at least one missing value are:')
print('{}'.format(na_freq[na_freq.Total_count > 0]))
# Take a look at distribution per country
df_null = data[data.isna().any(axis=1)]
df_null.country.value_counts().sort_values(ascending = False)
print('Taking a look at missing values per countries we find:')
print('{:-<10}{:->10}'.format('Country', '#_NA'))
[print('{:<10}{:>10}'.format(row[0], row[1])) for row in df_null.country.value_counts().sort_values(ascending = False).items()]
# We drop NA values
data, na_freq = check_missing_values_and_drop(data, True)


# 4 - Re-format columns

# 4-1 Change state column to result
data['result'] = data['state']
data.drop('state', inplace=True, axis=1)

# 4-2 Re-factor money related columns
data['goal_usd'] = data['goal']*data['static_usd_rate']
data.goal_usd.head(10)
# Create pledged_usd
data['pledged_usd'] = data['pledged']*data['static_usd_rate']
data.pledged_usd.head(10)
# Study differences from calculated to the value we had.
print(abs(data['usd_pledged'] - data['pledged_usd']).describe())
# Remove not any more useful cols.
data = remove_cols(data, ['goal', 'pledged', 'static_usd_rate', 'usd_pledged'])

# 4-3 Create date type variables for created_at, launched_at and deadline
data['launched_at'] = pd.to_datetime(data['launched_at'], unit='s')
data['deadline'] = pd.to_datetime(data['deadline'], unit='s')
data['created_at'] = pd.to_datetime(data['created_at'], unit='s')

# 4-4 Create category and sub-category columns
data = create_cat_and_subcat(data)

# 4-5 Create location columns.
df2 = create_location_vars(data)
data[data.country != data.country2]
# After some comparison we determine that country2 is more realiable than country.
data['country'] = data['country2']
data.drop('country2', inplace=True, axis=1)


# 5 - Exploratory Data Analysis.
#Things to do:
#    - Study data.
#    - Check valid key for data.
#    - Study of missing values.
#    - Study per variable.
#        + Study objective variable.
# 5-1 Overview of the data
data.info()
data.describe()
data.describe(include='all')

# 5-2 Check valid key
(data.id.value_counts() > 1).sum()
data.set_index('id', inplace=True)

# 5-3 Check missing data
data, na_freq = check_missing_values_and_drop(data, False)
print("As we can see, we have very low percentage of missing values, with a total of {} rows\
that correspond to {:.2f}%".format(int(na_freq.sum()[0]), na_freq.sum()[1]))
print('The columns with at least one missing value are:')
print('{}'.format(na_freq[na_freq.Total_count > 0]))
# Lets study twhere does missing values come from
df_null = data[data.isna().any(axis=1)]
unique_na_cities = np.sort(df_null.city.unique())
unique_na_countries = np.sort(df_null.country.unique())
print(unique_na_cities)
print(unique_na_countries)
# We see that they correspond to very small countries that may not contain states. So we set
# as the state the country. Fo rdoing so we make a dict from country code to country name.
country_to_name = {'AQ': 'Antarctica', 'AW': 'Aruba', 'CC': 'Cocos Islands', 'CW': 'Curacao',
                   'GI': 'Gibraltar', 'KI': 'Kiribati', 'MK': 'Macedonia', 'MO': 'Macau',
                   'NZ': 'New Zealand', 'PN': 'Pitcairn', 'SX': 'Sint Maarten', 'VA': 'Vatican',
                   'XK': 'Kosovo'}    
[check_na_column(data, 'country', country, ['country', 'state', 'city']) for country in unique_na_countries]

city_to_state = {'MK': {'Skopje': 'Skopje'}, 'NZ': {'Taupo': 'Waikato', 'Rotorua Central': 'Bay of Plenty', 
                 'Dannevirke': 'Manawatu Wanganui', 'Rotorua West': 'Bay of Plenty',
                 'Whanganui': 'Manawatu Wanganui', 'Taihape': 'Manawatu Wanganui', 
                 'Rotorua': 'Bay of Plenty', 'Oamaru Central': 'Otago', 'Stratford': 'Taranaki',
                 'South Oamaru': 'Otago'}}
country_to_state = {'AQ': 'Antarctica', 'AW': 'Aruba', 'CC': 'Cocos Islands', 'CW': 'Curacao',
                   'GI': 'Gibraltar', 'KI': 'Kiribati', 'MO': 'Macau', 'PN': 'Pitcairn',
                   'SX': 'Sint Maarten', 'VA': 'Vatican', 'XK': 'Kosovo'}

for country in country_to_name.keys():
    if country in country_to_state.keys():
        data.loc[data[data['country'] == country].index, 'state'] = country_to_state[country]
    else:
        cities = city_to_state[country]
        for city in cities.keys():
            data.loc[data[(data['country'] == country) & (data['city'] == city)].index, 'state'] = cities[city]
            
data, na_freq = check_missing_values_and_drop(data, False)
print('The columns with at least one missing value are:')
print('{}'.format(na_freq[na_freq.Total_count > 0]))

# 5-4 Per variable study.
# 5-4.1 Objective variable.
print(data['result'].value_counts().sort_values(ascending=False))
print('We see that we have a total of 5 different project results')
print('Successful and failed projects account for {:.2f}% of all projects of the dataset'.format(
        data['result'].value_counts(normalize=True).loc[['successful', 'failed']].sum()*100
        )
    )

fig = plt.figure()
ax = fig.add_subplot(111)
data.result.value_counts(normalize=True).sort_values(ascending=False).plot(kind='bar', ax=ax)
ax.set_yticklabels(['{:.0%}'.format(x) for x in ax.get_yticks()])
ax.xaxis.set_tick_params(labelrotation=0, size=20)
fig.tight_layout()
filename = os.path.join(imagesdir, 'pct_result.png')
fig.savefig(filename, dpi=fig.dpi)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
year_projs = data.groupby(pd.Grouper(key='created_at', freq='Y')).size()
ax.bar(year_projs.index.year, year_projs.values)
ax.yaxis.grid(True, linestyle='-.', zorder=1, color='lightgrey')
ax.set_yticklabels(['{:.0f}K'.format(x/1000) for x in ax.get_yticks()])
fig.tight_layout()
filename = os.path.join(imagesdir, 'projs_per_year.png')
fig.savefig(filename, dpi=fig.dpi)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
year_projs = data.groupby([pd.Grouper(key='created_at', freq='Y'), 'result']).size().unstack()
year_projs.plot(kind='bar', ax=ax, color=['orange', 'red', 'blue', 'green', 'yellow'])
ax.xaxis.set_ticklabels(year_projs.index.year.values)
ax.xaxis.set_tick_params(labelrotation=0, size=15)
#ax.yaxis.grid(True, linestyle='-.', zorder=1, color='lightgrey')
ax.set_yticklabels(['{:.0f}K'.format(x/1000) for x in ax.get_yticks()])
fig.tight_layout()
filename = os.path.join(imagesdir, 'dist_result_per_year.png')
fig.savefig(filename, dpi=fig.dpi)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
spotlight_projs = data.groupby(['spotlight', 'result']).size().unstack()
spotlight_projs.plot(kind='bar', ax=ax, stacked=True, color=['orange', 'red', 'blue', 'green', 'yellow'])
#ax.xaxis.set_ticklabels(year_projs.index.values)
ax.xaxis.set_tick_params(labelrotation=0, size=15)
#ax.yaxis.grid(True, linestyle='-.', zorder=1, color='lightgrey')
ax.set_yticklabels(['{:.0f}K'.format(x/1000) for x in ax.get_yticks()])
fig.tight_layout()
filename = os.path.join(imagesdir, 'spotlight_result_stacked.png')
fig.savefig(filename, dpi=fig.dpi)
plt.show()
print('We see that Spotlight is directly related to the result of the project and ith as value True\
only if the project was successful and False otherwise. This way we ca nget rid of the column as\
it does ont add valuable information')
# Drop spotlight column.
data.drop('spotlight', inplace=True, axis=1)
# Pick only [successful or failed] projects.
data = data[data.result.isin(['successful', 'failed'])].copy()

# Study per year products state with two possible values only.
fig = plt.figure()
ax = fig.add_subplot(111)
year_projs = data.groupby([pd.Grouper(key='created_at', freq='Y'), 'result']).size().unstack()
year_projs.plot(kind='bar', ax=ax, color=['red', 'green'])
ax.xaxis.set_ticklabels(year_projs.index.year.values)
ax.xaxis.set_tick_params(labelrotation=0, size=15)
ax.set_yticklabels(['{:.0f}K'.format(x/1000) for x in ax.get_yticks()])
fig.tight_layout()
filename = os.path.join(imagesdir, 'result_bool_per_year.png')
fig.savefig(filename, dpi=fig.dpi)
plt.show()

# Study total number of projects per state
fig = plt.figure()
ax = fig.add_subplot(121)
data.result.value_counts().plot(kind='bar', ax=ax, color=['green', 'red'])
ax.xaxis.set_tick_params(labelrotation=0, size=15)
ax.set_yticklabels(['{:.0f}K'.format(x/1000) for x in ax.get_yticks()])
ax.set_title('# of projects per result')
ax = fig.add_subplot(122)
data.result.value_counts(normalize=True).plot(kind='bar', ax=ax, color=['green', 'red'])
ax.xaxis.set_tick_params(labelrotation=0, size=15)
ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax.get_yticks()])
ax.set_title('% of projects per result')
fig.suptitle('Projects per State')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
filename = os.path.join(imagesdir, 'projs_per_result.png')
fig.savefig(filename, dpi=fig.dpi)
plt.show()

# study project result by staff_pick value
fig = plt.figure()
ax = fig.add_subplot(111)
staff_projs = data.groupby(['staff_pick', 'result']).size().unstack()
staff_projs.plot(kind='bar', ax=ax, color=['red', 'green'])
ax.set_yticklabels(['{:.0f}K'.format(x/1000) for x in ax.get_yticks()])
ax.xaxis.set_tick_params(rotation=0)
fig.tight_layout()
plt.show()

fig = plt.figure()
for i, value in enumerate(staff_projs.index.values):
    ax = fig.add_subplot(1,2,int(i+1))
    df = staff_projs.loc[value,:] /staff_projs.loc[value,:].sum()
    df.plot(kind='bar', ax=ax, color=['red', 'green'])
    ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax.get_yticks()])
    ax.xaxis.set_tick_params(rotation= 0)
    ax.set_title('% result for {} value'.format(value))
fig.suptitle('Projects result for Staff Pick values') 
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Lets change column result to bool type with True if succesful.
data['result'] = data['result'].map({'successful': True, 'failed': False})

#5-4.1 backers_count.
# First of all lets plot a histogram of the backers count variable.
print('Although we study the backers_count variable, this is not available prior to\
a project launch. So we may eliminate this variable before modeling.')
sns.distplot(data.backers_count, kde=True)
print('We see that it is highly positive skewed. Lets plot per category.')
ax = plt.gca()
sns.boxplot(data=data, x='result', y='backers_count', ax=ax)
print('We observe the wider and outliers for successful projects')

print('Lets try to apply a log transformation to the variable. We need to add 1 for\
avoiding mistakes from 0 values')
data['ln(backers_count+1)'] = np.log(data.backers_count +1)
fig = plt.figure()
ax = sns.distplot(data[data.result]['ln(backers_count+1)'], kde=True, label='successful')
sns.distplot(data[~data.result]['ln(backers_count+1)'], ax=ax, label='failed')
ax.legend()
print('We can see that succesful projects follow an appproximate normal distribution while\
the failed ones are decreasing from 0. The shape for the failed ones is expected as\
failed projects will have few backers_count.')
fig = plt.figure()
#sns.stripplot(x="result", y="backers_count", data=data, jitter=0.1, orient="v")
# TODO: Before plotting we need to establich the variable as categorical.