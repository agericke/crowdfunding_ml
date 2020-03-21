# Merging Results & Documentation.

We have obtained a total of x rows that represent x distinct projects. One of the key decisions taken during this process has been the selection of the columns that we will use for our data.

## Columns Differences by Year

We did an analysis of the columns present in the data for each of the years that we study (2016, 2017, 2018, 2019). We tried to identify differences in columns between years. We obtained that the year with most columns was 2019 and the differences in columns between the different years were:

- __Diff for 2019 and 2018__: ['source_url']
- __Diff for 2019 and 2017__: ['converted_pledged_amount', 'current_currency', 'fx_rate', 'is_starrable', 'usd_type']
- __Diff for 2019 and 2016__: ['converted_pledged_amount', 'current_currency', 'fx_rate', 'is_starrable', 'usd_type']
- __Diff for 2018 and 2017__: ['converted_pledged_amount', 'current_currency', 'fx_rate', 'is_starrable', 'usd_type']
- __Diff for 2018 and 2016__: ['converted_pledged_amount', 'current_currency', 'fx_rate', 'is_starrable', 'usd_type']
- __Diff for 2017 and 2016__ :[]

As we can see, between 2016 and 2017 there are no differences in the columns. Between 2019 and 2017 or 2016, the columns `['converted_pledged_amount', 'current_currency', 'fx_rate', 'is_starrable', 'usd_type']` are not present in the 2017 data nor in the 2016 data. We have checked the meaning of these columns and they are not critical at all, actually there are plenty of NaN values within these columns in the 2019 data.

Finally we can see 2018 is the only year that does not contain the column `source_url`. This column provides with an url for the subcategory in which is classified the specific project. This data can be obtained from other columns so it is not critical at all.

Given the study over the differences and information provided by each of these columns, we will drop all columns that are not present in all years. That means that all columns within the list `['converted_pledged_amount', 'current_currency', 'fx_rate', 'is_starrable', 'source_url', 'usd_type']` willl be dropped.

## Columns Selection

After dropping the columns specified before, we have a total of 31 columns: `['backers_count', 'blurb', 'category', 'country', 'created_at', 'creator', 'currency', 'currency_symbol', 'currency_trailing_code', 'deadline', 'disable_communication', 'friends', 'goal', 'id', 'is_backing', 'is_starred', 'launched_at', 'location', 'name', 'permissions', 'photo', 'pledged', 'profile', 'slug', 'spotlight', 'staff_pick', 'state', 'state_changed_at', 'static_usd_rate', 'urls', 'usd_pledged']`.

 

