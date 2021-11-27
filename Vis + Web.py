import numpy as np 
import pandas as pd
from datetime import datetime

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import json

drought_file_path = '/Users/seacow/Learning/UPenn/CIS 545/Project/US Drought/train_timeseries/train_timeseries.csv'
soil_file_path = '/Users/seacow/Learning/UPenn/CIS 545/Project/US Drought/soil_data.csv'

drought_data = pd.read_csv(drought_file_path)
soil_data = pd.read_csv(soil_file_path)

print(drought_data.columns)

# drop entries without score
drought_data = drought_data.dropna(subset = ['score'])

drought_data.shape

drought_data_joined = drought_data.merge(soil_data, on = 'fips')

print(drought_data_joined.shape)
print(drought_data_joined.columns)

# webscraping code ==================================================================================================

url_FIPS = 'https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697'

r = requests.get(url_FIPS)
html = r.text

soup = BeautifulSoup(html)

# get header and rows of the table
table = soup.find('table', {"class": "data"})
headers = table.find_all('th')
headers = [ele.text for ele in headers]
rows = table.find_all('tr')

# iteratively fill the rows with scraped entry
data = []
for row in rows[1:]:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele])

fips_data = pd.DataFrame(data, columns = headers)

#=====================================================================================================================

# change data type and column name to prepare for merge
fips_data['FIPS'] = fips_data['FIPS'].astype(int)
fips_data.columns = ['fips', 'county', 'state']

# join the two data set so that we know the state and country name from the FIPS code
drought_data_joined = drought_data_joined.merge(fips_data, how = 'left', left_on = 'fips', right_on = 'fips')

# cast date to datetime data type
drought_data_joined['date'] = [datetime.strptime(item, '%Y-%m-%d') for item in drought_data_joined['date']]
drought_data_joined['year'] = [int(ele.year) for ele in drought_data_joined['date']]

# group by state and year to get average yearly drought by state
# this data set will later be used for visualization
drought_data_yearly = drought_data_joined.groupby(['state', 'year']).mean().reset_index()
drought_data_yearly_county = drought_data_joined.groupby(['county', 'year']).mean().reset_index()

# Visualiation ==========================================================================================================

# check data types
drought_data_joined.dtypes

# check distribution of label
plt.figure(figsize = (8, 6), dpi = 100)
drought_data_joined['score'] = round(drought_data_joined['score'])
drought_data_joined['score'] = drought_data_joined['score'].astype(int)
sns.displot(drought_data_joined['score'], bins = 6)
plt.xlabel('Severeity of Drought (The higher the more severe)')
plt.ylabel('Count')
plt.title('Label Distribution of the US Drought Data')

sns.displot(drought_data_joined['state'], aspect = 4)
plt.xlabel('State Name')
plt.ylabel('Count')
plt.title('State Distribution of the US Drought Data')

px.choropleth(drought_data_yearly, 
              locations = 'state',
              color = "score", 
              range_color = (0, 3),
              animation_frame = "year",
              color_continuous_scale = "fall",
              locationmode = 'USA-states',
              scope = "usa",
              title = "Drought Severity by State"
             )

px.choropleth(drought_data_yearly_county, 
              geojson = counties, 
              locations = 'fips', 
              range_color = (0, 3),
              color = 'score',
              animation_frame = "year",
              color_continuous_scale = "fall",
              scope = 'usa',
              title = "Drought Severity by County",
              )

drought_corr = drought_data_joined.corr()
# creating mask
mask = np.triu(np.ones_like(drought_corr))
 
# plotting a triangle correlation heatmap
plt.figure(figsize = (16, 16), dpi = 120)
sns.heatmap(drought_corr, annot = False, mask = mask)
plt.title('Correlation Matrix of Features in the US Drought Data')

px.scatter(drought_data_yearly, 
           x = "GRS_LAND", 
           y = "score", 
           animation_frame = "year", 
           animation_group = "state",
           size = "GRS_LAND", 
           color = "state", 
           hover_name = "state")

score_corr = pd.DataFrame(abs(drought_corr['score']))
most_correlated = score_corr.sort_values(['score'], ascending = False).head(11).reset_index()[1:]


# modeling ====================================================================================================================================

# time series prediction based solely on drought data
drought_data.dtypes

drought_data_useful_col = ['PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET',
                         'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX',
                         'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN',
                         'WS50M_RANGE', 'score']

drought_data_train = drought_data[drought_data_useful_col]
drought_data_train = drought_data_train.reset_index(drop = True)

drought_data_train['score'] = drought_data_train['score'].astype(int)

# balance label
drought_data_train_grouped = drought_data_train.groupby('score')
drought_data_train_balanced = drought_data_train_grouped.apply(lambda x: x.sample(drought_data_train_grouped.size().min()).reset_index(drop = True))

# check distribution of label
plt.figure(figsize = (8, 6), dpi = 100)
sns.displot(drought_data_train_balanced['score'], bins = 6)
plt.xlabel('Severeity of Drought (The higher the more severe)')
plt.ylabel('Count')
plt.title('Label Distribution of the US Drought Data After Down Sampling')

drought_data_train_balanced = drought_data_train_balanced.sample(frac = 1, random_state = 545)

window_size = 30
num_labels = 5
num_features = drought_data_train_balanced.shape[1]

train_features_mtx = []
train_labels_mtx = []

# we could have this many samples... but these are too much for neural net (takes too long to train)
# num_samples = len(drought_data_train_balanced) // (window_size + num_labels)
# to make model trains within a reasonable time, we limit the number of samples
num_samples = 5000

# initial case
init_feat_mtx = np.array(drought_data_train_balanced[0:30])
train_features_mtx.append(init_feat_mtx)

init_label_vec = np.array(drought_data_train_balanced['score'][30:35])
train_labels_mtx.append(init_label_vec)

start_idx = window_size + num_labels
end_idx = window_size * 2 + num_labels

# iterative case
for i in range(1, num_samples):
    feat_mtx = np.array(drought_data_train_balanced[start_idx: end_idx])
    label_vec = np.array(drought_data_train_balanced['score'][end_idx: end_idx + 5])
    
    train_features_mtx.append(feat_mtx)
    train_labels_mtx.append(label_vec)
    
    # update index
    start_idx = end_idx + 5
    end_idx = start_idx + window_size


