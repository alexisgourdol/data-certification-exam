# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Imports & dataset loading

# <codecell>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# <codecell>

# Load the dataset
from data import load_data_viz_data

raw_data = load_data_viz_data()
raw_data[['LAT','LONG']] = raw_data[['LAT','LONG']].astype('float64')

print("Shape of the DataFrame:", raw_data.shape)

raw_data.head()

# <codecell>

data = raw_data.copy()

# <markdowncell>

# ### Timing

# <codecell>

timing_df = data.copy()
timing_df['OCCURRED_ON_DATE'] = pd.to_datetime(timing_df['OCCURRED_ON_DATE'])

# <codecell>

timing_df['day'] = timing_df['OCCURRED_ON_DATE'].dt.day
timing_df['month'] = timing_df['OCCURRED_ON_DATE'].dt.month
timing_df['year'] = timing_df['OCCURRED_ON_DATE'].dt.year
timing_df

# <codecell>

timing_df = pd.concat([timing_df, pd.get_dummies(timing_df.OFFENSE_CODE_GROUP)], axis=1)

# <codecell>



# <codecell>

timing_df.groupby('year').sum()[['Disputes','Drugs and disorderly conduct',\
                             'Fraud and law violations','Larceny and vandalism',\
                             'Other','Police investigation procedure','Violence and harassment']].astype(int).plot()
plt.legend(bbox_to_anchor=(1.05, 1))
ax = plt.gca()
fig = plt.gcf()
fig.set_size_inches((16,8));

# <codecell>

f, ax = plt.subplots(figsize=(16, 8))
with sns.color_palette("PuBuGn_d"):
    timing_df.groupby('OFFENSE_CODE_GROUP').count()

# <codecell>

tmp = timing_df.groupby(['year', 'month']).count()[['INCIDENT_NUMBER']]#.reset_index()
with sns.color_palette("PuBuGn_d"):
    tmp.plot()

# <codecell>

#timing_df.set_index('OCCURRED_ON_DATE', inplace=True)

# <codecell>

f, ax = plt.subplots(figsize=(16, 8))
with sns.color_palette("PuBuGn_d"):
    timing_df.groupby([(timing_df.index.year),(timing_df.index.month)]).count()['INCIDENT_NUMBER'].plot()

# <codecell>

timing_df.resample('1M').sum()['SHOOTING']

# <codecell>

f, ax = plt.subplots(figsize=(16, 8))
with sns.color_palette("PuBuGn_d"):
    timing_df.resample('1M').sum()['SHOOTING'].plot(kind='bar')
    plt.title('Shootings per month')

# <codecell>

f, ax = plt.subplots(figsize=(16, 8))
with sns.color_palette("PuBuGn_d"):
    timing_df.resample('1Q').sum()['SHOOTING'].plot(kind='bar')
    plt.title('Shootings per Quarter')

# <codecell>

timing_df.resample('1Y').sum()['SHOOTING']

# <codecell>

timing_df.resample('1Q').sum()['SHOOTING'].plot(kind='bar')

# <codecell>

f, ax = plt.subplots(figsize=(16, 8))
with sns.color_palette("PuBuGn_d"):
    sns.barplot(y=distance_df.index, x=distance_df.AVG_DISTANCE, orient="h")
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 24), ylabel="",
       xlabel="Average distance between Police Station and crime location")
    sns.despine(left=True, bottom=True)

# <codecell>

timing_df

# <markdowncell>

# ### Distance

# <codecell>

# Haversine distance function
from math import radians, sin, cos, asin, sqrt

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Compute distance (km) between two pairs of (lat, lng) coordinates
    See - (https://en.wikipedia.org/wiki/Haversine_formula)
    """
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    
    return 2 * 6371 * asin(sqrt(a))

# <codecell>

distance_df = data.copy()
distance_df['AVG_DISTANCE'] = distance_df[['LAT', 'LONG', 'LAT_POLICE_STATION', 'LONG_POLICE_STATION']].apply(lambda x: haversine_distance(*x), axis=1)
distance_df = distance_df.groupby(['NAME']).mean()[['AVG_DISTANCE']].sort_values(by='AVG_DISTANCE', ascending=False)
distance_df

# <codecell>

data

# <codecell>

f, ax = plt.subplots(figsize=(16, 8))
with sns.color_palette("PuBuGn_d"):
    sns.barplot(y=distance_df.index, x=distance_df.AVG_DISTANCE, orient="h")
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 24), ylabel="",
       xlabel="Average distance between Police Station and crime location")
    sns.despine(left=True, bottom=True)

# <codecell>


