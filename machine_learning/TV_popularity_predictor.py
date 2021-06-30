# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # TV shows Popularity Predictor (39%)
# 
# The goal of this challenge is to create a model that predicts the `popularity` of a movie or TV show
# 
# <img src="image.jpg" width=300 />
# 
# 
# 
# 
# The dataset contains a list of movies and TV shows with the following characteristics:
# - `title`: title of the movie in english
# - `original_title`: original title of the movie 
# - `duration_min`: duration of the movie in minutes
# - `popularity`: popularity of the movie in terms of review scores
# - `release_date`: release date
# - `description`: short summary of the movie
# - `budget`: budget spent to produce the movie in USD
# - `revenue`: movie revenue in USD 
# - `original_language`: original language 
# - `status`: is the movie already released or not
# - `number_of_awards_won`: number of awards won for the movie
# - `number_of_nominations`: number of nominations
# - `has_collection`: if the movie is part of a sequel or not
# - `all_genres`: genres that described the movie (can be zero, one or many!) 
# - `top_countries`: countries where the movie was produced (can be zero, one or many!) 
# - `number_of_top_productions`: number of top production companies that produced the film if any. 
# Top production companies includes: Warner Bros, Universal Pictures, Paramount Pictures, Canal+, etc...
# - `available_in_english`: whether the movie is available in english or not

# <markdowncell>

# ## Imports
# 
# Run the following cell to load the basic packages:

# <codecell>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nbresult import ChallengeResult

# <codecell>

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_validate

# Execute this cell to enable a nice display for your pipelines
from sklearn import set_config; set_config(display='diagram')
import math

# <markdowncell>

# ## Data collection
# 
# 📝 **Load the `movie_popularity.csv` dataset from the provided this [URL](https://wagon-public-datasets.s3.amazonaws.com/certification_france_2021_q2/tv_movies_popularity.csv)**
# - First, check and remove the rows that may be complete duplicate from one another (we never know!)
# - Then, drop the columns that have too much missing values
# - Finally, drop the few remaining rows that have missing values
# - Store the result in a `DataFrame` named `data`

# <codecell>

data = pd.read_csv('tv_movies_popularity.csv')
print(data.shape)
data.head(3)

# <codecell>

data.drop_duplicates(inplace=True)
data.shape #400 duplicate lines removed

# <codecell>

for col in data.columns:
    print(f'{col} has {data[col].isnull().sum() / data.shape[0] *100}% missing values')

# <codecell>

data.info()

# <codecell>

data.drop(columns='revenue', inplace=True)
data.shape #dropped 1 columns

# <codecell>

data.isnull().sum(axis=1).value_counts()

# <codecell>

data.dropna(axis=0, inplace=True)
data.shape #dropped 1 line

# <markdowncell>

# ### 🧪 Run the following cell to save your results

# <codecell>

from nbresult import ChallengeResult

result = ChallengeResult(
    "data_cleaning",
    columns=data.columns,
    cleaning=sum(data.isnull().sum()),
    shape=data.shape)
result.write()

# <markdowncell>

# ## Baseline model

# <markdowncell>

# ### The metric

# <markdowncell>

# 📝 **We want to predict `popularity`: Start by plotting a histogram of the target to visualize it**

# <markdowncell>

# 📝 **Which sklearn's scoring [metric](https://scikit-learn.org/stable/modules/model_evaluation.html) should we use if we want it to:**
# 
# - Be better when greater (i.e. metric_good_model > metric_bad_model)
# - Penalize **more** an error between 10 and 20 compared with an error between 110 and 120
# - Said otherwise, what matter should be the **relative error ratio**, more than the absolute error difference
# 
# Hint: the histogram plotted above should give you some intuition about it
# 
# 👉 Store its exact [sklearn scoring name](https://scikit-learn.org/stable/modules/model_evaluation.html) as `string` in the variable `scoring` below.
# 
# 🚨 You must use this metric for the rest of the challenge

# <codecell>

# Distribution 

# <codecell>

data.popularity.min(),data.popularity.max()

# <codecell>

sns.histplot(data.popularity, kde=False);

# <codecell>

sns.boxplot(x=data.popularity);

# <codecell>

sns.boxplot(x=data.popularity.sort_values()[:-30]);

# <codecell>

# Metric for regression that penalizes more on the low range ==> mean-squared-log-error
# https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error
# 'Note that this metric penalizes an under-predicted estimate greater than an over-predicted estimate.'

scoring = 'neg_mean_squared_log_error'

# <markdowncell>

# <details>
#     <summary>💡 Hint</summary>
# It is around here!
# <img src="scores.jpg" width=200 height=400 />
# </details>

# <markdowncell>

# ### X,y

# <markdowncell>

# **📝 Define `X` as the features Dataframe (keep all features) and `y` as the target Series.**

# <codecell>

X = data.copy().drop(columns='popularity')
y = data.popularity.copy()
print(X.columns, '\n', y.name)
X.shape, y.shape

# <codecell>

X.dtypes

# <markdowncell>

# ### Basic pipeline

# <markdowncell>

# 📝 **Check unique values per features**

# <codecell>

for col in X.columns:
    print(f'{col} has {X[col].nunique()} unique values \n {X[col].unique()}  \n\n')

# <markdowncell>

# In this baseline, let's forget about the columns below that are difficult to process

# <codecell>

text = ['description', 'original_title', 'title']
dates = ['release_date'] 

# <markdowncell>

# We will simply scale the numerical features and one-hot-encode the categorical ones remaining
# 
# 📝 **Prepare 2 `list`s of features names as `str`**:
# - `numerical` which contains **only** numerical features
# - `categorical` which contains **only** categorical features (exept text and dates above)

# <codecell>

X.columns

# <codecell>

X_baseline = X.drop(columns=['description', 'original_title', 'title', 'release_date'])
X_baseline.dtypes

# <codecell>

numerical   = ['duration_min', 'budget', 'number_of_awards_won', 'number_of_nominations', \
               'number_of_top_productions']
categorical = ['original_language','status', 'has_collection', 'available_in_english', 'all_genres',\
               'available_in_english']

print(len(numerical + categorical))
assert len(numerical + categorical) == len(X_baseline.columns)

# <codecell>

X_baseline

# <markdowncell>

# ### Pipelining
# 
# You are going to build a basic pipeline made of a basic preprocessing and a trees-based model of your choice.

# <markdowncell>

# #### Preprocessing pipeline
# 
# **📝 Create a basic preprocessing pipeline for the 2 types of features above:**
# - It should scale the `numerical` features
# - one-hot-encode the `categorical` and `boolean` features
# - drop the others
# - Store your pipeline in a `basic_preprocessing` variable

# <markdowncell>

# **📝 Encode the features and store the result in the variable `X_basic_preprocessing`.**

# <codecell>

num_transformer = RobustScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# Paralellize "num_transformer" and "One hot encoder"
preprocessor = ColumnTransformer([
    ('num_transformer', num_transformer, numerical),
    ('cat_transformer', cat_transformer, categorical)
])

basic_preprocessing = make_pipeline(preprocessor)

print(type(basic_preprocessing))
basic_preprocessing

# <codecell>

X_basic_preprocessing = basic_preprocessing.fit_transform(X)
X_basic_preprocessing

# <codecell>

X_basic_preprocessing.shape

# <codecell>

# Paralellize "num_transformer" and "One hot encoder"
preprocessor2 = ColumnTransformer([
    ('num_transformer', num_transformer, numerical),
    ('cat_transformer', cat_transformer, categorical)], 
    remainder='passthrough') #as the CT is at the beginning of the pipe, 
#we'll need the other columns for the other transformers below => remainder=‘passthrough’) 

basic_preprocessing2 = make_pipeline(preprocessor2)
print(type(basic_preprocessing2))
basic_preprocessing2

# <markdowncell>

# **❓ How many features has been generated by the preprocessing? What do you think about this number?**

# <markdowncell>

# There are 801 features generated by the preprocessing. We might want to reduce it with PCA for instance, or reduce even further the features selected from the original dataframe

# <markdowncell>

# #### Modeling pipeline
# 
# Let's add a model to our pipe. With so many features one-hot-encoded, we **need a model which can act as a feature selector**
# 
# 👉 A linear model regularized with L1 penalty is a good starting point.
# 
# 
# **📝 Create a `basic_pipeline` which encapsulate the `basic_preprocessing` pipeline + a linear model with a L1 penalty**
# 
# - store the resulting pipeline as `basic_pipeline`
# - don't fine-tune it
# 
# 
# <details>
#     <summary>Hints</summary>
# 
# Choose your model from the list [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
# 
# </details>

# <codecell>

basic_pipeline = make_pipeline(
    basic_preprocessing,
    ElasticNet()
)
basic_pipeline

# <markdowncell>

# ### Cross-validated baseline
# 
# **📝 Perform a cross-validated evaluation of your baseline model using the metric you defined above. Store the results of this evaluation as an `array` of floating scores in the `basic_scores` variable.**

# <codecell>

from sklearn.metrics import SCORERS
[metric for metric in sorted(SCORERS.keys()) if 'log' in metric]

# <codecell>

basic_scores_all = cross_validate(basic_pipeline, X, y, cv=5,
                              scoring=(scoring),
                              return_train_score=True)
basic_scores_all

# <codecell>

basic_scores = basic_scores_all['test_score']
basic_scores

# <markdowncell>

# ### 🧪 Save your results
# 
# Run the following cell to save your results

# <codecell>

ChallengeResult(
    'baseline',
    metric=scoring,
    features=[categorical,numerical],
    preproc=basic_preprocessing,
    preproc_shape=X_basic_preprocessing.shape,
    pipe=basic_pipeline,
    scores=basic_scores
).write()

# <markdowncell>

# ## Feature engineering

# <markdowncell>

# ### Time Features
# 
# 
# 👉 Let's try to improve performance using the feature `release_date`, and especially its `month` and `year`.
# 
# ℹ️ If you want to skip this section, you can move directly to the next one: _Advanced categorical features_.

# <markdowncell>

# **📝 Complete the custom transformer `TimeFeaturesExtractor` below**
# 
# Running
# ```python
# TimeFeaturesExtractor().fit_transform(X[['release_date']])
# ``` 
# should return something like
# 
# |    |   month |   year |
# |---:|--------:|-------:|
# |  0 |       2 |   2015 |
# |  1 |       8 |   2004 |
# |  2 |      10 |   2014 |
# |  3 |       3 |   2012 |
# |  4 |       8 |   2012 |


# <codecell>

from sklearn.base import BaseEstimator, TransformerMixin

class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extract the 2 time features from a date"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """
        Params:
        X: DataFrame
        y: Series
        
        Returns a DataFrame with 2 columns containing the time features as integers extracted from the release_date.
        """
        df = pd.concat((pd.to_datetime(X['release_date']).dt.month, pd.to_datetime(X['release_date']).dt.year), axis = 1)
        df.columns = ['month', 'year']
        return df

# <codecell>

# Try your transformer and save your new features here
X_time_features = TimeFeaturesExtractor().fit_transform(X[['release_date']])
X_time_features.head()

# <markdowncell>

# We still have 2 problems to solve
# - `month` is cyclical: 12 should be a close to 1 as to 11, right? 
# - `year` is not scaled
# 
# **📝 Build a final custom transformer `CyclicalEncoder` so that**
# 
# Running
# ```python
# CyclicalEncoder().fit_transform(X_time_features)
# ``` 
# should return something like this
# 
# |    |    month_cos |   month_sin |      year |
# |---:|-------------:|------------:|----------:|
# |  0 |  0.5         |    0.866025 | 0.0466039 |
# |  1 | -0.5         |   -0.866025 | 0.0411502 |
# |  2 |  0.5         |   -0.866025 | 0.0461081 |
# |  3 |  6.12323e-17 |    1        | 0.0451165 |
# |  4 | -0.5         |   -0.866025 | 0.0451165 |
# 
# With the cyclical encoding is done as below
# - `month_cos = 2 * math.pi / 12 * X[['month']] `
# - `month_sin = 2 * math.pi / 12 * X[['month']] `
# 
# And the `year` begin min-max scaled

# <codecell>

from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode a cyclical feature
    """
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Compute here what you need for the transform phase and store it as instance variable
        """
        #self.month_cos = np.cos(2 * math.pi / 12 * X[['month']])
        #self.month_sin = np.sin(2 * math.pi / 12 * X[['month']])
        #self.year = RobustScaler().fit_transform(X[['year']])
        
        return self

    def transform(self, X, y=None):
        """
        Compute and returns the final DataFrame
        """
                
        month_cos = np.cos(2 * math.pi / 12 * X[['month']])
        month_sin = np.sin(2 * math.pi / 12 * X[['month']])
        year = pd.DataFrame(RobustScaler().fit_transform(TimeFeaturesExtractor().fit_transform(X)[['year']]))
        
        X_transformed = pd.concat([month_cos, month_sin, year], axis=1)
        X_transformed.columns = ['month_cos', 'month_sin', 'year']
        return  X_transformed

# <codecell>

m = TimeFeaturesExtractor().fit_transform(X)[['month']]
c = np.cos(2 * math.pi / 12 * m)
s = np.sin(2 * math.pi / 12 * m)
year = pd.DataFrame(RobustScaler().fit_transform(TimeFeaturesExtractor().fit_transform(X)[['year']]))
d = pd.concat([c, s, year], axis=1)
d.columns = ['month_cos', 'month_sin', 'year']
d

# <codecell>

# Try your transformer and save your new features here
X_time_cyclical = d.dropna()
#ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
X_time_cyclical.head() 

# <codecell>

# Check that this form a circle with 12 points
plt.scatter(X_time_cyclical['month_cos'],
            X_time_cyclical['month_sin'])
plt.xlabel("month_cos"); plt.ylabel("month_sin");

# <markdowncell>

# **📝 Enhance your `basic_pipeline` with a new preprocessing including both `TimeFeaturesExtractor` and `CyclicalFeatureExtractor`:**
# 
# - Just use `TimeFeatureExtractor` if you haven't had time to do the `Cyclical` one
# - Store this new pipeline as `time_pipeline`
# - Keep same estimator for now

# <codecell>

time_pipeline = make_pipeline(
    basic_preprocessing2,
    TimeFeaturesExtractor(),
    #CyclicalEncoder(), #Implementation in the Class above fails
    ElasticNet()
)
time_pipeline

# <markdowncell>

# ### Advanced categorical encoder to reduce the number of features
# 
# ℹ️ Most of it has already been coded for you and it shouldn't take long. Still if you want to skip it and move to the next section: _Model Tuning_

# <markdowncell>

# 👉 We need to reduce the number of features to one-hot-encode, which arise from the high cardinality of `all_genres` and `top_countries`

# <codecell>

X[['all_genres', 'top_countries']].nunique()

# <markdowncell>

# 👇 Both share a common pattern: there can be more than 1 country and more than 1 genre per movie.

# <codecell>

X[['all_genres', 'top_countries']].tail()

# <markdowncell>

# 👉 Run the cell below where we have coded for you a custom transformer `CustomGenreAndCountryEncoder` which: 
# - Select the 10 most frequent genres and the 5 most frequent countries
# - Encode `all_genres` into 10 One Hot Encoded features
# - Encode `top_countries` into 5 One Hot Encoded features

# <codecell>

from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

class CustomGenreAndCountryEncoder(BaseEstimator, TransformerMixin):
    """
    Encoding the all_genres and top_companies features which are multi-categorical :
    a movie has several possible genres and countries of productions!
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        compute top genres and top countries of productions from all_genres and top_countries features
        """

        # compute top 10 genres       
        list_of_genres = list(X['all_genres'].apply(lambda x: [i.strip() for i in x.split(",")] if x != [''] else []).values)
        top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(10)]

        # save top_genres in dedicated instance variable
        self.top_genres = top_genres
        
         # compute top 5 countries       
        list_of_countries = list(X['top_countries'].apply(lambda x: [i.strip() for i in x.split(",")] if x != [''] else []).values)
        top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(5)]

        # save top_countries in dedicated instance variable
        self.top_countries = top_countries

        return self

    def transform(self, X, y=None):
        """
        encoding genre and country
        """
        X_new = X.copy()
        for genre in self.top_genres:
            X_new['genre_' + genre] = X_new['all_genres'].apply(lambda x: 1 if genre in x else 0)
        X_new = X_new.drop(columns=["all_genres"])
        for country in self.top_countries:
            X_new['country_' + country] = X_new['top_countries'].apply(lambda x: 1 if country in x else 0)
        X_new = X_new.drop(columns=["top_countries"])
        return X_new

# <codecell>

# Check it out
X_custom = CustomGenreAndCountryEncoder().fit_transform(X[['all_genres', 'top_countries']])
print(X_custom.shape)
X_custom.head()

# <markdowncell>

# **📝 Compute your `final_pipeline` by integrating all these transformers** (or all those you have coded)
# 
# - `CustomGenreAndCountryEncoder`
# - `TimeFeaturesExtractor`
# - `CyclicalFeatureExtractor`

# <codecell>

final_pipeline_attempt = make_pipeline(
    basic_preprocessing2,
    TimeFeaturesExtractor(),
    #CyclicalEncoder(), #Implementation in the Class above fails
    CustomGenreAndCountryEncoder(),
    ElasticNet()
)

final_pipeline_attempt

# <codecell>

ct = ColumnTransformer([
    ('num_transformer', num_transformer, numerical),
    ('cat_transformer', cat_transformer, ['original_language','status', 'has_collection', 'available_in_english']),
    ('time_transformer', TimeFeaturesExtractor(), ['release_date']),
    ('genre_country_transformer', CustomGenreAndCountryEncoder(), ['all_genres', 'top_countries'])
])

final_pipeline = make_pipeline(
    ct,
    ElasticNet()
)

final_pipeline

# <markdowncell>

# 📝 **Compute and store its cross validated scores as `final_scores` array of floats**
# 
# - It does not necessarily improve the performance before we can try-out doing model tuning
# - However, with a now limited number of features, we will be able to train more complex models in next section (ensemble...)

# <codecell>

final_scores_all = cross_validate(final_pipeline, X, y, cv=5,
                              scoring=(scoring),
                              return_train_score=True)
final_scores = final_scores_all['test_score']
final_scores

# <markdowncell>

# ### 🧪 Save your result
# 
# Run the following cell to save your results.

# <codecell>

ChallengeResult(
    'feature_engineering',
    X_time_features=X_time_features,
    X_time_cyclical= X_time_cyclical,
    time_pipeline=time_pipeline,
    final_pipeline=final_pipeline,
    final_scores=final_scores
).write()

# Hint: Try restarting your notebook if you obtain an error about saving a custom encoder

# <markdowncell>

# ## Model tuning

# <markdowncell>

# ### Random Forest

# <markdowncell>

# 📝 **Change the estimator of your `final_pipeline` by a Random Forest and checkout your new cross-validated score**

# <codecell>

final_pipeline_forest = make_pipeline(
    ct,
    RandomForestRegressor()
)

final_pipeline_forest

# <markdowncell>

# ### Best hyperparameters quest
# 
# 
# 
# **📝 Fine tune your model to try to get the best performance in the minimum amount of time!**
# 
# - Store the result of your search inside the `search` variable.
# - Store your 5 cross-validated scores inside `best_scores` array of floats

# <codecell>

search = None
best_scores = np.array([.0, .0, .0, .0, .0])

# <markdowncell>

# **📝 Re-train your best pipeline on the whole (X,y) dataset**
# - Store the trained pipeline inside the `best_pipeline` variable

# <codecell>

best_pipeline = final_pipeline_forest.fit(X, y)
best_pipeline

# <markdowncell>

# ### Prediction
# 
# Now you have your model tuned with the best hyperparameters, you are ready for a prediction.
# 
# Here is a famous TV show released in 2017:
# 
# ```python
# dict(
#         original_title=str("La Casa de Papel"),
#         title=str("Money Heist"), 
#         release_date= pd.to_datetime(["2017-05-02"]), 
#         duration_min=float(50),
#         description=str("An unusual group of robbers attempt to carry out the most perfect robbery"), 
#         budget=float(3_000_000), 
#         original_language =str("es"), 
#         status=str("Released"),
#         number_of_awards_won =int(2), 
#         number_of_nominations=int(5), 
#         has_collection=int(1),
#         all_genres=str("Action, Crime, Mystery"), 
#         top_countries=str("Spain, France, United States of America"), 
#         number_of_top_productions=int('1'),
#         available_in_english=bool('True') 
# )
# ```
# 
# **📝 Compute the predicted popularity of this TV show and store it into the `popularity` variable as a floating number.**

# <codecell>

show_to_predict = pd.DataFrame(dict(
        original_title=str("La Casa de Papel"),
        title=str("Money Heist"), 
        release_date= pd.to_datetime(["2017-05-02"]), 
        duration_min=float(50),
        description=str("An unusual group of robbers attempt to carry out the most perfect robbery"), 
        budget=float(3_000_000), 
        original_language =str("es"), 
        status=str("Released"),
        number_of_awards_won =int(2), 
        number_of_nominations=int(5), 
        has_collection=int(1),
        all_genres=str("Action, Crime, Mystery"), 
        top_countries=str("Spain, France, United States of America"), 
        number_of_top_productions=int('1'),
        available_in_english=bool('True') 
))

# <codecell>

show_to_predict

# <codecell>

popularity = final_pipeline_forest.predict(show_to_predict)
popularity

# <markdowncell>

# ### 🧪 Save your results
# 
# Run the following cell to save your results.

# <codecell>

ChallengeResult(
    "model_tuning",
    search=search,
    best_pipeline=best_pipeline,
    best_scores = best_scores,
    popularity=popularity
).write()

# <markdowncell>

# ## API 
# 
# Time to put a pipeline in production!
# 
# 👉 Go to https://github.com/lewagon/data-certification-api and follow instructions
# 
# **This final part is independent from the above notebook**
