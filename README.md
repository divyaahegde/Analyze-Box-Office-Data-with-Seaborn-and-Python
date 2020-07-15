# Analyze-Box-Office-Data-with-Seaborn-and-Python
Welcome to this project based on Analyzing Worldwide Box Office Revenue with Seaborn and Python.  In this project we will be working with the The Movie Database (TMDB) Box Office Prediction data set. Can we build models to accurately predict movie revenue?  Could the results from these models be used to further increase revenue?  We try to answer these questions by way of exploratory data analysis (EDA) in this project. We will be using a statistical data visualization library like Seaborn. 
 
 
## Introduction
In this project we will be using **Seaborn library** for data visualization. Seaborn is based on Matplotlib which is used for the creation of *statistical graphics*. 

### It has a number of functionalities like:
- api which is based on datasets allowing comparisons between multiple variables.
- supports multiplot grids that ease the building of complex visualization. 
- has univariate and bivariate visualization to compare between subsets of data.
- estimates and plots linear regression automatically.
 
Before we dive into the project, it is necessary to know the difference between seaborn and matplotlib and why we prefer the former over the latter.

In matplotlib, it is difficult to figure out what settings should be used to make plots more attractive. Seaborn has a number of themes and high-level interfaces which make it easier to personalize our plots. Matplotlib doesn't serve well when it comes to dataframes whereas seaborn works on dataframes.

## Import data and libraries
 
As a case study, in this project we will be working with **worldwide box office revenue data** collected by **The Movie Database**. The Dataset we will be using is gathered from the api of this website called [themoviedatabase](https://www.themoviedb.org) or themoviedb. Tmdb is a user editable database for movie and tv shows. Movies in this database are labelled with their unique ID and have features such as cast, crew, keywords, release date, language, budget, posters, etc. This database was initially set up for a regression type problem to predict revenue. Given all the features, a person had to predict revenue for that movie.

### Importing libraries

``` python
import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')
plt.style.use("tableau-colorblind10")
import datetime
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, KFold
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
nltk.download('stopwords')
stop = set(stopwords.words('english'))
import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import json
import ast
from urllib.request import urlopen
from PIL import Image
```
### Importing Data

```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```
Type in `train.head()` to have a look at the dataset. Since this is a regression problem, our target column is the *revenue* column where we will be predicting it's values. These are continuous values and in sort of data science or machine learning problem, we want to predict a particular column, it is almost indispensible in the EDA phase to look at the distribution of your target variable. This gives an insight on the modelling choices you will make or if you have to make any tranformations to the target variable itself.

```python

```

After plotting a histogram of the target column revenue using `train.revenue.hist()`, we can observe how skewed the distribution is (not shown here, type in the given code and verify). What we want to do is normalize this by taking the **logarithm of this distribution**.

So what we can do is use seaborn to create a subplot of two plots side by side where we have the original target distribution and the logarithmic distribution to compare. In most regression problems, you want your target column to ideally be normally distributed for the model to perform better.

`plt.subplot(1, 2, 1)` indicates **1** plot, **2** columns and **1** row (*first subplot*). The `kde= False` is taken to ensure no kernel density estimate is overlayed on top of the histogram. For the second subplot, we use the helper function from the numpy library 'log1p' because if we have some zero inputs/dataset, we don't get NaN or infinity as error values. After plotting the log values, we can observe that they are not as skewed, hence making it easier to work with. Create a new feature **train['log_revenue']** containing log values of the revenue column.

### Analyzing the Budget with Film revenue

We can all agree that in most cases, there is a correlation between the budget of a film and the revenue generated. If the film had a higher budget, it would probably generate a higher revenue or maybe vice-versa. In this segment we can find out correlation between the two and the degree of correlation using the given data.

![download (1)](https://user-images.githubusercontent.com/66896597/87536378-a0106d00-c6b6-11ea-8425-f07817f0a5e1.png)

A scatterplot `sns.scatterplot(train['budget'], train['revenue'])` is used to figure out the relation between the revenue and budget. We cannot really figure out the relation. 



Now we compare the log transformed budget and revenue to check for a relation between the two.






