# Box-Office-Data-Analysis-with-Seaborn-and-Python
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

## 1. Import data and libraries
 
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
Type in `train.head()` to have a look at the dataset. Since this is a regression problem, our target column is the *revenue* column where we will be predicting it's values. These are continuous values and in a data science or machine learning problem, if we want to predict a particular column, it is almost indispensible in the EDA phase to look at the distribution of your target variable. This gives an insight on the modelling choices you will make or if you have to make any tranformations to the target variable itself.

## 2. Visuaulizing the Target Distribution

After plotting a histogram of the target column revenue using `train.revenue.hist()`, we can observe how skewed the distribution is (as shown below *(left)*, type in the given code and verify). What we want to do is normalize this by taking the **logarithm of this distribution**.

So what we can do is use seaborn to create a subplot of two plots side by side where we have the original target distribution and the logarithmic distribution to compare. In most regression problems, you want your target column to ideally be normally distributed for the model to perform better.

![download (1)](https://user-images.githubusercontent.com/66896597/87536378-a0106d00-c6b6-11ea-8425-f07817f0a5e1.png)

`plt.subplot(1, 2, 1)` indicates **1** plot, **2** columns and **1** row (*first subplot*). The `kde= False` is taken to ensure no kernel density estimate is overlayed on top of the histogram. For the second subplot, we use the helper function from the numpy library 'log1p' because if we have some zero inputs/dataset, we don't get NaN or infinity as error values. After plotting the log values, we can observe that they are not as skewed, hence making it easier to work with. Create a new feature **train['log_revenue']** containing log values of the revenue column.

## 3. Analyzing the Budget with Film revenue

We can all agree that in most cases, there is a correlation between the budget of a film and the revenue generated. If the film had a higher budget, it would probably generate a higher revenue or maybe vice-versa. In this segment we can find out correlation between the two and the degree of correlation using the given data.


A scatterplot `sns.scatterplot(train['budget'], train['revenue'])` is used to figure out the relation between the revenue and budget but we cannot really figure out the relation. So, we compare the log transformed budget and revenue to check for a relation between the two and we can clearly see the relation between log transformed revenue and budget.

![download (2)](https://user-images.githubusercontent.com/66896597/87536700-22009600-c6b7-11ea-8e65-6a383f159202.png)

## 4. Does having an official homepage affect the revenue?

With `train['homepage'].value_counts()` we calculate the unique number of homepages for each film. Then we use this to create another feature called occurence of a homepage    `train['has_homepage']` with binary values which is used to indicate the presence or absence of a homapage. 

We use **.loc** to find out if a particular film has a homepage. This can be done by comparing its `isnull()== False`. If this condition is satisfied, it means that that film has a homepage and we assign it with 1.

As train['has_homepage'] is a categorical function, we can use catplot from seaborn to create our visualization. 

![download (4)](https://user-images.githubusercontent.com/66896597/87540532-7c045a00-c6bd-11ea-96ff-f9352515e6b7.png)

From the figure, we find out that films without a homepage *(left)* generate a lower revenue compared to the ones with a homepage *(right)*.

## 5. Distribution of Languages in Films

In this section, we will find out the distribution of languages in films. We will do this by locating a column *original_language* from the train dataset and filter in only those languages which are common using the function **value_counts()** from pandas like we did in the previous section and store all this data in a variable *language_data*. now we can plot the revenue vs language distribution using boxplots. Boxplot is a great way to identify outliers if there exist any. As per usual, we will be plotting two subplots. Here we are taking a look at mean revenue per language in a film. We can compare it to the subplot which will be log_revenue per language in a film. 

```python
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
sns.boxplot(x='original_language', y='revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)]);
plt.title('Mean revenue per language');
plt.subplot(1, 2, 2)
sns.boxplot(x='original_language', y='log_revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)]);
plt.title('Mean log revenue per language');
```

![image](https://user-images.githubusercontent.com/66896597/87542043-e8805880-c6bf-11ea-9b3f-5f1a84da1fe1.png)

When we look at the plot on the left, it gives us a biased opinion that the only films that generate a higher revenue are made in english but when we look at the log version on the right, we come across a lot of films generating high revenues despite not being in english.

## 6. Distribution/Frequency of Words used in Titles and Descriptions

In this section, we will create a **wordcloud** which is a **qualitative** way of telling that more frequently occuring words will be larger in size compared to those that occur less frequently. To find the words across our film titles, we can use simple text manipulation like

`text = ' '.join(train['original_title'].values)`

Now use the wordcloud function and setup details such as font size, background color etc. and then use the generate function to create the most frequently occured words. 

`wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)` 

![image](https://user-images.githubusercontent.com/66896597/87550236-77936d80-c6cc-11ea-8124-c5db844e1039.png)


Now do the same for film descriptions. Make sure you fill the null values in the *overview* column with an empty string to not generate error messages when creating a wordcloud for the film description. Therefore we can conclude that larger the word size, more frequent the word occurred.

## 7. Do Film Descriptions Impact Revenue?

After analyzing the frequency of words qualitatively in the previous section, we will find out if certain words that occur frequently in film descriptions impact revenue or not. We will do this using two libraries.
- **sklearn**: We will build a linear regression model. If you were wondering *how to build a linear regression model for these words which are strings*, we will be creating a **Tfidf vector** for each of these words. This basically helps us with the numerical representation of text. We will be creating a numerical representation of frequently occurred words and fit a linear model to predict the revenue of a film.
- **eli5**: It is python package which helps us to debug machine learning classifiers and also explain their predictions. Using this package, we will be able see visually, which words impact the revenue.

Check out some other resources if you don't have any idea about Tfidf vectors!!
```python
vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2),  min_df=5)
```
In this code snippet, **min_df** indicates the threshold frequency (which is set to 5 here) and all the values below this will be ignored. We will be looking at unigrams and bigrams --> `ngram_range=(1,2)` i.e, a word and pairs of words will be considered. 

After instantiating the Tfidf vector, fit it to the overview text. Fill null values with an empty string.

`overview_text = vectorizer.fit_transform(train['overview'].fillna(''))`

Now that we have numerically represented our words, we can build and fit a linear regression model.

```python
linreg = LinearRegression()
linreg.fit(overview_text, train['log_revenue'])
```
After fitting the model, we will use **eli5** which explains the predictions of scikit learn estimators in a visual way. We can do this by using the show_weights method in eli5 to look at the weights of our linear regression model. This will point to words which have high and low impact on revenue. We can filter the number of words as there could hundreds of thousands of them. Use a feature filter so that only those words that fit a pattern are considered. Here the pattern is a lambda function which is `x != '<BIAS>'`. This means that we want all tokens returned that donot have a bias. 

From this analysis, we can find out what features have an impact on the revenue generated by a film. Seaborn is an excellent library to visually interpret the relations between various features and the revenue generated. You can try plotting it for diferent features to learn more about the factors that impact a film's revenue. 






           
            
            
           
  

                      
                      
                      
                    











