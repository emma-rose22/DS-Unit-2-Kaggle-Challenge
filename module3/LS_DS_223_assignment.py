#!/usr/bin/env python
# coding: utf-8
Lambda School Data Science

*Unit 2, Sprint 2, Module 3*

---
# # Cross-Validation
# 
# 
# ## Assignment
# - [X] [Review requirements for your portfolio project](https://lambdaschool.github.io/ds/unit2), then submit your dataset.
# - [ ] Continue to participate in our Kaggle challenge. 
# - [X] Use scikit-learn for hyperparameter optimization with RandomizedSearchCV.
# - [ ] Submit your predictions to our Kaggle competition. (Go to our Kaggle InClass competition webpage. Use the blue **Submit Predictions** button to upload your CSV file. Or you can use the Kaggle API to submit your predictions.)
# - [ ] Commit your notebook to your fork of the GitHub repo.
# 
# 
# You won't be able to just copy from the lesson notebook to this assignment.
# 
# - Because the lesson was ***regression***, but the assignment is ***classification.***
# - Because the lesson used [TargetEncoder](https://contrib.scikit-learn.org/categorical-encoding/targetencoder.html), which doesn't work as-is for _multi-class_ classification.
# 
# So you will have to adapt the example, which is good real-world practice.
# 
# 1. Use a model for classification, such as [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
# 2. Use hyperparameters that match the classifier, such as `randomforestclassifier__ ...`
# 3. Use a metric for classification, such as [`scoring='accuracy'`](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)
# 4. If you’re doing a multi-class classification problem — such as whether a waterpump is functional, functional needs repair, or nonfunctional — then use a categorical encoding that works for multi-class classification, such as [OrdinalEncoder](https://contrib.scikit-learn.org/categorical-encoding/ordinal.html) (not [TargetEncoder](https://contrib.scikit-learn.org/categorical-encoding/targetencoder.html))
# 
# 
# 
# ## Stretch Goals
# 
# ### Reading
# - Jake VanderPlas, [Python Data Science Handbook, Chapter 5.3](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html), Hyperparameters and Model Validation
# - Jake VanderPlas, [Statistics for Hackers](https://speakerdeck.com/jakevdp/statistics-for-hackers?slide=107)
# - Ron Zacharski, [A Programmer's Guide to Data Mining, Chapter 5](http://guidetodatamining.com/chapter5/), 10-fold cross validation
# - Sebastian Raschka, [A Basic Pipeline and Grid Search Setup](https://github.com/rasbt/python-machine-learning-book/blob/master/code/bonus/svm_iris_pipeline_and_gridsearch.ipynb)
# - Peter Worcester, [A Comparison of Grid Search and Randomized Search Using Scikit Learn](https://blog.usejournal.com/a-comparison-of-grid-search-and-randomized-search-using-scikit-learn-29823179bc85)
# 
# ### Doing
# - Add your own stretch goals!
# - Try other [categorical encodings](https://contrib.scikit-learn.org/categorical-encoding/). See the previous assignment notebook for details.
# - In additon to `RandomizedSearchCV`, scikit-learn has [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). Another library called scikit-optimize has [`BayesSearchCV`](https://scikit-optimize.github.io/notebooks/sklearn-gridsearchcv-replacement.html). Experiment with these alternatives.
# - _[Introduction to Machine Learning with Python](http://shop.oreilly.com/product/0636920030515.do)_ discusses options for "Grid-Searching Which Model To Use" in Chapter 6:
# 
# > You can even go further in combining GridSearchCV and Pipeline: it is also possible to search over the actual steps being performed in the pipeline (say whether to use StandardScaler or MinMaxScaler). This leads to an even bigger search space and should be considered carefully. Trying all possible solutions is usually not a viable machine learning strategy. However, here is an example comparing a RandomForestClassifier and an SVC ...
# 
# The example is shown in [the accompanying notebook](https://github.com/amueller/introduction_to_ml_with_python/blob/master/06-algorithm-chains-and-pipelines.ipynb), code cells 35-37. Could you apply this concept to your own pipelines?
# 

# ### BONUS: Stacking!
# 
# Here's some code you can use to "stack" multiple submissions, which is another form of ensembling:
# 
# ```python
# import pandas as pd
# 
# # Filenames of your submissions you want to ensemble
# files = ['submission-01.csv', 'submission-02.csv', 'submission-03.csv']
# 
# target = 'status_group'
# submissions = (pd.read_csv(file)[[target]] for file in files)
# ensemble = pd.concat(submissions, axis='columns')
# majority_vote = ensemble.mode(axis='columns')[0]
# 
# sample_submission = pd.read_csv('sample_submission.csv')
# submission = sample_submission.copy()
# submission[target] = majority_vote
# submission.to_csv('my-ultimate-ensemble-submission.csv', index=False)
# ```

# In[4]:


get_ipython().run_cell_magic('capture', '', "import sys\n\n# If you're on Colab:\nif 'google.colab' in sys.modules:\n    DATA_PATH = 'https://raw.githubusercontent.com/LambdaSchool/DS-Unit-2-Kaggle-Challenge/master/data/'\n    !pip install category_encoders==2.*\n\n# If you're working locally:\nelse:\n    DATA_PATH = '../data/'")


# In[5]:


import pandas as pd

# Merge train_features.csv & train_labels.csv
train = pd.merge(pd.read_csv(DATA_PATH+'waterpumps/train_features.csv'), 
                 pd.read_csv(DATA_PATH+'waterpumps/train_labels.csv'))

# Read test_features.csv & sample_submission.csv
test = pd.read_csv(DATA_PATH+'waterpumps/test_features.csv')
sample_submission = pd.read_csv(DATA_PATH+'waterpumps/sample_submission.csv')


# In[7]:


import category_encoders as ce
import numpy as np
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[17]:


ce.OrdinalEncoder('status_group')


# In[19]:


train['status_group'].value_counts()


# In[54]:


status = {'functional': 1,'non functional': 2, 'functional needs repair' : 3} 
  
# traversing through dataframe 
# Gender column and writing 
# values where key matches 
train.status_group = [status[item] for item in train.status_group] 


# In[22]:


train['status_group']


# In[ ]:


#baseline model


# In[23]:


target = 'status_group'
features = train.columns.drop([target] + ['id', 'recorded_by'])
X_train = train[features]
y_train = train[target]

pipeline = make_pipeline(
    ce.OrdinalEncoder(), 
    SimpleImputer(strategy='mean'), 
    StandardScaler(), 
    SelectKBest(f_classif, k=20), 
    Ridge(alpha=1.0)
)

k = 3
scores = cross_val_score(pipeline, X_train, y_train, cv=k, 
                         scoring='neg_mean_absolute_error')
print(f'MAE for {k} folds:', -scores)


# In[63]:





# In[24]:


-scores.mean()


# In[ ]:


#first model, no cross validation


# In[32]:


from sklearn.ensemble import RandomForestClassifier

pipeline1 = make_pipeline(
    ce.OrdinalEncoder(), 
    SimpleImputer(strategy='median'), 
    RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
)

k = 5
scores1 = cross_val_score(pipeline, X_train, y_train, cv=k, 
                         scoring='neg_mean_absolute_error')
print(f'MAE for {k} folds:', -scores1)


# In[33]:


-scores1.mean()


# In[ ]:


#second model, with cross validation


# In[44]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

pipeline = make_pipeline(
    ce.TargetEncoder(),
    SimpleImputer(),
    SelectKBest(f_classif),
    RandomForestClassifier(random_state=42)
)

param_distributions = {
    'targetencoder__min_samples_leaf': randint(1, 1000),
    'targetencoder__smoothing': uniform(1, 1000),
    'simpleimputer__strategy': ['mean', 'median'],
    'selectkbest__k': randint(20, 38),
    'randomforestclassifier__n_estimators': randint(50, 500),
    'randomforestclassifier__max_depth': [5, 10, 15, 20, None],
    'randomforestclassifier__max_features': uniform(0, 1)
}

search3 = RandomizedSearchCV(
    pipeline,
    param_distributions= param_distributions,
    n_iter=20,
    cv = 5,
    scoring='neg_mean_absolute_error',
    verbose=10,
    return_train_score=True,
    n_jobs=-1
)

search3.fit(X_train, y_train);


# In[45]:


print('Best hyperparameters', search3.best_params_)
print('Cross-validation MAE', -search3.best_score_)


# In[46]:


pipeline = search3.best_estimator_


# In[50]:


status = {1: 'functional', 2: 'non functional', 3 : 'functional needs repair'} 
  
# traversing through dataframe 
# Gender column and writing 
# values where key matches 
train.status_group = [status[item] for item in train.status_group] 
print(train['status_group'])


# In[51]:


train['status_group']


# In[70]:


train['status_group']


# In[73]:


from sklearn.metrics import mean_absolute_error
target = 'status_group'
X_test = test[features]
y_test = test[target]

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test MAE: ${mae:,.0f}')

