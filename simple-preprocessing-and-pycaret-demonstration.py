#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and dataset

# In[2]:


# This is a Python 3 environment 
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# Importing libraries for
import numpy as np # linear algebra
import pandas as pd # data processing
import os # files
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
from sklearn.ensemble import RandomForestClassifier # machine learning


# ## PyCaret
# 
# PyCaret is an open source, low-code machine learning library in Python that allows you to go from preparing your data to deploying your model within minutes in your choice of notebook environment.
# For more, go to...
# https://pycaret.org
# 
# 

# In[2]:


# Installing and importing PyCaret, which will be demonstrated in this notebook
get_ipython().system('pip install pycaret')
from pycaret.classification import *


# In[3]:


# Importing the files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Data Dictionary
# * survival	Survival	0 = No, 1 = Yes
# * pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# * sex	Sex	
# * Age	Age in years	
# * sibsp	# of siblings / spouses aboard the Titanic	
# * parch	# of parents / children aboard the Titanic	
# * ticket	Ticket number	
# * fare	Passenger fare	
# * cabin	Cabin number	
# * embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# 
# ### Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# In[4]:


# Loading the training dataset
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head(10)


# In[5]:


# Loading the testing dataset
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head(10)


# In[6]:


# Training data information
train_data.info()


# In[7]:


# Null values count in each column
print(train_data.isnull().sum())


# In[8]:


# Description of dataset
train_data.describe()


# In[9]:


# Description of dataset including non-numeric values
train_data.describe(include="all")


# # Data Exploration

# ## Survival

# In[10]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train_data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train_data,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# ## Women who survived

# In[11]:


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("Women survival: ", round(rate_women*100, 2), "%")


# ## Men who survived

# In[12]:


men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("Men survivial: ", round(rate_men*100, 2), "%")


# In[13]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=train_data,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()


# ## Correlation

# In[14]:


sns.heatmap(train_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# # Data Preprocessing

# In[15]:


# Sorting the ages into logical categories
train_data["Age"] = train_data["Age"].fillna(-0.5)
test_data["Age"] = test_data["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train_data['AgeGroup'] = pd.cut(train_data["Age"], bins, labels = labels)
test_data['AgeGroup'] = pd.cut(test_data["Age"], bins, labels = labels)

# Bar plot
sns.barplot(x="AgeGroup", y="Survived", data=train_data)
plt.show()


# In[16]:


# CabinBool
train_data["CabinBool"] = (train_data["Cabin"].notnull().astype('int'))
test_data["CabinBool"] = (test_data["Cabin"].notnull().astype('int'))

print("Percentage of CabinBool = 1 who survived:", train_data["Survived"][train_data["CabinBool"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of CabinBool = 0 who survived:", train_data["Survived"][train_data["CabinBool"] == 0].value_counts(normalize = True)[1]*100)

# CabinBool vs. Survival
sns.barplot(x="CabinBool", y="Survived", data=train_data)
plt.show()


# In[17]:


# For filling in the missing values in the Embarked feature
print("Number of people embarking in Southampton (S):")
southampton = train_data[train_data["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train_data[train_data["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train_data[train_data["Embarked"] == "Q"].shape[0]
print(queenstown)


# In[18]:


# Replacing the missing values in the Embarked feature with S(with mode)
train_data = train_data.fillna({"Embarked": "S"})


# In[19]:


# Create a combined group of both datasets
combine = [train_data, test_data]

# Extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])


# In[20]:


# Replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[21]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_data.head()


# In[22]:


# Fill missing age with mode age group for each title
mr_age = train_data[train_data["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train_data[train_data["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train_data[train_data["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train_data[train_data["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train_data[train_data["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train_data[train_data["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

for x in range(len(train_data["AgeGroup"])):
    if train_data["AgeGroup"][x] == "Unknown":
        train_data["AgeGroup"][x] = age_title_mapping[train_data["Title"][x]]
        
for x in range(len(test_data["AgeGroup"])):
    if test_data["AgeGroup"][x] == "Unknown":
        test_data["AgeGroup"][x] = age_title_mapping[test_data["Title"][x]]


# In[23]:


# Map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train_data['AgeGroup'] = train_data['AgeGroup'].map(age_mapping)
test_data['AgeGroup'] = test_data['AgeGroup'].map(age_mapping)

train_data.head()


# In[24]:


# Remaining null values
print(train_data.isnull().sum())


# In[25]:


# Dropping the Cabin feature since not a lot more useful information can be extracted from it
train_data = train_data.drop(['Cabin'], axis = 1)
test_data = test_data.drop(['Cabin'], axis = 1)


# In[1]:


# Convering Embarked to numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)
train_data.head()


# In[27]:


print(train_data.info())
print('-'*25)
print("Null Values:")
print(train_data.isnull().sum())


# # Modeling and Training

# ## Basic RandomForest Classifier

# In[28]:


y = train_data["Survived"]

# Features to be taken into account for training
features = ["Pclass", "Sex", "SibSp", "Parch", "AgeGroup"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# Training
model.fit(X, y)
# Predicting using the test dataset
predictions = model.predict(X_test)


# In[29]:


# Saving the submission with PassengerId, and Survival prediction
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('random_forest_submission.csv', index=False)
print("Submission file was successfully saved!")


# # Now using PyCaret to visualize and train with various models

# ### Data Setup

# In[30]:


clf1 = setup(train_data, target = 'Survived', ignore_features = ['Ticket', 'Name', 'PassengerId'], silent = True, session_id = 786) 


# In[31]:


# Different models
models()


# ### Comparing different models

# In[32]:


compare_models()


# ### Using Logistic Regression

# In[33]:


lr = create_model('lr')


# ### Tuning model

# In[34]:


tuned_lr = tune_model(lr, optimize = 'AUC', n_iter = 100)


# 

# ### Area Under the Curve

# In[35]:


plot_model(tuned_lr)


# ### Confusion Matrix

# In[36]:


plot_model(tuned_lr, plot ="confusion_matrix")


# ### Threshold

# In[37]:


plot_model(tuned_lr, plot ="threshold")


# ### Preicision - Recall Curve

# In[38]:


plot_model(tuned_lr, plot ="pr")


# ### Class Prediction Error

# In[39]:


plot_model(tuned_lr, plot ="error")


# ### Recursive Feature Selection

# In[40]:


plot_model(tuned_lr, plot ="rfe")


# ### Learning Curve

# In[41]:


plot_model(tuned_lr, plot ="learning")


# ### Validation Curve

# In[42]:


plot_model(tuned_lr, plot ="vc")


# ### Feature Importance

# In[43]:


plot_model(tuned_lr, plot ="feature")


# ### Decision Boundary

# In[44]:


plot_model(tuned_lr, plot ="boundary")


# ### Making predicton on test data

# In[45]:


predictions = predict_model(tuned_lr, data = test_data)
predictions.head()


# ### Saving the submission

# In[46]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions.Label})
output.to_csv('lr_submission.csv', index=False)
print("Submission file was successfully saved!")
output.head()


# ## Now using CatBoost Classifier

# In[47]:


catboost = create_model('catboost')


# In[48]:


tuned_catboost = tune_model(catboost, optimize = 'AUC', n_iter = 100)


# ### Interpretations are implemented based on the SHAP (SHapley Additive exPlanations) 

# In[49]:


interpret_model(tuned_catboost)


# ### Correlation Plot

# In[50]:


interpret_model(tuned_catboost, plot="correlation")


# ### Reason Plot at Observation Level
# 

# In[51]:


interpret_model(tuned_catboost, plot = 'reason', observation = 10)


# ### Making prediction on test data

# In[52]:


predictions = predict_model(tuned_catboost, data = test_data)


# In[53]:


predictions.head()


# In[54]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions.Label})
output.to_csv('catboost_submission.csv', index=False)
print("Submission file was successfully saved!")

