#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(rc={'figure.figsize':(10,10)}, font_scale=1.2)


# In[6]:


df= pd.read_csv('Ecommerce Purchases.csv')
df


# In[9]:


df.isnull().sum()


# In[11]:


df.info()


# In[19]:


df


# In[20]:


x= df.drop('Yearly Amount Spent',axis=1)
x


# In[21]:


y= df['Yearly Amount Spent']
y


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[25]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# In[26]:


scaler.fit(x_train)

x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)


# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score


# In[28]:


models= {
    'LinearRegression':LinearRegression(),
    'KNeighborsRegressor':KNeighborsRegressor(),
    'SVR':SVR(),
    'DecisionTreeRegressor':DecisionTreeRegressor(),
    'RandomForestRegressor':RandomForestRegressor(),
    'XGBRegressor':XGBRegressor(),


    
    
}


# In[29]:


for name,model in models.items():
    print(f'Using model {name}')
    model.fit(x_train,y_train)
    y_pred= model.predict(x_test)
    print(f'RMSE:{np.sqrt(mean_squared_error(y_test,y_pred))}')
    print(f'R2 score:{r2_score(y_test,y_pred)}')

    print('------------------------------')


# In[33]:


model=LinearRegression()
model.fit(x_train,y_train)
y_pred= model.predict(x_test)


# In[34]:


y_pred


# In[35]:


y_test


# In[ ]:




