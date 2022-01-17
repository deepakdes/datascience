#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import datetime
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator


# In[3]:


from sklearn import metrics 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from yellowbrick.regressor import PredictionError
from yellowbrick.regressor import ResidualsPlot


# In[4]:


import plotly.graph_objs as go
from powerbiclient import Report, models

import openpyxl
from pathlib import Path 
from openpyxl import load_workbook 


# In[5]:


import matplotlib as plt


# In[6]:


model = tf.keras.models.load_model("lstm_uni_1daylag_model.h5")


# In[7]:


#filename = "/home/admin/Models/data/LSTM_UNIVARIATE_DAILY/input/daily tickers.csv"
filename = "daily tickers _OctEnd2021.csv"
df = pd.read_excel(filename,parse_dates=['Date'], sheet_name ='Sheet2')
print(df.info())


# In[8]:


df1 = df.dropna()
df1.info()


# In[9]:


df1['Date'] = pd.to_datetime(df1['Date'])
df1.set_axis(df1['Date'], inplace=True)


# In[10]:


df2 = df1.sort_index(ascending=True)
df2.tail()


# In[11]:


df2.tail()


# In[12]:


close_data = df2['IODEX62'].values
#close_data = close_data[:-36]
close_data = close_data.reshape((-1,1))


# In[13]:


close_data = close_data.reshape((-1))
look_back = 1

def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df2['Date'].values[-1]
    prediction_dates = pd.bdate_range(last_date, periods=num_prediction+1,freq='B').tolist()
    return prediction_dates

num_prediction = 60
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)


# In[14]:


len(forecast)


# In[15]:


len(forecast_dates)


# In[16]:


forecast_dates


# In[17]:


#convert timestamp to pd dataframe
date_time_index = pd.to_datetime(forecast_dates)
date_time_index


# In[18]:


import datetime as dt

# Create dataframe
dt2 = pd.DataFrame(date_time_index, columns=['Date'])
dt2.head()


# In[19]:


#pred = pd.DataFrame(forecast,columns=['predicted value'])


# In[20]:


forecast = pd.DataFrame(forecast,columns=['Prediction'])
forecast


# In[21]:


forecast_results = pd.concat([dt2,forecast],axis=1)
forecast_results.head(30)


# # Write results to csv file

# In[25]:


forecast_results.to_csv("Predictions.csv", index = False)


# In[ ]:




