#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator


# In[2]:


from sklearn import metrics 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from yellowbrick.regressor import PredictionError
from yellowbrick.regressor import ResidualsPlot


# In[3]:


import plotly.graph_objs as go
from powerbiclient import Report, models

import openpyxl
from pathlib import Path 
from openpyxl import load_workbook 


# In[4]:


import matplotlib as plt


# In[30]:


#filename = "/home/admin/Models/data/LSTM_UNIVARIATE_DAILY/input/daily tickers.csv"
#filename = "Iodex_62_daily_data_OctEnd2021.csv"
filename = "./InputFile.csv"
df = pd.read_excel(filename,parse_dates=['Date'], sheet_name ='Sheet2')
print(df.info())


# In[31]:


df1 = df.dropna()
df1.info()


# In[32]:


df1['Date'] = pd.to_datetime(df1['Date'])
df1.set_axis(df1['Date'], inplace=True)


# In[33]:


df2 = df1.sort_index(ascending=True)
df2.tail()


# In[34]:


df2.tail()


# In[35]:


from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df2['IODEX62'], model='additive', freq=52)
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)
result.plot()


# In[38]:


close_data = df2['IODEX62'].values
#close_data = close_data[:-36]
close_data = close_data.reshape((-1,1))


# In[39]:


close_data


# In[40]:


split_percent = 0.80
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df2['Date'][:split]
date_test = df2['Date'][split:]

print(len(close_train))
print(len(close_test))


# In[41]:


date_train.tail()


# In[42]:


date_test.head()


# In[43]:


look_back = 1

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)


# In[44]:


from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(
    LSTM(10,
        activation='relu',
        input_shape=(look_back,1))
)
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

num_epochs = 70
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)


# In[45]:


# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("lstm_uni_1daylag_model.h5")


# In[46]:



model_filepath = 'lstm_uni_1daylag_model.h5'
tf.keras.models.save_model(
    model, model_filepath, overwrite=True, include_optimizer=True, save_format=None,
    signatures=None, options=None, save_traces=True
)


# In[47]:


model1 = tf.keras.models.load_model("lstm_uni_1daylag_model.h5")


# In[48]:


prediction = model1.predict_generator(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))

trace1 = go.Scatter(
    x = date_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = date_test,
    y = close_test,
    mode='lines',
    name = 'Ground Truth'
)
layout = go.Layout(
    title = "IODEX62",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()


# In[49]:


prediction


# In[59]:


len(close_test)


# In[66]:


test = close_test[:-1]
test


# test = test[0:50]
# test

# In[67]:


len(test)


# In[68]:


testing = pd.DataFrame(test,columns=["Testing_data"])
len(testing)


# In[69]:


test_pred = pd.DataFrame(prediction,columns=["Pred_testing"])
len(test_pred)


# In[70]:


#test_date1


# # Validate model accuracy with testing data and prediction value.

# In[71]:


MSE = metrics.mean_squared_error(test,prediction)
R_squared = r2_score(test, prediction)
MAE = metrics.mean_absolute_error(test,prediction)
RMSE = np.sqrt(metrics.mean_squared_error(test, prediction))
MAPE = metrics.mean_absolute_percentage_error(test,prediction)
time = datetime.datetime.now()
print("MSE:",MSE)
print("R^2:",R_squared)
print("MAE:",MAE)
print("RMSE:",RMSE)
print("MAPE:",MAPE)
print("Time:",time)


# In[73]:


close_data = close_data.reshape((-1))

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
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1,freq='W-MON').tolist()
    return prediction_dates

num_prediction = 30
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)


# In[76]:


df444=close_data[-look_back:]
df444