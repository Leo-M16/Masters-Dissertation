# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:45:21 2020

@author: leoma
"""

import pandas as pd
import ta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

"""Loading the file containing closes prices, creating the technical indicators"""
data = pd.read_csv('DJI4.csv', header=0, usecols=['Date', 'Close'], parse_dates=True, index_col='Date')

sma = data['Close'].rolling(window=20).mean()
rstd = data['Close'].rolling(window=20).std()
upper_band = sma + (2 * rstd)
lower_band=sma-(2*rstd)

"""Creat a copy of the main data, add the technical indicators to the new copy"""
mata=data
mata['sma']=sma
mata['rstd']=rstd
mata['upper_band']=upper_band
mata['lower_band']=lower_band

"""Convert the copy of the data to numpy object"""
mata=mata.to_numpy()

"""Split the data 70% for training 30% for testing"""
train_length = int(len(mata) * 0.7)
test_length = len(mata) - train_length
train_data = mata[0:train_length,:]
test_data = mata[train_length:len(mata),:]

"""Create the dataset of training and testing"""
def create_dataset(dataset, timestep=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - timestep -1):
        data_x.append(dataset[i:(i+timestep),])
        data_y.append(dataset[i+timestep,])
    return np.array(data_x), np.array(data_y)

timestep = 1
train_x, train_y = create_dataset(train_data, timestep)
test_x, test_y = create_dataset(test_data, timestep)

"""Delete the technical indicators so only prices are remained in y (data to be predicted)"""
test_y=np.delete(test_y, [1,2,3,4],axis=1)
train_y=np.delete(train_y,[1,2,3,4],axis=1)

"""reshape the data to a format acceptable by MinMaxScaler() Function"""
train_x=np.reshape(train_x, (1582, 5))
test_x=np.reshape(test_x,(677,5))

"""Normalize the date prices"""
scaler1=MinMaxScaler()
train_x = scaler1.fit_transform(train_x).
scaler2=MinMaxScaler()
train_y=scaler2.fit_transform(train_y)
scaler3=MinMaxScaler()
test_x=scaler3.fit_transform(test_x)
scaler4=MinMaxScaler()
test_y=scaler4.fit_transform(test_y)

"""Add sentiment scores to X (data used by model to predict next day close price)"""
sent_score = pd.read_csv('Sentiment_scores.csv', header=0,  index_col='Date')
sent_score=sent_score.to_numpy()
sent_score_train,sent_score_test=create_dataset(sent_score,timestep)

train_x['sent_score']=sent_score_train
test_x['sent_score']=sent_score_test

"""Reshape the X data and create the model"""
train_x=np.reshape(train_x, (1582,1, 6))
test_x=np.reshape(test_x,(677,1,6))

model = Sequential()
model.add(LSTM(256, return_sequences=True,input_shape=(1,6)))
model.add(LSTM(256, return_sequences=True,input_shape=(1,6)))
model.add(LSTM(256, input_shape=(1,6)))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""Train the model, get the predictions""""
model.fit(train_x, train_y, epochs=6, batch_size=1, verbose=1)
score = model.evaluate(train_x, train_y, verbose=0)
print('Keras model loss = ', score[0])
print('Keras model accuracy = ', score[1])

train_predictions = model.predict(train_x)
test_predictions = model.predict(test_x)

"""De normalize the prices """
train_predictions = scaler2.inverse_transform(train_predictions)
test_predictions = scaler4.inverse_transform(test_predictions)

"""Plot the original prices(blue) vs prices predicting during train(red) and test(green)"""
train_predict_plot = np.empty_like(data)
train_predict_plot[:,:] = np.nan
train_predict_plot[1:len(train_predictions)+1, :] = train_predictions

test_predict_plot=np.empty_like(data)
test_predict_plot[:,:] = np.nan
test_predict_plot[len(train_predictions)+2+1:len(data)-1, :] = test_predictions

mata1=np.delete(mata, [1,2,3,4,5],axis=1)

plt.plot(mata1,color="blue")
plt.plot(train_predict_plot,color="red")
plt.plot(test_predict_plot,color="green")
plt.show()

"""Calculate the mean squared error"""
test_y1=np.delete(test_y,[1,2,3,4,5],axis=1)

test_y1=np.reshape(test_y1,(677,6))
test_y1=scaler3.inverse_transform(test_y1)
mean_squared_error( test_predictions,test_y1)



