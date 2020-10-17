# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:30:59 2020

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset and separating the columns required for training
df_train=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=df_train.iloc[:,1:2].values

#Data preprocessing
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
training_set_scaled=mm.fit_transform(training_set)

#Creating a datastructure with 60 timesteps and 1 output[any steps can be taken]
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping the X train by adding a new dimentionality as x train is already 2D
X_train= np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

#Building RNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#Initializing RNN
regressor= Sequential()

#Adding 1st LSTM layers with dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape= (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Adding 2nd LSTM layer with regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding 3rd LSTM layer with regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding 4th LSTM layer with regularization
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))

#Adding output layer
regressor.add(Dense(units=1))

#Compiling RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Training data set
regressor.fit(X_train,y_train, batch_size=32, epochs=100)


#Real stock price of 2017
df_test=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=df_test.iloc[:,1:2].values

#Getting predicted stock price of 2017
dataset_total=pd.concat((df_train['Open'], df_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(df_test)-60: ].values
inputs=inputs.reshape(-1,1)
inputs=mm.transform(inputs)
X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test= np.array(X_test)
X_test= np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price=regressor.predict(X_test)
predicted_stock_price = mm.inverse_transform(predicted_stock_price)

#Viz of results
plt.plot(real_stock_price, color='red', label='Real google price')
plt.plot(predicted_stock_price, color='blue', label='Predicted stock price')
plt.title('Stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(rmse)
