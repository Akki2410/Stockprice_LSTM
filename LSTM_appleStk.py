# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 23:47:54 2021

@author: aksha
"""
#stock market prediction using LSTM


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('AAPL.csv')

df.head()
df.shape

df2 = df[['close']]
print(df2)

plt.plot(df2)


#LSTM is sensitive to scale of data so we apply MinMax Scaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
#df2.iloc[:,:] = scaler.fit_transform(df2.iloc[:,:])
df2 = scaler.fit_transform(np.array(df2).reshape(-1,1))



#splitting dataset into train and test split
training_size = int(len(df2)*0.65)
test_size = len(df2)-training_size
train_data,test_data = df2[0:training_size,:], df2[training_size:len(df2),:]
len(test_data)
len(train_data)
#convert array values to 
def create_dataset(dataset,time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)
    
    
time_step = 100
X_train, y_train = create_dataset(train_data,time_step = time_step)
X_test, y_test = create_dataset(test_data,time_step)
print(X_train.shape)
print(X_test.shape)
y_train.shape

#----------- 19:00
#Create stack LSTM model
#before implementing LSTM always reshape X in 3 dimension
#reshape the input [samples ,time steps, features] 


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

X_train



### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


model.summary()
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)
#-----

import tensorflow as tf


tf.__version__



### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)



##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


### Test Data RMSE
math.sqrt(mean_squared_error(y_test,test_predict))


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df2)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df2)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df2)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df2))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

len(test_data)-100

x_input=test_data[341:].reshape(1,-1)
x_input.shape


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)



day_new=np.arange(1,101)
day_pred=np.arange(101,131)

len(df2)-100
plt.plot(day_new,scaler.inverse_transform(df2[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


df3=df2.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


df3=scaler.inverse_transform(df3).tolist()


plt.plot(df3)
