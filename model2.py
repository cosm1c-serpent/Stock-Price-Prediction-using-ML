import math
import os
from re import X
import pandas_datareader as web
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('bmh')

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

#get data for stock
#get your own TIINGO API key for data fetching


#df = web.get_data_tiingo('AAPL', api_key=key)
df = web.DataReader('AAPL', data_source='yahoo' , start = '2012-01-01')
df = df.to_csv('AAPL2.csv')

df=pd.read_csv('AAPL2.csv')

#print(df.tail(4))
print('shape of data : ',df.shape)

#visulize closing price data

plt.figure(figsize=(10,8))
plt.title('closing price ')
plt.plot(df['Close'])
plt.xlabel('Days', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
#plt.show()

#new dataframe for close price
data = df.filter(['Close'])

#convert to numpy array
dataset = data.values

training_data_len = math.ceil(len(dataset) * .8)

print('training data length : ',training_data_len)

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

print('scaled data : ',scaled_data)

#create training data set
train_data = scaled_data[0:training_data_len , :]
#split the data 
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<= 60:
        print(x_train)
        print(y_train)
        print()

#convert the x and y train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
print('print xtrain.shape : ', x_train.shape)


#Building LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model
model.fit(x_train, y_train, batch_size=1, epochs=2)


#create the testing data set
#create a new array containing scaled values from index

test_data = scaled_data[training_data_len - 60: , :]

#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range (60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


#convert the data to a numpy array
x_test = np.array(x_test)

#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

#get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#getting RMSE

rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print('print rmse value : ',rmse)


#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#visualize the data
plt.figure(figsize=(10,8))
plt.title('Model')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Value', 'Predictios'], loc='lower right')
plt.show()
print('\n\n\n')
print("Predicted values : ")
print(valid)


#future predict
apple = web.DataReader('AAPL', data_source='yahoo' , start = '2012-01-01')

new_df = apple.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print('predicted price furure: ',pred_price)

