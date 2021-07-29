#install the dependencies
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')

#import the apple stock data from Tiingo API and store
import pandas_datareader as pdr
key="5ffc41dd1c2e71d9831f92e9aa61933244cc26e9"

df = pdr.get_data_tiingo('AAPL', api_key=key)
df.to_csv('AAPL.csv')

#read the data
df=pd.read_csv('AAPL.csv')
#df.head(6)
#print(df.head(6))

#visualize the close price data
plt.figure(figsize=(10,5))
plt.title('AAPLE')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(df['close'])
plt.show()

#only get the close price
df=df[['close']]
print(df.head(6))

#predicting 'x' days into the future
future_days= 25
#create new com=lum  shifted 'x' units/days 
df['Prediction']= df[['close']].shift(-future_days)
print(df.tail(6))

#create data set X and convert into numpy array and remove the last 'x' rows/days
X=np.array(df.drop(['Prediction'],1))[:-future_days]
print(X) 

#Create data set (Y) and convert into numpy array and get all of the values except the last 'x' rows/days
y=np.array(df['Prediction'])[:-future_days]
print(y)

#Split data into train and test (75/25)%
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size =0.25)

#creating model
#Decision Tree regressor model
tree= DecisionTreeRegressor().fit(x_train,y_train)

#create Linear Regression Model
lr= LinearRegression().fit(x_train,y_train)

#get the last 'x' rows of the future data set
x_future = df.drop(['Prediction'],1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future

#show model Decision tree prediction
tree_prediction = tree.predict(x_future)
print(tree_prediction)
print()
#show model linear regression prediction
lr_prediction = lr.predict(x_future)
print(lr_prediction)

#visualize the data predicted( DECISION TREE)
predictions = tree_prediction

valid = df[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(10,5))
plt.title('DT Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(df['close'])
plt.plot(valid[['close','Predictions']])
plt.legend(['Original','Value','Prediction'])
plt.show()


#visualize the data predicted( Linear Regression)
predictions = lr_prediction

valid = df[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(10,5))
plt.title(' LR Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(df['close'])
plt.plot(valid[['close','Predictions']])
plt.legend(['Original','Value','Prediction'])
plt.show()

