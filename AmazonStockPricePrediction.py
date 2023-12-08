
file_path = r'C:\Users\ShashankPC\Downloads\AMZN.csv'


# ------------------------------------------------------------------------------------------------

# import pandas as pd

# import matplotlib.pyplot as plt

# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM

# from statsmodels.tsa.seasonal import seasonal_decompose

# df = pd.read_csv(file_path, index_col='Date',parse_dates=True)

# diff_df1 = pd.DataFrame(df.Close.diff(periods = 1))
# diff_df2 = pd.DataFrame(df.Close.diff(periods = 2))

# df = pd.merge(df, diff_df1, right_index = True,left_index = True)
# df = pd.merge(df, diff_df2, right_index = True,left_index = True)

# df = df.rename(columns={'Close_x':'close', 'Close_y':'diff_1', 'Close':'diff_2'})

# df = df.dropna()

# plt.plot(df.close)
# plt.show(block=False)

# results = seasonal_decompose(df['Close'], model='additive', period=1)
# results.plot()
# plt.show(block=False)

# results = seasonal_decompose(df.diff_1, model='additive', period=1)
# results.plot()
# plt.show(block=False)

# results = seasonal_decompose(df.diff_2, model='additive', period=1)
# results.plot()
# plt.show(block=False)

# ------------------------------------------------------------------------------------------------

# df = pd.DataFrame(df.diff_1)

# train = df.iloc[:6200]
# test = df.iloc[6200:]


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(train)
# scaled_train = scaler.transform(train)
# scaled_test = scaler.transform(test)

# from keras.preprocessing.sequence import TimeseriesGenerator

# n_input = 3
# n_features = 1
# generator = TimeseriesGenerator(scaled_train,
#                                 scaled_train,
#                                 length=n_input,
#                                 batch_size=1)
# X, y = generator[0]
# print(f'Given the Array: \n{X.flatten()}')
# print(f'Predict this y: \n {y}')
# # We do the same thing, but now instead for 12 months
# n_input = 12
# generator = TimeseriesGenerator(scaled_train,
#                                 scaled_train,
#                                 length=n_input,
#                                 batch_size=1)

# # define model
# model = Sequential()
# model.add(LSTM(100, activation='relu',
#             input_shape=(n_input, n_features)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.summary()
# model.fit(generator, epochs=2)

# pred_train = model.predict(train)
# pred_test = model.predict(test)

# ------------------------------------------------------------------------------------------------

# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
tf.random.set_seed(7)

# load the dataset
dataframe = pd.read_csv(file_path, usecols=['Close'], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.95)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
test_r2_Score = r2_score(testY[0], testPredict[:,0])
print('Test Score: %.2f r2_score' % (test_r2_Score))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show(block=False)

# pd.DataFrame({'test':testY[0], 'pred':testPredict[:,0], 'predtestdiff':testY[0]-testPredict[:,0], 'diffperc':((testY[0]-testPredict[:,0])/testY[0])*100})