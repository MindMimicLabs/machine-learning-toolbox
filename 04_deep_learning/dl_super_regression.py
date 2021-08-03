import pandas as pd
import numpy as np
import re

stock = pd.read_csv("data/google_estimates_v2.csv")
stock['p_date'] = pd.to_datetime(stock['p_date'])
stock['LAST_PRICE'] = stock['ACT_CLOSE_PRICE'].shift(periods=1, freq=None, axis=0)
stock['LAST_VOLUME'] = stock['ACT_VOLUME'].shift(periods=1, freq=None, axis=0)
stock['5MED_VOLUME'] = stock['LAST_VOLUME'].rolling(5).median().reset_index(0,drop=True)
stocktrain = stock.loc[0:1000]
stocktest = stock.loc[1001:1018]
stock.to_csv("stockV2.csv")
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras import callbacks
from keras import optimizers
from keras import regularizers
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler
stock = stock.fillna(0)
stock = stock.replace(np.inf, 0)
stock = stock.round(decimals=2)
stocktrain = stocktrain.fillna(0)
stocktrain = stocktrain.replace(np.inf, 0)
stocktrain = stocktrain.round(decimals=2)
# Train, Test Split highly correlated features from stock.
X_train, X_test, y_train, y_test = \
train_test_split(stocktrain[['EPS_Q_EST', 'REV_Q_EST', 'EPS_A_EST',
       'REV_A_EST', 'EPS_LT', 'PRICE_LT', 'REV_Q_ACT', 'NI_Q_ACT',
       'REV_A_ACT', 'NI_A_ACT','LAST_PRICE','5MED_VOLUME']], stocktrain['ACT_CLOSE_PRICE']            
            )    
ss = StandardScaler()
# Fit and transform the training data.

X_train = ss.fit_transform(X_train) # only fit the train, not the test
X_test = ss.transform(X_test)
# Setup the Keras model.
opt = optimizers.Nadam(lr=0.001)
model = Sequential()
model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(9, activation='relu', kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(6, activation='relu', kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(Dense(3, activation='relu', kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(Dense(9, activation='relu', kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)))
model.add(Dense(6, activation='relu', kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(Dense(3, activation='relu', kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience= 50)
callback=[early_stop]
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2000, callbacks= callback, verbose=1)
y_pred = model.predict(X_test)
r2_score(y_test, y_pred)
X_new = stocktest[['EPS_Q_EST', 'REV_Q_EST', 'EPS_A_EST',
       'REV_A_EST', 'EPS_LT', 'PRICE_LT', 'REV_Q_ACT', 'NI_Q_ACT',
       'REV_A_ACT', 'NI_A_ACT','LAST_PRICE','5MED_VOLUME']]
X_new = ss.transform(X_new)
pred = model.predict(X_new)
Stock_Pred = pd.DataFrame(list(zip(stocktest['p_date'],pred,stocktest['ACT_CLOSE_PRICE'])))
Stock_Pred
