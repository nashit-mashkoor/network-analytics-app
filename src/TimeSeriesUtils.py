import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go
import os
import warnings

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
import prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from numpy import sqrt
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import keras.backend as k
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Bidirectional, Dropout
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def plot_train_test_pred(train, test, predicted):
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle("Train, Test, and Forecast plots")
    past, = plt.plot(train.index, train, 'r.-', label="Train")
    future, = plt.plot(test.index[-70:], test[-70:], color ='blue', label="Test")
    #predicted_future, = plt.plot(test.index[-70:], predicted[-70:], 'g.-', label="Predicted")
    predicted_future, = plt.plot(test.index[370:], predicted[370:], 'g.-', label="Predicted")
    #forecast, = plt.plot(test.index[85:], predicted[85:0], 'b.-', label="Forecast")
    plt.legend()
    plt.show()

#Plotting Functions
def plot_train_test_pred(train, test, predicted):
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle("Train, Test, and Forecast plots")
    past, = plt.plot(train.index, train, 'r.-', label="Train")
    future, = plt.plot(test.index[-70:], test[-70:], color ='blue', label="Test")
    predicted_future, = plt.plot(test.index[-70:], predicted[-70:], 'g.-', label="Predicted")
    plt.legend()
    st.pyplot(fig)
    
def plot_forecasts(df):
    fig, axs = plt.subplots(5, 2, figsize=(20, 20), sharex=False)
    axx = axs.ravel()
    for i in range(0, 10):
        df[df.columns[i]].loc[str(df.iloc[[0]].index[0]).split(' ')[0] : str(df.iloc[[len(df)-1]].index[0]).split(' ')[0]].plot(ax=axx[i])
        
        axx[i].set_xlabel("date")
        axx[i].set_ylabel(df.columns[i])
    st.pyplot(fig)
def train_test_split(data, n_test):
    return data[:-n_test], data[n_test:]

def measure_rmse(actual, predicted):
    return sqrt(mean_absolute_error(actual, predicted))

# HWSE
def exp_smoothing_forecast(history):
    #t, d, s, p, b, r = config
    history = np.array(history)
    model = HWES(history, seasonal='add', seasonal_periods=30)
    model_fit = model.fit(optimized=True)
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0], model_fit

def walk_forward_validation_hwse(data, n_test, data_mode):
    forecast_15days, forecast_1Month, forecast_3Months = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in data.columns:
        st.write("Fitting ", i)
        curr_data = data[i]
        predictions = list()
        train, test = train_test_split(curr_data, n_test)
        history = [x for x in train]
        for k in range(len(test)):
            yhat, model = exp_smoothing_forecast(history)
            predictions.append(yhat)
            history.append(test[k])
            
            
        
        mae = mean_absolute_error(test, predictions)
        mape = mean_absolute_percentage_error(test, predictions)
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        st.write("MAE  : ", mae)
        st.write("MAPE : ", mape)
        st.write("MSE  : ", mse)
        st.write("RMSE : ", rmse)
        
        plot_train_test_pred(train, test, predictions)
        
        # Forecast
       
        forecast_15days[i] = model.forecast((15+70))
        forecast_1Month[i] = model.forecast((30+70))
        forecast_3Months[i] = model.forecast((90+70))

        st.write("======================================")
        
    forecast_15days = forecast_15days.iloc[70:]
    forecast_1Month = forecast_1Month.iloc[70:]
    forecast_3Months = forecast_3Months.iloc[70:]
        
    forecast_15days["Date"] = pd.date_range(start= '2021-04-07', periods=15, freq='1D')
    forecast_15days = forecast_15days.set_index("Date")
    forecast_1Month["Date"] = pd.date_range(start= '2021-04-07', periods=30, freq='1D')
    forecast_1Month = forecast_1Month.set_index("Date")
    forecast_3Months["Date"] = pd.date_range(start= '2021-04-07', periods=90, freq='1D')
    forecast_3Months = forecast_3Months.set_index("Date")
            
    return forecast_15days, forecast_1Month, forecast_3Months

# Sarima
def sarima_forecast(history):
    history = np.array(history)
    model = SARIMAX(history, seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit(optimized=True)
    yhat = model_fit.predict(len(history), len(history))
    #print("yhat", yhat)
    return yhat[0], model_fit

def walk_forward_validation_sarima(data, n_test):
    forecast_15days, forecast_1Month, forecast_3Months = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in data.columns:
        st.write("Fitting ", i)
        curr_data = data[i]
        predictions = list()
        train, test = train_test_split(curr_data, n_test)
        history = [x for x in train]
        for k in range(len(test)):
            yhat, model = sarima_forecast(history)
            predictions.append(yhat)
            history.append(test[k])
        
        mae = mean_absolute_error(test, predictions)
        mape = mean_absolute_percentage_error(test, predictions)
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        st.write("MAE  : ", mae)
        st.write("MAPE : ", mape)
        st.write("MSE  : ", mse)
        st.write("RMSE : ", rmse)
        
        # Forecast
       
        forecast_15days[i] = model.forecast((15+70))
        forecast_1Month[i] = model.forecast((30+70))
        forecast_3Months[i] = model.forecast((90+70))
        st.write("======================================")
        
    forecast_15days = forecast_15days.iloc[70:]
    forecast_1Month = forecast_1Month.iloc[70:]
    forecast_3Months = forecast_3Months.iloc[70:]
        
    forecast_15days["Date"] = pd.date_range(start= '2021-04-07', periods=15, freq='1D')
    forecast_15days = forecast_15days.set_index("Date")
    forecast_1Month["Date"] = pd.date_range(start= '2021-04-07', periods=30, freq='1D')
    forecast_1Month = forecast_1Month.set_index("Date")
    forecast_3Months["Date"] = pd.date_range(start= '2021-04-07', periods=90, freq='1D')
    forecast_3Months = forecast_3Months.set_index("Date")
            
    return forecast_15days, forecast_1Month, forecast_3Months

# XGBoost
# transform timeseries dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    #n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with nan values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

def xgb_train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:, :]

def xgb_forecast(data, model, period):
    for col in data.columns:
        curr_data = data[col]
        predictions = []
        curr_value = []
        for p in range(0, period):
            if p == 0:
                curr_val = [curr_data[-1:]] 
            future = model.predict((curr_val))
            predictions.append(future[0])
            curr_val.clear()
            curr_val.append(future)
        return predictions

def xgbplothist(train, test, predicted):
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle("Train, Test, and Forecast plots")
    past, = plt.plot(train.index, train.values, 'r.-', label="Train")
    future, = plt.plot(test.index, test.values, color ='blue', label="Test")
    predicted_future, = plt.plot(test.index, predicted, 'g.-', label="Predicted")
    plt.legend()
    st.pyplot(fig)

# fit an xgboost model and make a one step prediction
def xgboost_model(train, testX):
    # transfor list into  array
    train = np.asarray(train)
    # split into input and output colums
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    yhat = model.predict([testX])
    #print("Test val : ", testX)
    #print("Predicted : ", yhat)
    return yhat[0], model

def walk_forward_validation_xbg(data, n_test):
    forecast_15days, forecast_1Month, forecast_3Months = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in data.columns:
        st.write("Fitting ", i)
        print("Fitting ", i)
        curr_data = data[i]
        curr_series = series_to_supervised(curr_data.values)
        #print("series", curr_data)
        predictions = list()
        train, test = train_test_split(curr_series, n_test)
        ##
        history = [x for x in train]
        for k in range(len(test)):
            testX, testy = test[k, :1], test[k, -1]
            # fit model on history and make a prediction
            yhat, model = xgboost_model(history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[k])
            #print(f"TestX : {testX} testy: {testy} pred {yhat}")
            
        
        mae = mean_absolute_error(test[:, -1], predictions)
        mape = mean_absolute_percentage_error(test[:, -1], predictions)
        mse = mean_squared_error(test[:, -1], predictions)
        rmse = np.sqrt(mse)
        st.write("MAE  : ", mae)
        st.write("MAPE : ", mape)
        st.write("MSE  : ", mse)
        st.write("RMSE : ", rmse)
        
        xgbplothist(curr_data[:-70], curr_data[-70:], predictions)
        
        # forecast
        forecast_15days[i] = xgb_forecast(data, model, (15+70))
        forecast_1Month[i] = xgb_forecast(data, model, (30+70))
        forecast_3Months[i] = xgb_forecast(data, model, (90+70))
        
    forecast_15days = forecast_15days.iloc[70:]
    forecast_1Month = forecast_1Month.iloc[70:]
    forecast_3Months = forecast_3Months.iloc[70:]
        
    forecast_15days["Date"] = pd.date_range(start= '2021-04-07', periods=15, freq='1D')
    forecast_15days = forecast_15days.set_index("Date")
    forecast_1Month["Date"] = pd.date_range(start= '2021-04-07', periods=30, freq='1D')
    forecast_1Month = forecast_1Month.set_index("Date")
    forecast_3Months["Date"] = pd.date_range(start= '2021-04-07', periods=90, freq='1D')
    forecast_3Months = forecast_3Months.set_index("Date")
        
    return forecast_15days, forecast_1Month, forecast_3Months
# Prophet
def prophet_plot_hist(train, test, predicted):
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle("Train, Test, and Forecast plots")
    past, = plt.plot(train.index, train['y'], 'r.-', label="Train")
    future, = plt.plot(test.index, test['y'], color ='blue', label="Test")
    predicted_future, = plt.plot(test.index, predicted, 'g.-', label="Predicted")
    plt.legend()
    st.pyplot(fig)
def prophet_forecast(model, start_date, period):
    dates = pd.date_range(start_date, periods=period)
    future = pd.DataFrame(dates)
    future.columns = ['ds']
    future['ds'] = pd.to_datetime(future['ds'])
    preds = model.predict(future)
    return preds['yhat']

def prophet_model(start_date, data):
    forecast_15days, forecast_1Month, forecast_3Months = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in data.columns:
        st.write(i)
        df = pd.DataFrame()
        df['ds'] = data.index
        df['index'] = data.index
        df = df.set_index('index')
        df['y'] = data[i].values
        df['ds'] = pd.to_datetime(df['ds'])
        train, test = df[:300], df[300:]
        model = prophet.Prophet(daily_seasonality=True, yearly_seasonality=False)
        model.fit(train)
        
        # validation
        y_pred = model.predict(test)['yhat']
        mae = mean_absolute_error(test['y'], y_pred)
        mape = mean_absolute_percentage_error(test['y'], y_pred)
        mse = mean_squared_error(test['y'], y_pred)
        rmse = np.sqrt(mse)
        st.write("MAE  : ", mae)
        st.write("MAPE : ", mape)
        st.write("MSE  : ", mse)
        st.write("RMSE : ", rmse)
        
        
        # forecast
        prophet_plot_hist(train, test, y_pred)
        
        forecast_15days[i] = prophet_forecast(model,start_date, (15+70))
        forecast_1Month[i] = prophet_forecast(model,start_date, (30+70))
        forecast_3Months[i] = prophet_forecast(model,start_date, (90+70))
        
        
    forecast_15days = forecast_15days.iloc[70:]
    forecast_1Month = forecast_1Month.iloc[70:]
    forecast_3Months = forecast_3Months.iloc[70:]
        
    forecast_15days["Date"] = pd.date_range(start= '2021-04-07', periods=15, freq='1D')
    forecast_15days = forecast_15days.set_index("Date")
    forecast_1Month["Date"] = pd.date_range(start= '2021-04-07', periods=30, freq='1D')
    forecast_1Month = forecast_1Month.set_index("Date")
    forecast_3Months["Date"] = pd.date_range(start= '2021-04-07', periods=90, freq='1D')
    forecast_3Months = forecast_3Months.set_index("Date")
        
    return forecast_15days, forecast_1Month, forecast_3Months

# LSTM
def lstm_split(data, split_idx):
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]
    return train, test

def lstm_split_series(series, n_past, n_future):
    series = np.array(series).reshape(-1, 1)
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
            
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)

def lstm_train_set_preprocess(train_set, n_past, n_future, n_features):
    X_train, y_train = lstm_split_series(train_set.values, n_past, n_future)
    X_train = X_train.reshape((X_train.shape[0], n_past, n_features))
    y_train = y_train.reshape((X_train.shape[0], n_future, n_features))
    return X_train, y_train
def lstm_test_set_preprocess(test_set, n_past, n_future, n_features):
    X_test, y_test = lstm_split_series(test_set.values, n_past, n_future)
    X_test = X_test.reshape((X_test.shape[0], n_past, n_features))
    y_test = y_test.reshape((X_test.shape[0], n_future, n_features))
    return X_test, y_test

def lstm_create_model(n_past, n_features, n_future):
    model = Sequential()
    model.add(Input(shape=(n_past, n_features)))
    model.add(LSTM(360, activation='relu', return_sequences=True))
    model.add(LSTM(30, activation='relu', return_sequences=False))
    model.add(Dense(n_future, activation='linear'))
    return model

def lstm_forecast(data, model, period):
    for col in data.columns:
        curr_data = data[col]
        past_10_hist = list(curr_data[-14:])
        predictions = []
        for p in range(0, period):
            past_10 = np.array(past_10_hist).reshape(1, 14, 1)
        
            #print(past_10)
            [future] = model.predict(past_10)
            future = future[0]
            predictions.append(future)
            #print("Future : ", future[0])
            past_10_hist.pop(0)
            past_10_hist.append(future)
        return predictions

def lstm_plot(train, test, y_test, predicted):
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle("Train, Test, and Forecast plots")
    past, = plt.plot(train.index, train, 'r.-', label="Train")
    future, = plt.plot(test.index[-70:], y_test[-70:], color ='blue', label="Test")
    predicted_future, = plt.plot(test.index[-70:], predicted[-70:], 'g.-', label="Predicted")
    plt.legend()
    st.pyplot(fig)

def lstm_models(data, split_idx, n_past, n_future, n_features, model):
    forecast_15days, forecast_1Month, forecast_3Months = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in range(0, len(data.columns)):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore')
        tf.keras.backend.clear_session()
        curr_data = data[data.columns[i]]
        st.write("Fitting ", data.columns[i])
        train, test = lstm_split(curr_data, split_idx)
        X_train, y_train = lstm_train_set_preprocess(train, n_past, n_future, n_features)
        X_test, y_test = lstm_test_set_preprocess(test, n_past, n_future, n_features)
        

        model.compile(optimizer='adam', loss='mse')
        history=model.fit(X_train,y_train,epochs=20, batch_size=30,verbose=1)
        
        # validation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
        mape = mean_absolute_percentage_error(y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
        mse = mean_squared_error(y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
        rmse = np.sqrt(mse)
        st.write("MAE  : ", mae)
        st.write("MAPE : ", mape)
        st.write("MSE  : ", mse)
        st.write("RMSE : ", rmse)
        
        lstm_plot(train, test, y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
        
        # Forecast
       
        forecast_15days[data.columns[i]] = lstm_forecast(data, model, (15+70))
        forecast_1Month[data.columns[i]] = lstm_forecast(data, model, (30+70))
        forecast_3Months[data.columns[i]] = lstm_forecast(data, model, (90+70))
        
    forecast_15days = forecast_15days.iloc[70:]
    forecast_1Month = forecast_1Month.iloc[70:]
    forecast_3Months = forecast_3Months.iloc[70:]
        
    forecast_15days["Date"] = pd.date_range(start= '2021-04-07', periods=15, freq='1D')
    forecast_15days = forecast_15days.set_index("Date")
    forecast_1Month["Date"] = pd.date_range(start= '2021-04-07', periods=30, freq='1D')
    forecast_1Month = forecast_1Month.set_index("Date")
    forecast_3Months["Date"] = pd.date_range(start= '2021-04-07', periods=90, freq='1D')
    forecast_3Months = forecast_3Months.set_index("Date")
        
    return forecast_15days, forecast_1Month, forecast_3Months

# LSTM_GRU
# METRIC
def lstm_gru_mean_absolute_scaled_error(y_true, y_pred, y_train):
    e_t = y_true - y_pred
    scale = mean_absolute_error(y_train[1:], y_train[:-1])
    return np.mean(np.abs(e_t / scale))

def lstm_gru_plothist(dates, train, test, predicted):
    for i in range(0, train.shape[1]):
        fig = plt.figure(figsize=(15, 4))
        fig.suptitle("Train, Test, and Forecast plots")
        past, = plt.plot(dates[:1716], train.reshape(50, 1716)[i], 'r.-', label="Train")
        future, = plt.plot(dates[1716:], test, color ='blue', label="Actual")
        predicted_future, = plt.plot(test.index, predicted, 'g.-', label="Predicted")
        plt.legend()
        st.pyplot(fig)

def lstm_gru_forecast(data, model, period):
    curr_data = data
    past_14_hist = list(curr_data[-14:])
    predictions = []
    for p in range(0, period):
        past_14 = np.array(past_14_hist)
    
        future = model.predict(past_14)
        predictions.append(future[0])
        past_14_hist.pop(0)
        past_14_hist.append(future)
    return np.array(predictions)

def lstm_gru_plot_forecasts(df):
    fig, axs = plt.subplots(5, 2, figsize=(20, 20), sharex=False)
    axx = axs.ravel()
    for i in range(0, 10):
        df[df.columns[i]].loc[str(df.iloc[[0]].index[0]).split(' ')[0] : str(df.iloc[[len(df)-1]].index[0]).split(' ')[0]].plot(ax=axx[i])
        axx[i].set_xlabel("date")
        axx[i].set_ylabel(df.columns[i]) 
    st.pyplot(fig)
# NBEATS
# METRIC
def nbeats_mean_absolute_scaled_error(y_true, y_pred, y_train):
    e_t = y_true - y_pred
    scale = mean_absolute_error(y_train[1:], y_train[:-1])
    return np.mean(np.abs(e_t / scale))
def nbeats_forecast(testX, model, period):
    curr_data = testX
    past_14_hist = None
    predictions = []
    for p in range(0, period):
        if p == 0:
            past_14_hist = curr_data[-1:]
            #print(past_14.shape)
        future = model.predict(past_14_hist.reshape(50, 14, 1))
        predictions.append(future[-1:].reshape(1, 50))
        #temp = past_14_hist
        past_14_hist = future[36:] #np.append(temp, future[0].reshape(1, 50), axis=1)
        
        #past_14_hist = future.reshape(1, 14, 50)
    preds = np.array(predictions).reshape(period, 50)
    return preds
# CNN_Wavenets
def cnn_wavenets_plot_random_series(df, n_series, data_start_date, data_end_date):
    
    sample = df.sample(n_series, random_state=8)
    page_labels = sample['Unnamed: 0'].tolist()
    series_samples = sample.loc[:,data_start_date:data_end_date]
    
    fig = plt.figure(figsize=(10,6))
    
    for i in range(series_samples.shape[0]):
        np.log1p(pd.Series(series_samples.iloc[i]).astype(np.float64)).plot(linewidth=1.5)
    
    plt.title('APN)')
    plt.legend(page_labels)
    st.pyplot(fig)
    
def cnn_wavenets_get_time_block_series(series_array, date_to_index, start_date, end_date):
    
    inds = date_to_index[start_date:end_date]
    return series_array[:,inds]

def cnn_wavenets_transform_series_encode(series_array):
    
    series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0
    series_mean = series_array.mean(axis=1).reshape(-1,1) 
    series_array = series_array - series_mean
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    
    return series_array, series_mean

def cnn_wavenets_transform_series_decode(series_array, encode_series_mean):
    
    series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0
    series_array = series_array - encode_series_mean
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    
    return series_array

def cnn_wavenets_mean_absolute_scaled_error(y_true, y_pred, y_train):
    e_t = y_true - y_pred
    scale = mean_absolute_error(y_train[1:], y_train[:-1])
    return np.mean(np.abs(e_t / scale))

def cnn_wavenets_forecast(model, series, period):
    predictions = []
    last_tmstep = None
    for p in range(0, period):
        if p == 0:
            last_tmstep = series[-50:]
            
        temp = last_tmstep
        future_pred = model.predict(last_tmstep)#.reshape(1, 50)
        predictions.append(future_pred.reshape(temp.shape[2], temp.shape[0]))
        init_last_tmstep = last_tmstep.reshape(temp.shape[0], temp.shape[1])
        rest_vals = init_last_tmstep
        last_tmstep = np.append(rest_vals, future_pred.reshape(temp.shape[0], temp.shape[2]), axis=1)
        last_tmstep = last_tmstep.reshape(last_tmstep.shape[1], temp.shape[0])[1:]
        last_tmstep = last_tmstep.reshape(temp.shape[0], temp.shape[1] )
        last_tmstep = last_tmstep.reshape(temp.shape[0], temp.shape[1], temp.shape[2])
    return predictions

def cnn_wavenets_plot_forecasts(df):
    fig, axs = plt.subplots(5, 2, figsize=(20, 20), sharex=False)
    axx = axs.ravel()
    for i in range(0, 10):
        df[df.columns[i]].loc[str(df.iloc[[0]].index[0]).split(' ')[0] : str(df.iloc[[len(df)-1]].index[0]).split(' ')[0]].plot(ax=axx[i])
        axx[i].set_xlabel("date")
        axx[i].set_ylabel(df.columns[i])  