import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go
import os
from . import TimeSeriesUtils
from numpy import sqrt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, Dropout
from nbeats_keras.model import NBeatsNet as NBeatsKeras
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, concatenate
from tensorflow.keras.optimizers import Adam
from datetime import timedelta
import keras.backend as k
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

def msc_traffic_process_render(dataset_name, model_name):
    
    st.subheader('Exploratory data analysis')
    with st.spinner('Loading Data...'):        
        prompt = st.empty()     
        #DATA LOADING
        data = pd.read_csv("data/msc_traffic_daily_1year_10nodes.csv")
        data.drop("Unnamed: 1", axis=1, inplace=True)
        # Transpose
        data = data.T
        # make first row as column headers and select all rows except the first
        data.columns = data.iloc[0]
        data = data[1:]
        data = data.rename(columns=({"index":"Date"}))
        data.index = pd.to_datetime(data.index)
            # column names
        cols = [data.columns]
        with st.expander('Raw Sample Data', expanded=False):
            st.write('Head')
            st.write(data.head(5))
            st.write('Tail')
            st.write(data.tail(5))
        with st.expander('Data Summary', expanded=False):
            st.write('Shape:')
            st.write(data.shape)
            st.write('Info:')
            st.write(data.info())
            st.write('Null values before imputation:')
            st.table(data.isnull().sum())
            data = data.fillna(method='bfill')
            st.write('Null values after imputation:')
            st.table(data.isnull().sum())
            st.write('Dataset description')
            st.table(data.describe())
        with st.expander('Data summary visualisation', expanded=False):
            st.write('Summary plots')
            fig, ax = plt.subplots(5, 2, figsize=(15, 12))
            fig.tight_layout(pad=3)
            ax[0, 0].plot(data.index, data['NR_MSC_1'])
            ax[0, 1].plot(data.index, data['NR_MSC_2'])
            ax[1, 0].plot(data.index, data['NR_MSC_3'])
            ax[1, 1].plot(data.index, data['NR_MSC_4'])
            ax[2, 0].plot(data.index, data['NR_MSC_5'])
            ax[2, 1].plot(data.index, data['NR_MSC_6'])
            ax[3, 0].plot(data.index, data['NR_MSC_7'])
            ax[3, 1].plot(data.index, data['NR_MSC_8'])
            ax[4, 0].plot(data.index, data['NR_MSC_9'])
            ax[4, 1].plot(data.index, data['NR_MSC_10'])
            st.pyplot(fig)

            # Initialize figure with subplots
            fig = make_subplots(
                rows=5, cols=2, subplot_titles=("NR_MSC_1", "NR_MSC_2", "NR_MSC_3", "NR_MSC_4", "NR_MSC_5", "NR_MSC_6", "NR_MSC_7", "NR_MSC_8", "NR_MSC_9", "NR_MSC_10")
            )

            # Add traces
            fig.add_trace(go.Scatter(x=data.index, y=data['NR_MSC_1'], name = "NR_MSC_1"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['NR_MSC_2'], name = "NR_MSC_2"), row=1, col=2)
            fig.add_trace(go.Scatter(x=data.index, y=data['NR_MSC_3'], name = "NR_MSC_3"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['NR_MSC_4'], name = "NR_MSC_4"), row=2, col=2)
            fig.add_trace(go.Scatter(x=data.index, y=data['NR_MSC_5'], name = "NR_MSC_5"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['NR_MSC_6'], name = "NR_MSC_6"), row=3, col=2)
            fig.add_trace(go.Scatter(x=data.index, y=data['NR_MSC_7'], name = "NR_MSC_7"), row=4, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['NR_MSC_8'], name = "NR_MSC_8"), row=4, col=2)
            fig.add_trace(go.Scatter(x=data.index, y=data['NR_MSC_9'], name = "NR_MSC_9"), row=5, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['NR_MSC_10'], name = "NR_MSC_10"), row=5, col=2)
            fig.update_layout(title_text="MSC Traffic", height=1500)
            st.plotly_chart(fig)
        with st.expander('Data Box plots', expanded=False):
            #drawing figure with title and single axis. Size and resolution are specified
            fig = plt.figure(figsize=(15,5))
            plt.title('MSC Voice Traffic',fontsize=15)
            #setting y axis label
            plt.ylabel('MSC Traffic')
            #rotating x axis ticks by 90 degrees
            plt.xticks(rotation=0)
            #drawing boxplot for
            sns.boxplot(data=data)
            st.pyplot(fig)
            fig = go.Figure()
            for col in data:
                fig.add_trace(go.Box(y=data[col].values, name=data[col].name))
            fig.update_layout(title_text="MSC Traffic", height=700)                       
            st.plotly_chart(fig)
    prompt.success('Data Loaded!')

    st.subheader('Training plots and performance')
    with st.spinner('Prediciting...'):
        prompt = st.empty()
        if model_name == 'HWES':
            l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            tr, te  = TimeSeriesUtils.train_test_split(l, 3)
            with st.expander('Training & Testing split', expanded=False):
                st.write(tr, te)
            with st.expander('Fitting Daily Data', expanded=False):
                f15d, f1m, f3m = TimeSeriesUtils.walk_forward_validation_hwse(data, 70, "Days")
        elif model_name == 'SARIMA':
            with st.expander('Auto-correlation plots', expanded=False):
                for i in data.columns:
                    plot_acf(data[i], lags=12, title=i + " Autocorrelation"), plot_pacf(data[i], lags=12, title= i + " Partial Autocorrelation")
                    st.pyplot(plt.gcf())
            with st.expander('Fitting Daily Data', expanded=False):
                f15d, f1m, f3m = TimeSeriesUtils.walk_forward_validation_sarima(data, 70)
        elif model_name =='XGBoost':
            with st.expander('Fitting Daily Data', expanded=False):
                f15d, f1m, f3m = TimeSeriesUtils.walk_forward_validation_xbg(data, 70)
        elif model_name =='Prophet':
            with st.expander('Fitting Daily Data', expanded=False):
                f15d, f1m, f3m = TimeSeriesUtils.prophet_model('2021-04-06', data)
        elif model_name =='LSTM':
            with st.expander('Preprocessing data for LSTM', expanded=False):
                n_past = 14
                n_future = 10
                n_features = 1
                scaler = MinMaxScaler(feature_range=(0, 1))
                cols = data.columns
                dt = pd.DataFrame(scaler.fit_transform(data))
                dt.columns = cols
                dt.index = data.index
                st.write(dt.head())
            with st.expander('Creating Model', expanded=False):
                k.clear_session()
                model = TimeSeriesUtils.lstm_create_model(n_past, n_features, n_future)
                st.write('Model Created!')
            with st.expander('Fitting Daily Data', expanded=False):
                f15d, f1m, f3m = TimeSeriesUtils.lstm_models(data, 300, n_past, n_future, n_features, model)   
    prompt.success('Training Completed!')

    st.subheader('Forecast dataframes')
    with st.spinner('Forecasting...'):
        prompt = st.empty()
        with st.expander('15 Day', expanded=False):
            st.write('Data dimensions:')
            st.write(f15d.shape)
            st.write('15 Day Dataframe')
            st.write(f15d.head())
        with st.expander('1 Month', expanded=False):
            st.write('Data dimensions:')
            st.write(f1m.shape)
            st.write('1 Month Dataframe')
            st.write(f1m.head())
        with st.expander('3 Month', expanded=False):
            st.write('Data dimensions:')
            st.write(f3m.shape)
            st.write('3 Month Dataframe')
            st.write(f3m.head())
    prompt.success('Forecasting Completed!')

    st.subheader('Forecast plots')
    with st.spinner('Plotting Forecast...'):
        prompt = st.empty()
        with st.expander('15 Day', expanded=False):
           TimeSeriesUtils.plot_forecasts(f15d)
        with st.expander('1 Month', expanded=False):
           TimeSeriesUtils.plot_forecasts(f1m)
        with st.expander('3 Month', expanded=False):
            TimeSeriesUtils.plot_forecasts(f3m)
    prompt.success('Plotting Completed!')


def apn_utilisation_process_render(dataset_name, model_name):
    st.subheader('Exploratory data analysis')
    with st.spinner('Loading Data...'):        
        prompt = st.empty()     
        #DATA LOADING
        apn = pd.read_csv('data/apn_utilization_hourly_3months_50nodes.csv', index_col=0, parse_dates=True)
        with st.expander('Raw Sample Data', expanded=False):
            st.write('Head')
            # DISPLAY FIRST FIVE ROWS
            st.write(apn.head())
        with st.expander('Data Summary', expanded=False):
            # EXTRACT COLUMNS
            cols = apn.columns[2:]   
            # CHECK FOR NULL VALUES
            st.write('Null values before imputation:')
            st.write(apn.isnull().sum().sum())
            apn = apn.fillna(method='bfill')
            st.write('Null values after imputation:')
            st.write(apn.isnull().sum().sum())
            st.write('Shape:')
            st.write(apn.shape)
            # DROP UNNECESSARY COLUMN
            apn.drop('Unnamed: 1', axis=1, inplace=True)
            # TRANSPOSE THE DATASET
            apn = apn.T
            # SET INDEX TO BE OF FORM DATETIME
            apn.index = pd.to_datetime(apn.index)
            # SET INDEX NAME TO `DATE`
            apn.index.name = 'DATE'
            st.write('Dataset Description:')
            st.table(apn.describe())
            st.write('Sample Data:')
            st.write(apn.head(3))
        with st.expander('Data Preparation', expanded=False):
            # REPLICATE THE DATA
            df = apn.copy()
            # separate dates for future plotting
            train_dates = pd.to_datetime(df.index)
            st.write('Replicated dataframe created')
            st.write('Dates for training:')
            st.write(train_dates[:15])
            # TRAINING VARIABLES
            cols = list(df)[0:50]
            st.write('Training Variables')
            st.write(cols[:5])
    st.subheader('Training plots and performance')
    with st.spinner('Prediciting...'):
        prompt = st.empty()
        if model_name == 'LSTM_GRU':
            with st.expander('Training & Testing split', expanded=False):
                # CONVERT ALL DATAPOINTS TO FLOAT
                df_for_training = df[cols].astype(float)
                # NORMALIZE DATASET FOR : FASTER TRAINING AND BETTER PERFORMANCE
                scaler = StandardScaler()
                scaler = scaler.fit(df_for_training)
                df_for_training_scaled = scaler.transform(df_for_training)
                # CREATE EMPTY XTRAIN AND YTRAIN LISTS
                trainX = []
                trainy  = []
                # FUTURE AND PAST TIMESTEPS
                n_future = 1
                n_past = 14
                # DATASET TO TIMESERIES
                for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
                    trainX.append(df_for_training_scaled[i - n_past:i])
                    trainy.append(df_for_training_scaled[i + n_future - 1:i + n_future])
                # CONVERT LISTS TO NUMPY ARRAYS
                X = np.array(trainX)
                y = np.array(trainy)
                # SPLIT DATASET 80% TRAINING AND 20% TEST
                trainX, testX = X[:1716], X[1716:]
                trainy, testy = y[:1716], y[1716:]
                # PRINT SHAPES
                st.write('Training Data Shape:')
                st.write(trainX.shape, trainy.shape)
                st.write('Testing Data Shape:')
                st.write(testX.shape, testy.shape)
            with st.expander('Fitting Daily Data', expanded=False):
                # MODEL ARCHITECTURE
                k.clear_session()
                model = Sequential()
                model.add(LSTM(100, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
                model.add(GRU(120, activation='relu', return_sequences=True))
                model.add(LSTM(90, activation='relu', return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(trainX.shape[2], activation='linear'))
                # COMPILE MODEL WITH adam `optimizer` AND  `mse` loss function
                model.compile(optimizer='adam', loss='mse')
                model.summary()
                st.write('Model Created!')
                # FIT THE MODEL ON TRAINING SETS
                history = model.fit(trainX, trainy, epochs=50, batch_size=8, verbose=1)
                st.write('Model Loss Plot')
                # plot history
                plt.plot(history.history['loss'], label='Train')
                #plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                st.pyplot(plt.gcf())
            with st.expander('Model performance', expanded=False):
                st.write('Global Metrics')
                # PERFORM PREDICTION
                pred = model.predict(testX)
                # PRINT PERFORMANCE METRICS
                gtrue = testy.reshape(50, 430)
                gpredicted = pred.reshape(50, 430)

                mae_metr = mean_absolute_error(gtrue, gpredicted)
                mape_metr = mean_absolute_percentage_error(gtrue, gpredicted)
                mse_metr = mean_squared_error(gtrue, gpredicted)
                rmse_metr = np.sqrt(mse_metr)
                mase_metr = TimeSeriesUtils.lstm_gru_mean_absolute_scaled_error(gtrue, gpredicted, trainy.reshape(trainy.shape[0], trainy.shape[2]))

                st.write("MAE  : ", mae_metr)
                st.write("MAPE : ", mape_metr)
                st.write("MSE  : ", mse_metr)
                st.write("RMSE : ", rmse_metr)
                st.write("MASE : ", mase_metr)
                st.write("==============")

                true = testy.reshape(430, 50) 
                predicted = pred
                st.write('Feature metrics')
                ytrue = true
                predicted = predicted
                try:
                    for i in range(len(ytrue)):
                        mae_metr = mean_absolute_error(ytrue[i], predicted[i])
                        mape_metr = mean_absolute_percentage_error(ytrue[i], predicted[i])
                        mse_metr = mean_squared_error(ytrue[i], predicted[i])
                        rmse_metr = np.sqrt(mse_metr)
                        st.write(apn.columns[i])
                        st.write("MAE  : ", mae_metr)
                        st.write("MAPE : ", mape_metr)
                        st.write("MSE  : ", mse_metr)
                        st.write("RMSE : ", rmse_metr)
                        st.write("==============")
                except IndexError:
                    pass

                tx = trainX.reshape(14, 1716, 50)[:1].reshape(1716, 1, 50).reshape(1716, 50)
                ty = trainy.reshape(1716, 50)   
                tsX = testX.reshape(14, 430, 50)[:1].reshape(430, 1, 50).reshape(430, 50)
                dates = apn.index
                tx.reshape(50, 1716)[1].shape
            
                f7 = TimeSeriesUtils.lstm_gru_forecast(testX, model, 168)
                f7 = scaler.inverse_transform(f7)
                cols = apn.columns
                f7d = pd.DataFrame(f7,columns=cols)
                f7d["Date"] = pd.date_range(start= '2021-04-01', periods=168, freq='1H')
                f7d = f7d.set_index("Date")

                f15 = TimeSeriesUtils.lstm_gru_forecast(testX, model, 360)
                f15 = scaler.inverse_transform(f15)
                cols = apn.columns
                f15d = pd.DataFrame(f15,columns=cols)
                f15d["Date"] = pd.date_range(start= '2021-04-01', periods=360, freq='1H')
                f15d = f15d.set_index("Date")

                f1 = TimeSeriesUtils.lstm_gru_forecast(testX, model, 720)
                f1 = scaler.inverse_transform(f1)
                cols = apn.columns
                f1m = pd.DataFrame(f1,columns=cols)
                f1m["Date"] = pd.date_range(start= '2021-04-01', periods=720, freq='1H')
                f1m = f1m.set_index("Date")
                
        elif model_name == 'NBEATS':
            with st.expander('Training & Testing split', expanded=False):
                # CONVERT ALL DATAPOINTS TO FLOAT
                df_for_training = df[cols].astype(float)
                # NORMALIZE DATASET FOR : FASTER TRAINING AND BETTER PERFORMANCE
                scaler = StandardScaler()
                scaler = scaler.fit(df_for_training)
                df_for_training_scaled = scaler.transform(df_for_training)
                # CREATE EMPTY XTRAIN AND YTRAIN LISTS
                trainX = []
                trainy  = []
                # FUTURE AND PAST TIMESTEPS
                n_future = 1
                n_past = 14
                # DATASET TO TIMESERIES
                for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
                    trainX.append(df_for_training_scaled[i - n_past:i])
                    trainy.append(df_for_training_scaled[i + n_future - 1:i + n_future])
                # CONVERT LISTS TO NUMPY ARRAYS
                X = np.array(trainX)
                y = np.array(trainy)
                # SPLIT DATASET 80% TRAINING AND 20% TEST
                trainX, testX = X[:1716], X[1716:]
                trainy, testy = y[:1716], y[1716:]
                # PRINT SHAPES
                st.write('Training Data Shape:')
                st.write(trainX.shape, trainy.shape)
                st.write('Testing Data Shape:')
                st.write(testX.shape, testy.shape)
            with st.expander('Fitting Daily Data', expanded=False):
                # MODEL ARCHITECTURE
                warnings.filterwarnings(action='ignore')
                num_samples, time_steps, input_dim, output_dim = trainX.shape[0], trainX.shape[1], trainX.shape[2], trainX.shape[2] 
                backend = NBeatsKeras(
                                backcast_length=time_steps, forecast_length=output_dim,
                                stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
                                nb_blocks_per_stack=2, thetas_dim=(4, 4), share_weights_in_stack=True,
                                hidden_layer_units=128)
                
                backend.compile(loss='mae', optimizer='adam')
                backend.fit(trainX, trainy, epochs=30, batch_size=64, verbose=False)
                st.write('Model Created!')
            with st.expander('Model performance', expanded=False):
                preds = backend.predict(testX).reshape(430, 1, 50)
    
                mae = mean_absolute_error(testy[0], preds[0])
                mape = mean_absolute_percentage_error(testy[0], preds[0])
                mse = mean_squared_error(testy[0], preds[0])
                rmse = np.sqrt(mse)
                mase = TimeSeriesUtils.nbeats_mean_absolute_scaled_error(testy[0], preds[0], trainy.reshape(1716, 50))

                st.write("Global Metrics : ")
                st.write("MAE  : ", mae)
                st.write("MAPE : ", mape)
                st.write("MSE  : ", mse)
                st.write("RMSE : ", rmse)
                st.write("MASE : ", mase)
                
                st.write("==============")
    
                try:
                    st.write("Feature Metrics : ")
                    for i in range(0, len(testy)):
                        mae = mean_absolute_error(testy[i], preds[i])
                        mape = mean_absolute_percentage_error(testy[i], preds[i])
                        mse = mean_squared_error(testy[i], preds[i])
                        rmse = np.sqrt(mse)
                        mase = TimeSeriesUtils.nbeats_mean_absolute_scaled_error(testy[i], preds[i], trainy.reshape(1716, 50))

                        st.write(apn.columns[i])
                        st.write("MAE  : ", mae)
                        st.write("MAPE : ", mape)
                        st.write("MSE  : ", mse)
                        st.write("RMSE : ", rmse)
                        st.write("MASE : ", mase)
                        st.write("==============")
                except IndexError as e:
                    pass

                f7 = TimeSeriesUtils.nbeats_forecast(testX, backend, 168)
                f15 = TimeSeriesUtils.nbeats_forecast(testX, backend, 360)
                f1 = TimeSeriesUtils.nbeats_forecast(testX, backend, 720)
                inv_f7 = scaler.inverse_transform(f7)
                inv_f15 = scaler.inverse_transform(f15)
                inv_f1 = scaler.inverse_transform(f1)
                
                f7 = pd.DataFrame(inv_f7, columns=df.columns)
                f7["Date"] = pd.date_range(start= '2021-04-01', periods=168, freq='1H')
                f7 = f7.set_index("Date")
                f15 = pd.DataFrame(inv_f15, columns=df.columns)
                f15["Date"] = pd.date_range(start= '2021-04-01', periods=360, freq='1H')
                f15 = f15.set_index("Date")
                f1 = pd.DataFrame(inv_f1, columns=df.columns)
                f1["Date"] = pd.date_range(start= '2021-04-01', periods=720, freq='1H')
                f1 = f1.set_index("Date")
                
                f7d, f15d, f1m = f7, f15, f1
        elif model_name =='CNN_Wavenets':
            # CONVERT ALL DATAPOINTS TO FLOAT
            df_for_training = df[cols].astype(float)
            # NORMALIZE DATASET FOR : FASTER TRAINING AND BETTER PERFORMANCE
            scaler = StandardScaler()
            scaler = scaler.fit(df_for_training)
            df_for_training_scaled = scaler.transform(df_for_training)
            # CREATE EMPTY XTRAIN AND YTRAIN LISTS
            trainX = []
            trainy  = []
            # FUTURE AND PAST TIMESTEPS
            n_future = 1
            n_past = 14
            # DATASET TO TIMESERIES
            for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
                trainX.append(df_for_training_scaled[i - n_past:i])
                trainy.append(df_for_training_scaled[i + n_future - 1:i + n_future])
            # CONVERT LISTS TO NUMPY ARRAYS
            X = np.array(trainX)
            y = np.array(trainy)
            # SPLIT DATASET 80% TRAINING AND 20% TEST
            trainX, testX = X[:1716], X[1716:]
            trainy, testy = y[:1716], y[1716:]
            with st.expander('Training & Testing split', expanded=False):
                data = pd.read_csv('data/apn_utilization_hourly_3months_50nodes.csv')
                data.drop('Unnamed: 1', axis=1, inplace=True)
                cols = pd.to_datetime(data.columns[1:])
                df = data.copy()
                data_start_date = df.columns[2]
                data_end_date = df.columns[-1]
                st.write('Selected Data ranges from %s to %s' % (data_start_date, data_end_date))
                st.write('Plot Random Series:')
                TimeSeriesUtils.cnn_wavenets_plot_random_series(df, 6, data_start_date, data_end_date)
                df = df.set_index('Unnamed: 0')

                pred_steps = 1
                pred_length=timedelta(pred_steps)

                first_day = pd.to_datetime(data_start_date) 
                last_day = pd.to_datetime(data_end_date)

                val_pred_start = last_day - pred_length + timedelta(1)
                val_pred_end = last_day

                train_pred_start = val_pred_start - pred_length
                train_pred_end = val_pred_start - timedelta(days=1) 
                enc_length = train_pred_start - first_day

                train_enc_start = first_day
                train_enc_end = train_enc_start + enc_length - timedelta(1)

                val_enc_start = train_enc_start + pred_length
                val_enc_end = val_enc_start + enc_length - timedelta(1)
                st.write('Train encoding:', train_enc_start, '-', train_enc_end)
                st.write('Train prediction:', train_pred_start, '-', train_pred_end, '\n')
                st.write('Val encoding:', val_enc_start, '-', val_enc_end)
                st.write('Val prediction:', val_pred_start, '-', val_pred_end)

                st.write('Encoding interval:', enc_length.days)
                st.write('Prediction interval:', pred_length.days)
                date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in df.columns[1:]]),
                                        data=[i for i in range(len(df.columns[1:]))])
                series_array = df[df.columns[1:]].values
                st.write(series_array.shape)

            with st.expander('Fitting Daily Data', expanded=False):
                # MODEL ARCHITECTURE
                k.clear_session()
                # convolutional layer parameters
                n_filters = 32 
                filter_width = 2
                dilation_rates = [2**i for i in range(8)] 

                # define an input history series and pass it through a stack of dilated causal convolutions. 
                history_seq = Input(shape=(None, 1))
                x = history_seq

                for dilation_rate in dilation_rates:
                    x = Conv1D(filters=n_filters,
                            kernel_size=filter_width, 
                            padding='causal',
                            dilation_rate=dilation_rate)(x)

                x = Dense(128, activation='relu')(x)
                x = Dense(64, activation='relu')(x)
                x = Dropout(.2)(x)
                x = Dense(1, activation='linear')(x)

                # extract the last 14 time steps as the training target
                def slice(x, seq_length):
                    return x[:,-seq_length:,:]

                pred_seq_train = Lambda(slice, arguments={'seq_length':1})(x)

                model = Model(history_seq, pred_seq_train)
                st.write('Model Created!')
                k.clear_session()
                # FIT THE MODEL ON TRAINING SETS
                first_n_samples = 40000
                batch_size = 2**11
                epochs = 50

                # sample of series from train_enc_start to train_enc_end  
                encoder_input_data = TimeSeriesUtils.cnn_wavenets_get_time_block_series(series_array, date_to_index, 
                                                        train_enc_start, train_enc_end)[:first_n_samples]
                encoder_input_data, encode_series_mean = TimeSeriesUtils.cnn_wavenets_transform_series_encode(encoder_input_data)

                # sample of series from train_pred_start to train_pred_end 
                decoder_target_data = TimeSeriesUtils.cnn_wavenets_get_time_block_series(series_array, date_to_index, 
                                                            train_pred_start, train_pred_end)[:first_n_samples]
                decoder_target_data = TimeSeriesUtils.cnn_wavenets_transform_series_decode(decoder_target_data, encode_series_mean)

                # we append a lagged history of the target series to the input data, 
                # so that we can train with teacher forcing
                lagged_target_history = decoder_target_data[:,:-1,:1]
                encoder_input_data = np.concatenate([encoder_input_data, lagged_target_history], axis=1)

                model.compile(Adam(), loss='mean_absolute_error')
                history = model.fit(encoder_input_data, decoder_target_data,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_split=0.2)
                st.write('Model Loss Plot')
                fig = plt.figure()
                plt.plot(history.history['loss'], label='Train')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                st.pyplot(fig)
    
            with st.expander('Model performance', expanded=False):
                # VAlidation
                # sample of series from train_enc_start to train_enc_end  
                val_input_data = TimeSeriesUtils.cnn_wavenets_get_time_block_series(series_array, date_to_index, 
                                                        val_enc_start, val_enc_end)[:first_n_samples]
                val_input_data, val_encode_series_mean = TimeSeriesUtils.cnn_wavenets_transform_series_encode(val_input_data)

                # sample of series from train_pred_start to train_pred_end 
                val_target_data = TimeSeriesUtils.cnn_wavenets_get_time_block_series(series_array, date_to_index, 
                                                            val_pred_start, val_pred_end)[:first_n_samples]
                val_target_data = TimeSeriesUtils.cnn_wavenets_transform_series_decode(val_target_data, val_encode_series_mean)

                st.write('Validation Target Reshaping')
                st.write(val_target_data.reshape(1, 50))
                predicted = model.predict(val_input_data).reshape(50, 1)
                ytrue = val_target_data.reshape(50, 1)

                st.write('Global Metrics')
                # Performance metrics
                mae_metr = mean_absolute_error(ytrue, predicted)
                mape_metr = mean_absolute_percentage_error(ytrue, predicted)
                mse_metr = mean_squared_error(ytrue, predicted)
                rmse_metr = np.sqrt(mse_metr)
                mase_metr = TimeSeriesUtils.cnn_wavenets_mean_absolute_scaled_error(ytrue, predicted, decoder_target_data.reshape(50, 1))
                    
                st.write("MAE  : ", mae_metr)
                st.write("MAPE : ", mape_metr)
                st.write("MSE  : ", mse_metr)
                st.write("RMSE : ", rmse_metr)
                st.write("MASE : ", mase_metr)
                st.write("==============")

                # Feature metrics
                for i in range(len(ytrue)):
                    mae_metr = mean_absolute_error(ytrue[i], predicted[i])
                    mape_metr = mean_absolute_percentage_error(ytrue[i], predicted[i])
                    mse_metr = mean_squared_error(ytrue[i], predicted[i])
                    rmse_metr = np.sqrt(mse_metr)
                    st.write(apn.columns[i])
                    st.write("MAE  : ", mae_metr)
                    st.write("MAPE : ", mape_metr)
                    st.write("MSE  : ", mse_metr)
                    st.write("RMSE : ", rmse_metr)
                    st.write("==============")

                f7 = np.array(TimeSeriesUtils.cnn_wavenets_forecast(model, encoder_input_data, 168)).reshape(168, 50)
                cols = apn.columns
                f7d = pd.DataFrame(f7,columns=cols)
                f7d["Date"] = pd.date_range(start= '2021-04-01', periods=168, freq='1H')
                f7d = f7d.set_index("Date")

                f15 = np.array(TimeSeriesUtils.cnn_wavenets_forecast(model, encoder_input_data, 360)).reshape(360, 50)
                cols = apn.columns
                f15d = pd.DataFrame(f15,columns=cols)
                f15d["Date"] = pd.date_range(start= '2021-04-01', periods=360, freq='1H')
                f15d = f15d.set_index("Date")

                f1 = np.array(TimeSeriesUtils.cnn_wavenets_forecast(model, encoder_input_data, 720)).reshape(720, 50)
                cols = apn.columns
                f1m = pd.DataFrame(f1,columns=cols)
                f1m["Date"] = pd.date_range(start= '2021-04-07', periods=720, freq='1H')
                f1m = f1m.set_index("Date")

    prompt.success('Training Completed!')

    st.subheader('Forecast dataframes')
    with st.spinner('Forecasting...'):
        prompt = st.empty()
        with st.expander('7 Day', expanded=False):
            st.write('Data dimensions:')
            st.write(f7d.shape)
            st.write('7 Day Dataframe')
            st.write(f7d.head())
        with st.expander('15 Day', expanded=False):
            st.write('Data dimensions:')
            st.write(f15d.shape)
            st.write('15 Day Dataframe')
            st.write(f15d.head())
        with st.expander('1 Month', expanded=False):
            st.write('Data dimensions:')
            st.write(f1m.shape)
            st.write('1 Month Dataframe')
            st.write(f1m.head())
    prompt.success('Forecasting Completed!')

    st.subheader('Forecast plots')
    with st.spinner('Plotting Forecast...'):
        prompt = st.empty()
        with st.expander('7 Day', expanded=False):
           TimeSeriesUtils.plot_forecasts(f7d)
        with st.expander('15 Day', expanded=False):
           TimeSeriesUtils.plot_forecasts(f15d)
        with st.expander('1 Month', expanded=False):
            TimeSeriesUtils.plot_forecasts(f1m)
    prompt.success('Plotting Completed!')
