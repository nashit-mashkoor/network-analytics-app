import streamlit as st
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns # plotting library
import tensorflow as tf

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from pandas import DataFrame,Series
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn import svm
from sklearn.svm import SVR

from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from keras import utils as np_utils
from tensorflow.keras import layers
from . import PredictionUtils

def eCell_Accessibility_process_render(dataset_name, model_name):
    st.subheader('Exploratory data analysis')
    with st.spinner('Loading Data...'):        
        prompt = st.empty()  
        with st.expander('Data Summary', expanded=False):
           #Importing the given Data Set
            data = pd.read_csv('data/eCell_Accessibility_data.csv')
            st.write('Data sample')
            st.write(data.head())

            st.write('Data types')
            st.write(data.dtypes.astype(str))

            st.write('Number of Null Values')
            st.write(data.isnull().sum())

            st.write('Data Shape')
            st.write(data.shape)

            st.write('Data size')
            st.write(data.size)

            st.write('Data Description')
            st.write(data.describe())

        with st.expander('Data Plots', expanded=False):
            st.write('RRC Setup Param1 distribition boxplot')
            plt.subplots(figsize=(8,8))
            sns.boxplot(x = data['RRC Setup Param1'])
            st.pyplot(plt.gcf())
            plt.clf()
            st.write('===============================================')

            st.write('Accessibility Count')
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 5)
            sns.histplot(data['Accessibility'],ax=ax)
            st.pyplot(fig)
            st.write('===============================================')

            st.write('Added ERAB Establishment Param1')
            plt.subplots(figsize=(15,6))
            plt.plot(data['Added ERAB Establishment Param1'])
            st.pyplot(plt.gcf())
            plt.clf()
            st.write('===============================================')

            st.write('RRC success rate')
            RRC_Rate = data[['eCell_Id','RRC Setup Param2','RRC Setup Param1']]
            RRC_Rate['RRC Success Rate'] = RRC_Rate['RRC Setup Param1']/RRC_Rate['RRC Setup Param2']*100
            RRC_df= RRC_Rate[['eCell_Id','RRC Success Rate','RRC Setup Param2']]
            ax=RRC_df.set_index('eCell_Id').sort_values('RRC Setup Param2', ascending=False).head(10)
            st.write(ax)
            ax.plot(kind='bar')
            st.pyplot(plt.gcf())
            plt.clf()
            st.write('===============================================')

            st.write('Initial ERAB Establishment success rate')
            Initial_ERAB_Establishment_Rate = data[['eCell_Id','Initial ERAB Establishment Param2','Initial ERAB Establishment Param1']]
            Initial_ERAB_Establishment_Rate['Initial ERAB Establishment Success Rate'] = Initial_ERAB_Establishment_Rate['Initial ERAB Establishment Param1']/Initial_ERAB_Establishment_Rate['Initial ERAB Establishment Param2']*100
            Initial_ERAB_Establishment_df= Initial_ERAB_Establishment_Rate[['eCell_Id','Initial ERAB Establishment Success Rate','Initial ERAB Establishment Param2']]
            ax=Initial_ERAB_Establishment_df.set_index('eCell_Id').sort_values('Initial ERAB Establishment Param2', ascending=False).head(10)
            st.write(ax)
            ax.plot(kind='barh')
            st.pyplot(plt.gcf())
            plt.clf()
            st.write('===============================================')

            st.write('Added ERAB Establishment success rate')
            Added_ERAB_Establishment_Attempt_Rate = data[['eCell_Id','Added ERAB Establishment Param2','Added ERAB Establishment Param1']]
            Added_ERAB_Establishment_Attempt_Rate['Added_ERAB_Establishment_Success_Rate'] = Added_ERAB_Establishment_Attempt_Rate['Added ERAB Establishment Param1']/Added_ERAB_Establishment_Attempt_Rate['Added ERAB Establishment Param2']*100
            Added_ERAB_Establishment_Attempt_df= Added_ERAB_Establishment_Attempt_Rate[['eCell_Id','Added ERAB Establishment Param2','Added_ERAB_Establishment_Success_Rate']]
            ax=Added_ERAB_Establishment_Attempt_df.set_index('eCell_Id').sort_values('Added ERAB Establishment Param2', ascending=False).head(10)
            st.write(ax)
            ax.plot(kind='barh')
            st.pyplot(plt.gcf())
            plt.clf()
            st.write('===============================================')

            st.write('Joint Distributions of independent variables')
            sns.pairplot(data[['RRC Setup Param1','RRC Setup Param2','Initial ERAB Establishment Param1','Initial ERAB Establishment Param2','Added ERAB Establishment Param1','Added ERAB Establishment Param2','S1 Setup Param1','S1 Setup Param2']], diag_kind="kde")
            st.pyplot(plt.gcf())
            plt.clf()
            st.write('===============================================')

            st.write('Corelation of feature variables')
            corr = data.select_dtypes(include = ['float64','int64']).corr()
            plt.figure(figsize=(10, 10))
            sns.heatmap(corr, vmax=1, square=True, annot = True)
            st.pyplot(plt.gcf())
            plt.clf()

        with st.expander('Data Preparation', expanded=False):
            #seperating independent and dependent variables
            x = data.drop(['Accessibility','eCell_Id'], axis=1)
            y = data['Accessibility']
            st.write('Data Shape')
            st.write(x.shape, y.shape)
    prompt.success('Data Loaded!')

    st.subheader('Model Performance Metrics')
    with st.spinner('Training Model...'):        
        prompt = st.empty()

        if model_name == 'Linear_Regression':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                sc = StandardScaler()
                x_train_std = sc.fit_transform(x_train)
                x_test_std = sc.fit_transform(x_test)
                st.write('Scaling applied: Standard Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            with st.expander('Model Performance', expanded=False):
                lin_reg = LinearRegression() 
                lin_reg.fit(x_train_std,y_train) 
                #Prediction using test set
                y_pred = lin_reg.predict(x_test_std) 
                mae=metrics.mean_absolute_error(y_test, y_pred) 
                mse=metrics.mean_squared_error(y_test, y_pred)
                rmse=np.sqrt(mse)
                # Printing the metrics 
                st.write('R2 square:',metrics.r2_score(y_test, y_pred)) 
                st.write('MAE: ', mae) 
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)
                # Prediction frame
                x_test= pd.DataFrame(x_test)
                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Accessibility']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)
                
        elif model_name == 'Decision_Tree_Regression':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                sc = StandardScaler()
                x_train_std = sc.fit_transform(x_train)
                x_test_std = sc.fit_transform(x_test)
                st.write('Scaling applied: Standard Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                param_grid = {'max_features': ['auto', 'sqrt'],
                            'max_depth': np.arange(5, 36, 5),
                            'min_samples_split': [5, 10, 20, 40],
                            'min_samples_leaf': [2, 6, 12, 24],
                            }
                tree_reg = RandomizedSearchCV(estimator = DecisionTreeRegressor(),param_distributions = param_grid, n_iter = 100, verbose = 2, n_jobs = -1)
                tree_reg.fit(x_train_std, y_train)
                y_pred = tree_reg.predict(x_test_std) 
                mae=metrics.mean_absolute_error(y_test, y_pred) 
                mse=metrics.mean_squared_error(y_test, y_pred)
                rmse=np.sqrt(mse)
                st.write('R2 square:',metrics.r2_score(y_test, y_pred)) 
                st.write('MAE: ', mae) 
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)
                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Accessibility']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

        elif model_name =='Gradient_Boosting_Regression':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                sc = StandardScaler()
                x_train_std = sc.fit_transform(x_train)
                x_test_std = sc.fit_transform(x_test)
                st.write('Scaling applied: Standard Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                param_grid = {"learning_rate"    : [0.01, 0.1, 0.3],
                            "subsample"        : [0.5, 1.0],
                            "max_depth"        : [3, 4, 5, 10, 15, 20],
                            "max_features"     : ['auto', 'sqrt'],
                            "min_samples_split": [5, 10, 20, 40],
                            "min_samples_leaf" : [2, 6, 12, 24]
                            }
                grad_reg = RandomizedSearchCV(GradientBoostingRegressor(), param_distributions = param_grid, n_iter = 100, verbose = 2, n_jobs = -1)
                grad_reg.fit(x_train_std, y_train)
                
                y_pred = grad_reg.predict(x_test_std)
                mae=metrics.mean_absolute_error(y_test, y_pred)
                mse=metrics.mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                # Printing the metrics
                st.write('R2 square:',metrics.r2_score(y_test, y_pred))
                st.write('MAE: ', mae)
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)

                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Accessibility']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

        elif model_name =='AdaBoost_Regression':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                sc = StandardScaler()
                x_train_std = sc.fit_transform(x_train)
                x_test_std = sc.fit_transform(x_test)
                st.write('Scaling applied: Standard Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                param_grid = {"learning_rate" : [0.01, 0.1, 0.3],
              "loss"          : ['linear', 'square', 'exponential']
             }
                ada_reg = RandomizedSearchCV(AdaBoostRegressor( DecisionTreeRegressor(), n_estimators=100), param_distributions = param_grid, n_iter = 100, verbose = 2, n_jobs = -1)
                ada_reg.fit(x_train_std, y_train)
                y_pred = ada_reg.predict(x_test_std)
                mae=metrics.mean_absolute_error(y_test, y_pred)
                mse=metrics.mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                # Printing the metri
                st.write('R2 square:',metrics.r2_score(y_test, y_pred))
                st.write('MAE: ', mae)
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)

                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Accessibility']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

        elif model_name =='Support_Vector_Machine':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                sc = StandardScaler()
                x_train_std = sc.fit_transform(x_train)
                x_test_std = sc.fit_transform(x_test)
                st.write('Scaling applied: Standard Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                regressor= SVR(kernel='rbf')
                regressor.fit(x_train_std,y_train)
                y_pred_svm=regressor.predict(x_test_std)
                #y_pred_svm = cross_val_predict(regressor, x, y)
                mae=metrics.mean_absolute_error(y_test, y_pred_svm)
                mse=metrics.mean_squared_error(y_test, y_pred_svm)
                rmse= np.sqrt(mse)
                # Printing the metrics
                st.write('R2 square:',metrics.r2_score(y_test, y_pred_svm))
                st.write('MAE: ', mae)
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)
                y_pred= pd.DataFrame(y_pred_svm)
                y_pred.columns= ['Pred_Accessibility']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

        elif model_name =='Regression_Using_Neural_Networks':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                sc = StandardScaler()
                x_train_std = sc.fit_transform(x_train)
                x_test_std = sc.fit_transform(x_test)
                st.write('Scaling applied: Standard Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                hidden_units1 = 60
                hidden_units2 = 40
                hidden_units3 = 20
                learning_rate = 0.01
                model = Sequential([
                                    Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
                                    Dropout(0.2),
                                    Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
                                    Dropout(0.2),
                                    Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
                                    Dense(1, kernel_initializer='normal', activation='linear')
                                ])
                # loss function
                mse = MeanSquaredError()
                rmse = tf.keras.metrics.RootMeanSquaredError()
                model.compile(
                    loss=mse,
                    optimizer=Adam(learning_rate=learning_rate), 
                    metrics=[mse,rmse]
                )
                # train the model
                history = model.fit(
                    x_train_std, 
                    y_train, 
                    epochs=20, 
                    batch_size=64,
                    validation_split=0.2
                )
                st.write('Model training Error')
                PredictionUtils.nn_plot_history(history, 'root_mean_squared_error')
                y_pred= model.predict(x_test_std).tolist()
                mse = metrics.mean_squared_error(y_test, y_pred, squared=False)
                rmse = np.sqrt(mse)
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)
                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Accessibility']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

        elif model_name == 'SGD_Neural_Network':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                sc = StandardScaler()
                x_train_std = sc.fit_transform(x_train)
                x_test_std = sc.fit_transform(x_test)
                st.write('Scaling applied: Standard Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                input_size=8
                output_size=1
                models = tf.keras.Sequential([
                                            tf.keras.layers.Dense(output_size)
                                            ])
                custom_optimizer=tf.keras.optimizers.SGD(learning_rate=0.02)
                models.compile(optimizer=custom_optimizer,loss='mean_squared_error')
                models.fit(x_train_std,y_train,epochs=10,verbose=1)
                y_pred= models.predict(x_test_std)
                mse = metrics.mean_squared_error(y_test, y_pred, squared=False)
                rmse = np.sqrt(mse)
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)
                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Retainability']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

    prompt.success('Model trained!')

    st.subheader('Model Prediction Dataframes')
    with st.spinner('Predicting dataframes...'):        
        prompt = st.empty()
        with st.expander('Predicted Dataframe', expanded=False):
            st.write(pred_df)    
    prompt.success('Dataframes created!')

def eCell_Retainability_process_render(dataset_name, model_name):
    st.subheader('Exploratory data analysis')
    with st.spinner('Loading Data...'):        
        prompt = st.empty()  
        with st.expander('Data Summary', expanded=False):
           #Importing the given Data Set
            data = pd.read_csv('data/eCell_Retainability_data.csv')
            st.write('Data sample')
            st.write(data.head())

            st.write('Data types')
            st.write(data.dtypes.astype(str))

            st.write('Number of Null Values before imputation')
            st.write(data.isnull().sum())
            data['ErabRelAbnormalEnb_Param2'].fillna(data['ErabRelAbnormalEnb_Param2'].mean(), inplace=True)
            data['ErabRelAbnormalMme_Param2'].fillna(data['ErabRelAbnormalMme_Param2'].mean(), inplace=True)
            data['ErabRelMme_Param1'].fillna(data['ErabRelMme_Param1'].mean(), inplace=True)
            data['ErabRelMme_Param2'].fillna(data['ErabRelMme_Param2'].mean(), inplace=True)
            data['ErabRelNormalEnb_Param1'].fillna(data['ErabRelNormalEnb_Param1'].mean(), inplace=True)
            data['ErabRelNormalEnb_Param2'].fillna(data['ErabRelNormalEnb_Param2'].mean(), inplace=True)
            st.write('Number of Null Values after imputation')
            st.write(data.isnull().sum())

            st.write('Data Shape')
            st.write(data.shape)

            st.write('Data size')
            st.write(data.size)

            st.write('Data Description')
            st.write(data.describe())

        with st.expander('Data Plots', expanded=False):
            
            st.write('Joint Distributions of independent variables')
            sns.pairplot(data[['ErabRelAbnormalEnb_Param1','ErabRelAbnormalEnb_Param2','ErabRelAbnormalMme_Param1','ErabRelAbnormalMme_Param2','ErabRelMme_Param1','ErabRelMme_Param2','ErabRelNormalEnb_Param1','ErabRelNormalEnb_Param2']], diag_kind="kde")
            st.pyplot(plt.gcf())
            plt.clf()
            st.write('===============================================')

            st.write('Corelation of feature variables')
            corr = data.select_dtypes(include = ['float64','int64']).corr()
            plt.figure(figsize=(10, 10))
            sns.heatmap(corr, vmax=1, square=True, annot = True)
            st.pyplot(plt.gcf())
            plt.clf()
            st.write('===============================================')

            st.write('ErabRelAbnormalEnb Failure rate')
            ErabRelAbnormalEnb_rate= data[['eCell_Id','ErabRelAbnormalEnb_Param1','ErabRelAbnormalEnb_Param2']]
            ErabRelAbnormalEnb_rate['ErabRelAbnormalEnb Failure Rate']= ErabRelAbnormalEnb_rate['ErabRelAbnormalEnb_Param2']/ErabRelAbnormalEnb_rate['ErabRelAbnormalEnb_Param1']*100
            ErabRelAbnormalEnb_df = ErabRelAbnormalEnb_rate[['eCell_Id','ErabRelAbnormalEnb Failure Rate']]
            ax = ErabRelAbnormalEnb_df.set_index('eCell_Id').head(10)
            st.write(ax)
            ax.plot(kind='bar')
            st.pyplot(plt.gcf())
            plt.clf()
            st.write('===============================================')

            st.write('ErabRelAbnormalMme Failure rate')
            ErabRelAbnormalMme_rate= data[['eCell_Id','ErabRelAbnormalMme_Param1','ErabRelAbnormalMme_Param2']]
            ErabRelAbnormalMme_rate['ErabRelAbnormalMme Failure Rate']= ErabRelAbnormalMme_rate['ErabRelAbnormalMme_Param2']/ErabRelAbnormalMme_rate['ErabRelAbnormalMme_Param1']*100
            ErabRelAbnormalMme_df = ErabRelAbnormalMme_rate[['eCell_Id','ErabRelAbnormalMme Failure Rate']]
            ax = ErabRelAbnormalMme_df.set_index('eCell_Id').head(10)
            st.write(ax)
            ax.plot(kind='bar')
            st.pyplot(plt.gcf())
            plt.clf()
            st.write('===============================================')

            st.write('ErabRelNormalEnb Failure rate')
            ErabRelNormalEnb_rate= data[['eCell_Id','ErabRelNormalEnb_Param1','ErabRelNormalEnb_Param2']]
            ErabRelNormalEnb_rate['ErabRelNormalEnb Failure Rate']= ErabRelNormalEnb_rate['ErabRelNormalEnb_Param2']/ErabRelNormalEnb_rate['ErabRelNormalEnb_Param1']*100
            ErabRelNormalEnb_df = ErabRelNormalEnb_rate[['eCell_Id','ErabRelNormalEnb Failure Rate']]
            ax = ErabRelNormalEnb_df.set_index('eCell_Id').head(10)
            st.write(ax)
            ax.plot(kind='bar')
            st.pyplot(plt.gcf())
            plt.clf()
        with st.expander('Data Preparation', expanded=False):
            #seperating independent and dependent variables
            x = data.drop(['Retainability','eCell_Id'], axis=1)
            y = data['Retainability']
            st.write('Data Shape')
            st.write(x.shape, y.shape)
    prompt.success('Data Loaded!')

    st.subheader('Model Performance Metrics')
    with st.spinner('Training Model...'):        
        prompt = st.empty()
        if model_name == 'Linear_Regression':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                scaler = MinMaxScaler()
                x_train_std = scaler.fit_transform(x_train)
                x_test_std = scaler.fit_transform(x_test)
                st.write('Scaling applied: Min Max Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            with st.expander('Model Performance', expanded=False):
                lin_reg = LinearRegression() 
                lin_reg.fit(x_train_std,y_train) 
                #Prediction using test set
                y_pred = lin_reg.predict(x_test_std) 
                mae=metrics.mean_absolute_error(y_test, y_pred) 
                mse=metrics.mean_squared_error(y_test, y_pred)
                rmse=np.sqrt(mse)
                # Printing the metrics 
                st.write('R2 square:',metrics.r2_score(y_test, y_pred)) 
                st.write('MAE: ', mae) 
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)
                # Prediction frame
                x_test= pd.DataFrame(x_test)
                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Retainability']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)
                
        elif model_name == 'Decision_Tree_Regression':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                scaler = MinMaxScaler()
                x_train_std = scaler.fit_transform(x_train)
                x_test_std = scaler.fit_transform(x_test)
                st.write('Scaling applied: MinMax Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                param_grid = {'max_features': ['auto', 'sqrt'],
                            'max_depth': np.arange(5, 36, 5),
                            'min_samples_split': [5, 10, 20, 40],
                            'min_samples_leaf': [2, 6, 12, 24],
                            }
                tree_reg = RandomizedSearchCV(estimator = DecisionTreeRegressor(),param_distributions = param_grid, n_iter = 100, verbose = 2, n_jobs = -1)
                tree_reg.fit(x_train_std, y_train)
                y_pred = tree_reg.predict(x_test_std) 
                mae=metrics.mean_absolute_error(y_test, y_pred) 
                mse=metrics.mean_squared_error(y_test, y_pred)
                rmse=np.sqrt(mse)
                st.write('R2 square:',metrics.r2_score(y_test, y_pred)) 
                st.write('MAE: ', mae) 
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)
                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Retainability']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

        elif model_name =='Gradient_Boosting_Regression':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                scaler = MinMaxScaler()
                x_train_std = scaler.fit_transform(x_train)
                x_test_std = scaler.fit_transform(x_test)
                st.write('Scaling applied: MinMax Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                param_grid = {"learning_rate"    : [0.01, 0.1, 0.3],
                            "subsample"        : [0.5, 1.0],
                            "max_depth"        : [3, 4, 5, 10, 15, 20],
                            "max_features"     : ['auto', 'sqrt'],
                            "min_samples_split": [5, 10, 20, 40],
                            "min_samples_leaf" : [2, 6, 12, 24]
                            }
                grad_reg = RandomizedSearchCV(GradientBoostingRegressor(), param_distributions = param_grid, n_iter = 100, verbose = 2, n_jobs = -1)
                grad_reg.fit(x_train_std, y_train)
                
                y_pred = grad_reg.predict(x_test_std)
                mae=metrics.mean_absolute_error(y_test, y_pred)
                mse=metrics.mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                # Printing the metrics
                st.write('R2 square:',metrics.r2_score(y_test, y_pred))
                st.write('MAE: ', mae)
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)

                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Retainability']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

        elif model_name =='AdaBoost_Regression':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                scaler = MinMaxScaler()
                x_train_std = scaler.fit_transform(x_train)
                x_test_std = scaler.fit_transform(x_test)
                st.write('Scaling applied: MinMax Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                param_grid = {"learning_rate" : [0.01, 0.1, 0.3],
              "loss"          : ['linear', 'square', 'exponential']
             }
                ada_reg = RandomizedSearchCV(AdaBoostRegressor( DecisionTreeRegressor(), n_estimators=100), param_distributions = param_grid, n_iter = 100, verbose = 2, n_jobs = -1)
                ada_reg.fit(x_train_std, y_train)
                y_pred = ada_reg.predict(x_test_std)
                mae=metrics.mean_absolute_error(y_test, y_pred)
                mse=metrics.mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                # Printing the metri
                st.write('R2 square:',metrics.r2_score(y_test, y_pred))
                st.write('MAE: ', mae)
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)

                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Retainability']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

        elif model_name =='Support_Vector_Machine':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                scaler = MinMaxScaler()
                x_train_std = scaler.fit_transform(x_train)
                x_test_std = scaler.fit_transform(x_test)
                st.write('Scaling applied: MinMax Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                regressor= SVR(kernel='rbf')
                regressor.fit(x_train_std,y_train)
                y_pred_svm=regressor.predict(x_test_std)
                #y_pred_svm = cross_val_predict(regressor, x, y)
                mae=metrics.mean_absolute_error(y_test, y_pred_svm)
                mse=metrics.mean_squared_error(y_test, y_pred_svm)
                rmse= np.sqrt(mse)
                # Printing the metrics
                st.write('R2 square:',metrics.r2_score(y_test, y_pred_svm))
                st.write('MAE: ', mae)
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)
                y_pred= pd.DataFrame(y_pred_svm)
                y_pred.columns= ['Pred_Retainability']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

        elif model_name =='Regression_Using_Neural_Networks':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                scaler = MinMaxScaler()
                x_train_std = scaler.fit_transform(x_train)
                x_test_std = scaler.fit_transform(x_test)
                st.write('Scaling applied: MinMax Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                hidden_units1 = 60
                hidden_units2 = 40
                hidden_units3 = 20
                learning_rate = 0.01
                model = Sequential([
                                    Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
                                    Dropout(0.2),
                                    Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
                                    Dropout(0.2),
                                    Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
                                    Dense(1, kernel_initializer='normal', activation='linear')
                                ])
                # loss function
                mse = MeanSquaredError()
                rmse = tf.keras.metrics.RootMeanSquaredError()
                model.compile(
                    loss=mse,
                    optimizer=Adam(learning_rate=learning_rate), 
                    metrics=[mse,rmse]
                )
                # train the model
                history = model.fit(
                    x_train_std, 
                    y_train, 
                    epochs=30, 
                    batch_size=64,
                    validation_split=0.2
                )
                st.write('Model training Error')
                PredictionUtils.nn_plot_history(history, 'root_mean_squared_error')
                y_pred= model.predict(x_test_std).tolist()
                mse = metrics.mean_squared_error(y_test, y_pred, squared=False)
                rmse = np.sqrt(mse)
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)
                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Retainability']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

        elif model_name == 'SGD_Neural_Network':
            with st.expander('Training & Test Split', expanded=False):
                x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 56,test_size=0.3)
                scaler = MinMaxScaler()
                x_train_std = scaler.fit_transform(x_train)
                x_test_std = scaler.fit_transform(x_test)
                st.write('Scaling applied: MinMax Scaler')
                st.write('Split percentage: {}'.format(0.3))
                st.write('Train Y labels')
                st.write(y_train)
            
            with st.expander('Model Performance', expanded=False):
                input_size=8
                output_size=1
                models = tf.keras.Sequential([
                                            tf.keras.layers.Dense(output_size)
                                            ])
                custom_optimizer=tf.keras.optimizers.SGD(learning_rate=0.02)
                models.compile(optimizer=custom_optimizer,loss='mean_squared_error')
                models.fit(x_train_std,y_train,epochs=10,verbose=1)
                y_pred= models.predict(x_test_std)
                mse = metrics.mean_squared_error(y_test, y_pred, squared=False)
                rmse = np.sqrt(mse)
                st.write('MSE: ', mse)
                st.write('RMSE: ', rmse)
                y_pred= pd.DataFrame(y_pred)
                y_pred.columns= ['Pred_Retainability']
                pred_df = pd.concat([x_test.reset_index(drop='True'),y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)

    prompt.success('Model trained!')

    st.subheader('Model Prediction Dataframes')
    with st.spinner('Predicting dataframes...'):        
        prompt = st.empty()
        with st.expander('Predicted Dataframe', expanded=False):
            st.write(pred_df)    
    prompt.success('Dataframes created!')