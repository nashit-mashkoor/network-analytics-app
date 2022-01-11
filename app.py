import streamlit as st
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")

from tqdm import notebook
from prettytable import PrettyTable 
from astropy.table import Table, Column
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from src.AnomalyDataLoader import AnomalyDataLoader
from src.AnomalyModel import AnomalyModel
from src.TimeSeries import msc_traffic_process_render, apn_utilisation_process_render
from src.Prediction import eCell_Accessibility_process_render, eCell_Retainability_process_render

# Helper function
def get_model(selected_usecase, model_name):
    if selected_usecase == 'Time_Series_Forecasting':
        return None
    elif selected_usecase == 'Prediction':
        return None
    elif selected_usecase ==  'Anomaly_Detection':
        return AnomalyModel()
    

@st.cache()
def get_dataset(selected_usecase, dataset_name):
    if selected_usecase == 'Time_Series_Forecasting':
        return None
    elif selected_usecase == 'Prediction':
        return None
    elif selected_usecase ==  'Anomaly_Detection':
        return AnomalyDataLoader(dataset_name)

# App setting
st.set_page_config(
    page_title="Network Analytics App", layout="wide", initial_sidebar_state="collapsed",
    page_icon='🕸'
)
HIDE_STREAMLIT_STYLE = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)
with st.container():
    st.title('Network Analytics App')
    TITLE_ALIGNMENT="""
                    <style>
                    #network-analytics-app {
                    text-align: center
                    }
                    </style>
                    """
    st.markdown(TITLE_ALIGNMENT, unsafe_allow_html=True)

# Setting Sidebar for configuration
st.sidebar.header('Configuration for Network Analysis')
selected_usecase = st.sidebar.radio('Select the usecase to explore', ('Time_Series_Forecasting','Prediction', 'Anomaly_Detection'), index=1)
if selected_usecase == 'Time_Series_Forecasting':
    dataset_name = st.sidebar.selectbox('Select the dataset to analyse',
                                ('MSC_Traffic', 'APN_Utilization'))
    if  dataset_name == 'MSC_Traffic':       
        model_name = st.sidebar.selectbox('Select the model to analyse',
                                    ('HWES', 'SARIMA', 'XGBoost', 'Prophet', 'LSTM'))
    elif dataset_name == 'APN_Utilization': 
        model_name = st.sidebar.selectbox('Select the model to analyse',
                                    ('LSTM_GRU', 'NBEATS', 'CNN_Wavenets'))
elif selected_usecase == 'Prediction':
    dataset_name = st.sidebar.selectbox('Select the dataset to analyse',
                                ('eCell_Accessibility', 'eCell_Retainability'))
    model_name = st.sidebar.selectbox('Select the model to analyse',
                            ('Linear_Regression', 'Decision_Tree_Regression', 'Gradient_Boosting_Regression', 'AdaBoost_Regression', 'Support_Vector_Machine',
                            'Regression_Using_Neural_Networks', 'SGD_Neural_Network'))    
elif selected_usecase == 'Anomaly_Detection':
    dataset_name = st.sidebar.selectbox('Select the dataset to analyse',
                                ('IP_Link_Jitter', 'IP_Router_Port_Total_Traffic_Rate'))
    model_name = st.sidebar.selectbox('Select the model to analyse',
                            ('Isolation_Forest', 'Local_Outlier_Factor', 'One_Class_SVM', 'DBSCAN', 'Autoencoder'))
    compare = st.sidebar.checkbox('Mark to compare with others')
process = st.sidebar.button('Analyze')

with st.container():
    st.header(f'{selected_usecase} Based Analysis:')

# Loading Model and Dataset classes
model = get_model(selected_usecase, model_name)
dataset = get_dataset(selected_usecase, dataset_name)

# Creating main skeleton of each page
if selected_usecase == 'Time_Series_Forecasting':
    if process:
        if dataset_name == 'MSC_Traffic':
            msc_traffic_process_render(dataset_name, model_name)
        elif dataset_name == 'APN_Utilization':
            apn_utilisation_process_render(dataset_name, model_name)

elif selected_usecase == 'Prediction':
    if process:
        if dataset_name == 'eCell_Accessibility':
            eCell_Accessibility_process_render(dataset_name, model_name)
        elif dataset_name == 'eCell_Retainability':
            eCell_Retainability_process_render(dataset_name, model_name)

elif selected_usecase == 'Anomaly_Detection':
    st.subheader('Exploratory data analysis')
    with st.spinner('Loading Data...'):        
        prompt = st.empty() 
        with st.expander('Raw Sample Data', expanded=False):
            anomaly_df = dataset.get_raw_df()
            st.write(anomaly_df)

        with st.expander('Pre-Processed Data', expanded=False):
            anomaly_df = dataset.get_preprocessed_df(anomaly_df)
            st.write(anomaly_df)

            # Fill Null values
            anomaly_df = dataset.fill_null_values(anomaly_df)

            # Get helper day df
            anomaly_df_day = dataset.get_df_day(anomaly_df)
        with st.expander('Data Visualisation',expanded=False):
            dataset.visualise_data(anomaly_df, anomaly_df_day)
            st.write(anomaly_df_day[0])

        with st.expander('Individual instances',expanded=False):
            dataset.visualise_all_instances(anomaly_df_day)

        with st.expander('Boxplot visualisation',expanded=False):
            dataset.visualise_boxplot(anomaly_df_day)
    prompt.success('Data Loaded!')

    if process:
        st.subheader('Training dataframes and results')
        with st.spinner('Training Model...'):        
            prompt = st.empty() 
            with st.expander('Outlier detection results', expanded=False):
                X = anomaly_df_day.T
                outlier = model.train_predict(model_name, X, True, dataset.get_output_labels())

            with st.expander('All instances with Anomlies', expanded=False):
                dataset.visualise_all_instances_with_anomaly(X, anomaly_df_day, outlier)

        prompt.success('Model Loaded!')

        st.subheader('Testing and Evaluation')
        with st.spinner('Test Model...'):        
            prompt = st.empty() 
            with st.expander('Test Data', expanded=False):
                X_test = dataset.get_test_day(X)
                st.write(X_test)

            with st.expander('Test Results', expanded=False):
                model.test(dataset_name, X_test, outlier, verbose=True)
        prompt.success('Testing Complete!')
        if compare:
            st.subheader('Testing and Evaluation')
            with st.spinner('Comparing...'):
                prompt = st.empty()
                with st.expander('Compare Results', expanded=False):
                    model.compare_all(dataset_name, X, X_test)
            prompt.success('Comparison Complete!')

else:
    st.error('Configure dataset and model to start analysing!')