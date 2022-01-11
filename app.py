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

from src.TimeSeries import msc_traffic_process_render, apn_utilisation_process_render
from src.Prediction import eCell_Accessibility_process_render, eCell_Retainability_process_render
from src.Anomaly import jitter_15min_15days_30nodes_process_render, total_traffic_rate_5min_7days_20nodes

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
selected_usecase = st.sidebar.radio('Select the usecase to explore', ('Time_Series_Forecasting','Prediction', 'Anomaly_Detection'), index=0)
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
                            ('Isolation_Forest', 'Autoencoder', 'Local_Outlier_Factor', 'One_Class_SVM', 'DBSCAN'))
    compare = st.sidebar.checkbox('Mark to compare with others')
process = st.sidebar.button('Analyze')

with st.container():
    st.header(f'{selected_usecase} Based Analysis:')

prompt = st.empty()
prompt.error('Configure dataset and model to start analysing!')

# Creating main skeleton of each page
if selected_usecase == 'Time_Series_Forecasting':
    if process:
        prompt.empty()
        if dataset_name == 'MSC_Traffic':
            msc_traffic_process_render(dataset_name, model_name)
        elif dataset_name == 'APN_Utilization':
            apn_utilisation_process_render(dataset_name, model_name)

elif selected_usecase == 'Prediction':
    if process:
        prompt.empty()
        if dataset_name == 'eCell_Accessibility':
            eCell_Accessibility_process_render(dataset_name, model_name)
        elif dataset_name == 'eCell_Retainability':
            eCell_Retainability_process_render(dataset_name, model_name)

elif selected_usecase == 'Anomaly_Detection':
    if process:
        prompt.empty()
        if dataset_name == 'IP_Link_Jitter':
            jitter_15min_15days_30nodes_process_render(dataset_name, model_name, compare)
        elif dataset_name == 'IP_Router_Port_Total_Traffic_Rate':
            total_traffic_rate_5min_7days_20nodes(dataset_name, model_name, compare)

else:
    st.error('Configure dataset and model to start analysing!')