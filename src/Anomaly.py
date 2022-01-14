import warnings
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
from tqdm import notebook
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")
from prettytable import PrettyTable 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from astropy.table import Table, Column
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
from . import AnomalyUtils

def jitter_15min_15days_30nodes_process_render(dataset_name, model_name, compare):
    st.subheader('Exploratory data analysis')
    with st.spinner('Loading Data...'):        
        prompt = st.empty() 
        with st.expander('Raw Sample Data', expanded=False):
            sample_data = pd.read_csv("data/Jitter_15min_15days_30nodes.csv")
            st.write(sample_data)

        with st.expander('Pre-Processed Data', expanded=False):
            sample_data = sample_data.T[1:]
            sample_data = sample_data.reset_index()
            sample_data = sample_data.rename({'index': 'date'}, axis=1)
            sample_data['date'] = pd.to_datetime(sample_data['date'], dayfirst = True, errors='coerce')
            st.write(sample_data)
            for i in range(len(sample_data.columns)-1):
                sample_data[i] = sample_data[i].groupby(sample_data.date.dt.hour).transform(lambda x: x.fillna(x.median()))
        with st.expander('Data Visualisation',expanded=False):
            df_day = pd.DataFrame()
            for i in range(len(sample_data.columns)-1):
                df_day[i] = sample_data.groupby(sample_data.date)[i].sum()
            df_day.index = pd.DatetimeIndex(df_day.index.values, freq=df_day.index.inferred_freq)
            AnomalyUtils.ip_link_visualise_all(df_day)
        
        with st.expander('Individual instances',expanded=False):
            AnomalyUtils.ip_link_visualise_all_instances(df_day)

        with st.expander('Boxplot visualisation',expanded=False):
            AnomalyUtils.ip_link_visualise_all_boxplot(df_day)
    prompt.success('Data Loaded!')

    st.subheader('Training dataframes and results')
    with st.spinner('Training Model...'):        
        prompt = st.empty() 
        with st.expander('Outlier detection results', expanded=False):
            X = df_day.T
            # ###############################################
            # Identify outliers in the training dataset
            Isolation_Forest = IsolationForest()
            if model_name == 'Isolation_Forest':
                st.write("Outliers Detected by Isolation Forest:")
                st.write("======================================")
            outlier1 = pd.DataFrame()
            list_1 = []
            for i in range(len(X)):
                a = X.loc[i].to_numpy().reshape((len(X.loc[i]), 1))
                b = Isolation_Forest.fit_predict(a)
                c = pd.DataFrame(b)
                if model_name == 'Isolation_Forest':
                    st.write("Outliers Detected in IP_Link_"+str(i+1)+":", c[c == -1].count().sum())
                    st.write("====================================")
                outlier1 = outlier1.append(c.T)
                list_1.append(c[c == -1].count().sum())

            if model_name == 'Isolation_Forest':
                st.write("Total Outliers Detected: ", outlier1[outlier1 == -1].count().sum())
                st.write("=================================")

                x = AnomalyUtils.ip_link_get_labels()
                y = list_1

                fig = px.bar(x=x, y=y)
                st.plotly_chart(fig)

            outlier1 = outlier1.reset_index(drop=True)
            # ######################################################

            # ######################################################
            model = AnomalyUtils.AutoEncoder(output_units = 1)
            if model_name == 'Autoencoder':
                st.write("Outliers Detected by Autoencoder:")
                st.write("======================================")
            outlier2 = pd.DataFrame()
            list_2 = []
            for i in range(len(X)):
                a = X.loc[i].to_numpy().reshape((len(X.loc[i]), 1))
                threshold = AnomalyUtils.auto_encoder_find_threshold(model, a)
                predictions = AnomalyUtils.auto_encoder_get_predictions(model, a, threshold)
                c = pd.DataFrame(predictions)
                c = c.replace({0: -1}).astype("int")
                if model_name == 'Autoencoder':
                    st.write("Outliers Detected in IP_Link_"+str(i+1)+":", c[c == -1].count().sum())
                    st.write("====================================")
                outlier2 = outlier2.append(c.T)
                list_2.append(c[c == -1].count().sum())

            if model_name == 'Autoencoder':
                st.write("Total Number of Outliers Detected: ", outlier2[outlier2 == -1].count().sum())
                st.write("=================================")

                x = AnomalyUtils.ip_link_get_labels()
                y = list_2

                fig = px.bar(x=x, y=y)
                st.plotly_chart(fig)

            outlier2 = outlier2.reset_index(drop=True)

            # ######################################################

            # ######################################################
            # Identify outliers in the training dataset
            Local_Outlier_Factor = LocalOutlierFactor()
            if model_name == 'Local_Outlier_Factor':
                st.write("Outliers Detected by LocalOutlierFactor:")
                st.write("========================================")
            outlier3 = pd.DataFrame()
            list_3 = []
            for i in range(len(X)):
                a = X.loc[i].to_numpy().reshape((len(X.loc[i]), 1))
                b = Local_Outlier_Factor.fit_predict(a)
                c = pd.DataFrame(b)
                if model_name == 'Local_Outlier_Factor':
                    st.write("Outliers Detected in IP_Link_"+str(i+1)+":", c[c == -1].count().sum())
                    st.write("====================================")
                outlier3 = outlier3.append(c.T)
                list_3.append(c[c == -1].count().sum())
            if model_name == 'Local_Outlier_Factor':
                st.write("Number Outliers Detected: ", outlier3[outlier3 == -1].count().sum())
                st.write("=================================")

                x = AnomalyUtils.ip_link_get_labels()
                y = list_3

                fig = px.bar(x=x, y=y)
                st.plotly_chart(fig)

            outlier3 = outlier3.reset_index(drop=True)
            # ######################################################

            # ######################################################
            # Identify outliers in the training dataset
            One_Class_SVM = OneClassSVM()
            if model_name == 'One_Class_SVM':
                st.write("Outliers Detected by OneClassSVM:")
                st.write("=================================")

            outlier4 = pd.DataFrame()
            list_4 = []
            for i in notebook.tqdm(range(len(X))):
                a = X.loc[i].to_numpy().reshape((len(X.loc[i]), 1))
                b = One_Class_SVM.fit_predict(a)
                c = pd.DataFrame(b)
                if model_name == 'One_Class_SVM':
                    st.write("Outliers Detected in IP_Link_"+str(i+1)+":", c[c == -1].count().sum())
                    st.write("====================================")
                outlier4 = outlier4.append(c.T)
                list_4.append(c[c == -1].count().sum())

            if model_name == 'One_Class_SVM':
                st.write("Number Outliers Detected: ", outlier4[outlier4 == -1].count().sum())
                st.write("=================================")

                x = AnomalyUtils.ip_link_get_labels()
                y = list_4

                fig = px.bar(x=x, y=y)
                st.plotly_chart(fig)

            outlier4 = outlier4.reset_index(drop=True)
            # ######################################################

            # ######################################################
            # Identify outliers in the training dataset
            outlier_detection = DBSCAN()
            if model_name == 'DBSCAN':
                st.write("Outliers Detected by DBSCAN:")
                st.write("============================")
            outlier5 = pd.DataFrame()
            list_5 = []
            for i in notebook.tqdm(range(len(X))):
                a = X.loc[i].to_numpy().reshape((len(X.loc[i]), 1))
                b = outlier_detection.fit_predict(a)
                c = pd.DataFrame(b)
                c = c.replace({0: 1}).astype("int")
                if model_name == 'DBSCAN':
                    st.write("Outliers Detected in IP_Link_"+str(i+1)+":", c[c == -1].count().sum())
                    st.write("====================================")
                outlier5 = outlier5.append(c.T)
                list_5.append(c[c == -1].count().sum())
            if model_name == 'DBSCAN':
                st.write("Number Outliers Detected: ", outlier5[outlier5 == -1].count().sum())
                st.write("=================================")

                x = AnomalyUtils.ip_link_get_labels()
                y = list_5

                fig = px.bar(x=x, y=y)
                st.plotly_chart(fig)

            outlier5 = outlier5.reset_index(drop=True)
            for i in range(len(X)):
                outlier5.loc[i] = np.where(outlier5.loc[i] != -1, 1, outlier5.loc[i])
            # ######################################################

        with st.expander('All instances with Anomlies', expanded=False):
            if model_name == 'Isolation_Forest':
                AnomalyUtils.ip_link_visualise_all_instances_with_anomaly(X, df_day, outlier1)
            elif model_name == 'Autoencoder':
                AnomalyUtils.ip_link_visualise_all_instances_with_anomaly( X, df_day, outlier2)
            elif model_name == 'Local_Outlier_Factor':
                AnomalyUtils.ip_link_visualise_all_instances_with_anomaly( X, df_day, outlier3)   
            elif model_name == 'One_Class_SVM':
                AnomalyUtils.ip_link_visualise_all_instances_with_anomaly( X, df_day, outlier4) 
            elif model_name == 'DBSCAN':
                AnomalyUtils.ip_link_visualise_all_instances_with_anomaly( X, df_day, outlier5) 
    prompt.success('Model Loaded!')

    st.subheader('Testing and Evaluation')
    with st.spinner('Test Model...'):        
        prompt = st.empty() 
        with st.expander('Test Data', expanded=False):
            X_test = X.copy()
            for i in range(len(X_test)):
                condition = np.average(X_test.loc[i]) + 3 * (statistics.stdev(X_test.loc[i]))
                X_test.loc[i] = np.where(X_test.loc[i] > condition, -1, X_test.loc[i])

            for i in range(len(X_test)):
                X_test.loc[i] = np.where(X_test.loc[i] != -1, 1, X_test.loc[i])
                
            X_test = X_test.astype("int")
            st.write(X_test)
        with st.expander('Test Results', expanded=False):
            accuracy_1 = AnomalyUtils.ip_link_test(X_test, outlier1, False)
            accuracy_2 = AnomalyUtils.ip_link_test(X_test, outlier2, False)
            accuracy_3 = AnomalyUtils.ip_link_test(X_test, outlier3, False)
            accuracy_4 = AnomalyUtils.ip_link_test(X_test, outlier4, False)
            accuracy_5 = AnomalyUtils.ip_link_test(X_test, outlier5, False)
            if model_name == 'Isolation_Forest':
                AnomalyUtils.ip_link_test(X_test, outlier1, True)
            elif model_name == 'Autoencoder':
                AnomalyUtils.ip_link_test(X_test, outlier2, True)
            elif model_name == 'Local_Outlier_Factor':
                AnomalyUtils.ip_link_test(X_test, outlier3, True)   
            elif model_name == 'One_Class_SVM':
               AnomalyUtils.ip_link_test(X_test, outlier4, True) 
            elif model_name == 'DBSCAN':
                AnomalyUtils.ip_link_test(X_test, outlier5, True)
    prompt.success('Testing Complete!')
    if compare:
        st.subheader('Comparison Of Models')
        with st.spinner('Comparing...'):
            prompt = st.empty()
            with st.expander('Compare Results', expanded=False):
                for i in range(30):
                    st.write("========================================================")
                    st.write("========================= IP_LINK_" + str(i+1) + " ======================")
                    st.write("========================================================")
                    
                    st.write("Performance with Models")
                    st.write("=======================")
                    x=PrettyTable()
                    x.add_column("Classifier Name",["Isolation Forest","Autoencoder","Local Outlier Factor", "One Class SVM", "DBSCAN"])
                    x.add_column("Accuracy Score",[str(round(accuracy_1[i]*100,2)) + " %", str(round(accuracy_2[i]*100,2)) + " %", str(round(accuracy_3[i]*100,2)) + " %", str(round(accuracy_4[i]*100,2)) + " %", str(round(accuracy_5[i]*100,2)) + " %"])
                    st.write(x)

                    x = ["Isolation Forest","Autoencoder","Local Outlier Factor", "One Class SVM", "DBSCAN"]
                    y = [round(accuracy_1[i],2), round(accuracy_2[i],2), round(accuracy_3[i],2), round(accuracy_4[i],2), round(accuracy_5[i],2)]

                    plt.rcParams["figure.figsize"] = (14,5)
                    fig, ax = plt.subplots() 
                    width = 0.25
                    ind = np.arange(len(y))
                    ax.barh(ind, y, width, color="blue")
                    ax.set_yticks((ind+width/2)-0.1)
                    ax.set_yticklabels(x, minor=False)
                    for i, v in enumerate(y):
                        ax.text(v + 0.01, i , str(v), color='blue', fontweight='bold')
                    plt.title('Comparision')
                    plt.xlabel('Accuracy Score')
                    plt.ylabel('Classifiers')      
                    st.pyplot(fig)
                    plt.clf()
        prompt.success('Comparison Complete!')

def total_traffic_rate_5min_7days_20nodes(dataset_name, model_name, compare):
    st.subheader('Exploratory data analysis')
    with st.spinner('Loading Data...'):        
        prompt = st.empty() 
        with st.expander('Raw Sample Data', expanded=False):
            sample_data = pd.read_csv("data/Total_Traffic_Rate_5min_7days_20nodes.csv")
            st.write(sample_data)

        with st.expander('Pre-Processed Data', expanded=False):
            sample_data = sample_data.T[1:]
            sample_data = sample_data.reset_index()
            sample_data = sample_data.rename({'index': 'date'}, axis=1)
            sample_data['date'] = pd.to_datetime(sample_data['date'], errors='coerce')
            st.write(sample_data)
            for i in range(len(sample_data.columns)-1):
                sample_data[i] = sample_data[i].groupby(sample_data.date.dt.hour).transform(lambda x: x.fillna(x.median()))
        with st.expander('Data Visualisation',expanded=False):
            df_day = pd.DataFrame()
            for i in range(len(sample_data.columns)-1):
                df_day[i] = sample_data.groupby(sample_data.date)[i].sum()
            df_day.index = pd.DatetimeIndex(df_day.index.values, freq=df_day.index.inferred_freq)
            AnomalyUtils.ip_router_visualise_all(df_day)
        
        with st.expander('Individual instances',expanded=False):
            AnomalyUtils.ip_router_visualise_all_instances(df_day)

        with st.expander('Boxplot visualisation',expanded=False):
            AnomalyUtils.ip_router_visualise_all_boxplot(df_day)
    prompt.success('Data Loaded!')

    st.subheader('Training dataframes and results')
    with st.spinner('Training Model...'):        
        prompt = st.empty() 
        with st.expander('Outlier detection results', expanded=False):
            X = df_day.T
            # ###############################################
            # Identify outliers in the training dataset
            Isolation_Forest = IsolationForest()
            if model_name == 'Isolation_Forest':
                st.write("Outliers Detected by Isolation Forest:")
                st.write("======================================")
            outlier1 = pd.DataFrame()
            list_1 = []
            for i in range(len(X)):
                a = X.loc[i].to_numpy().reshape((len(X.loc[i]), 1))
                b = Isolation_Forest.fit_predict(a)
                c = pd.DataFrame(b)
                if model_name == 'Isolation_Forest':
                    st.write("Outliers Detected in NR_RTR_PORT_"+str(i+1)+":", c[c == -1].count().sum())
                    st.write("====================================")
                outlier1 = outlier1.append(c.T)
                list_1.append(c[c == -1].count().sum())

            if model_name == 'Isolation_Forest':
                st.write("Total Outliers Detected: ", outlier1[outlier1 == -1].count().sum())
                st.write("=================================")

                x = AnomalyUtils.ip_router_get_labels()
                y = list_1

                fig = px.bar(x=x, y=y)
                st.plotly_chart(fig)


            outlier1 = outlier1.reset_index(drop=True)
            # ######################################################

            # ######################################################
            model = AnomalyUtils.AutoEncoder(output_units = 1)
            if model_name == 'Autoencoder':
                st.write("Outliers Detected by Autoencoder:")
                st.write("======================================")
            outlier2 = pd.DataFrame()
            list_2 = []
            for i in range(len(X)):
                a = X.loc[i].to_numpy().reshape((len(X.loc[i]), 1))
                threshold = AnomalyUtils.auto_encoder_find_threshold(model, a)
                predictions = AnomalyUtils.auto_encoder_get_predictions(model, a, threshold)
                c = pd.DataFrame(predictions)
                c = c.replace({0: -1}).astype("int")
                if model_name == 'Autoencoder':
                    st.write("Outliers Detected in NR_RTR_PORT_"+str(i+1)+":", c[c == -1].count().sum())
                    st.write("====================================")
                outlier2 = outlier2.append(c.T)
                list_2.append(c[c == -1].count().sum())

            if model_name == 'Autoencoder':
                st.write("Total Number of Outliers Detected: ", outlier2[outlier2 == -1].count().sum())
                st.write("=================================")

                x = AnomalyUtils.ip_router_get_labels()
                y = list_2

                fig = px.bar(x=x, y=y)
                st.plotly_chart(fig)

            outlier2 = outlier2.reset_index(drop=True)

            # ######################################################

            # ######################################################
            # Identify outliers in the training dataset
            Local_Outlier_Factor = LocalOutlierFactor()

            outlier3 = pd.DataFrame()
            if model_name == 'Local_Outlier_Factor':
                st.write("Outliers Detected by LocalOutlierFactor:")
                st.write("========================================")
            list_3 = []
            for i in range(len(X)):
                a = X.loc[i].to_numpy().reshape((len(X.loc[i]), 1))
                b = Local_Outlier_Factor.fit_predict(a)
                c = pd.DataFrame(b)
                if model_name == 'Local_Outlier_Factor':
                    st.write("Outliers Detected in NR_RTR_PORT_"+str(i+1)+":", c[c == -1].count().sum())
                    st.write("====================================")
                outlier3 = outlier3.append(c.T)
                list_3.append(c[c == -1].count().sum())
            if model_name == 'Local_Outlier_Factor':
                st.write("Number Outliers Detected: ", outlier3[outlier3 == -1].count().sum())
                st.write("=================================")

                x = AnomalyUtils.ip_router_get_labels()
                y = list_3

                fig = px.bar(x=x, y=y)
                st.plotly_chart(fig)

            outlier3 = outlier3.reset_index(drop=True)
            # ######################################################

            # ######################################################
            # Identify outliers in the training dataset
            One_Class_SVM = OneClassSVM()
            if model_name == 'One_Class_SVM':
                st.write("Outliers Detected by OneClassSVM:")
                st.write("=================================")
            outlier4 = pd.DataFrame()
            list_4 = []
            for i in notebook.tqdm(range(len(X))):
                a = X.loc[i].to_numpy().reshape((len(X.loc[i]), 1))
                b = One_Class_SVM.fit_predict(a)
                c = pd.DataFrame(b)
                if model_name == 'One_Class_SVM':
                    st.write("Outliers Detected in NR_RTR_PORT_"+str(i+1)+":", c[c == -1].count().sum())
                    st.write("====================================")
                outlier4 = outlier4.append(c.T)
                list_4.append(c[c == -1].count().sum())

            if model_name == 'One_Class_SVM':
                st.write("Number Outliers Detected: ", outlier4[outlier4 == -1].count().sum())
                st.write("=================================")

                x = AnomalyUtils.ip_router_get_labels()
                y = list_4

                fig = px.bar(x=x, y=y)
                st.plotly_chart(fig)

            outlier4 = outlier4.reset_index(drop=True)
            # ######################################################

            # ######################################################
            # Identify outliers in the training dataset
            outlier_detection = DBSCAN()
            if model_name == 'DBSCAN':
                st.write("Outliers Detected by DBSCAN:")
                st.write("============================")
            outlier5 = pd.DataFrame()
            list_5 = []
            for i in notebook.tqdm(range(len(X))):
                a = X.loc[i].to_numpy().reshape((len(X.loc[i]), 1))
                b = outlier_detection.fit_predict(a)
                c = pd.DataFrame(b)
                c = c.replace({0: 1}).astype("int")
                if model_name == 'DBSCAN':
                    st.write("Outliers Detected in NR_RTR_PORT_"+str(i+1)+":", c[c == -1].count().sum())
                    st.write("====================================")
                outlier5 = outlier5.append(c.T)
                list_5.append(c[c == -1].count().sum())
            if model_name == 'DBSCAN':
                st.write("Number Outliers Detected: ", outlier5[outlier5 == -1].count().sum())
                st.write("=================================")

                x = AnomalyUtils.ip_router_get_labels()
                y = list_5

                fig = px.bar(x=x, y=y)
                st.plotly_chart(fig)

            outlier5 = outlier5.reset_index(drop=True)
            for i in range(len(X)):
                outlier5.loc[i] = np.where(outlier5.loc[i] != -1, 1, outlier5.loc[i])
            # ######################################################

        with st.expander('All instances with Anomlies', expanded=False):
            if model_name == 'Isolation_Forest':
                AnomalyUtils.ip_router_visualise_all_instances_with_anomaly(X, df_day, outlier1)
            elif model_name == 'Autoencoder':
                AnomalyUtils.ip_router_visualise_all_instances_with_anomaly( X, df_day, outlier2)
            elif model_name == 'Local_Outlier_Factor':
                AnomalyUtils.ip_router_visualise_all_instances_with_anomaly( X, df_day, outlier3)   
            elif model_name == 'One_Class_SVM':
                AnomalyUtils.ip_router_visualise_all_instances_with_anomaly( X, df_day, outlier4) 
            elif model_name == 'DBSCAN':
                AnomalyUtils.ip_router_visualise_all_instances_with_anomaly( X, df_day, outlier5) 
    prompt.success('Model Loaded!')

    st.subheader('Testing and Evaluation')
    with st.spinner('Test Model...'):        
        prompt = st.empty() 
        with st.expander('Test Data', expanded=False):
            X_test = X.copy()
            for i in range(len(X_test)):
                condition = np.average(X_test.loc[i]) + 3 * (statistics.stdev(X_test.loc[i]))
                X_test.loc[i] = np.where(X_test.loc[i] > condition, -1, X_test.loc[i])

            for i in range(len(X_test)):
                X_test.loc[i] = np.where(X_test.loc[i] != -1, 1, X_test.loc[i])
                
            X_test = X_test.astype("int")
            st.write(X_test)
        with st.expander('Test Results', expanded=False):
            accuracy_1 = AnomalyUtils.ip_router_test(X_test, outlier1, False)
            accuracy_2 = AnomalyUtils.ip_router_test(X_test, outlier2, False)
            accuracy_3 = AnomalyUtils.ip_router_test(X_test, outlier3, False)
            accuracy_4 = AnomalyUtils.ip_router_test(X_test, outlier4, False)
            accuracy_5 = AnomalyUtils.ip_router_test(X_test, outlier5, False)
            if model_name == 'Isolation_Forest':
                AnomalyUtils.ip_router_test(X_test, outlier1, True)
            elif model_name == 'Autoencoder':
                AnomalyUtils.ip_router_test(X_test, outlier2, True)
            elif model_name == 'Local_Outlier_Factor':
                AnomalyUtils.ip_router_test(X_test, outlier3, True)   
            elif model_name == 'One_Class_SVM':
               AnomalyUtils.ip_router_test(X_test, outlier4, True) 
            elif model_name == 'DBSCAN':
                AnomalyUtils.ip_router_test(X_test, outlier5, True)
    prompt.success('Testing Complete!')
    if compare:
        st.subheader('Comparison Of Models')
        with st.spinner('Comparing...'):
            prompt = st.empty()
            with st.expander('Compare Results', expanded=False):
                for i in range(20):
                    st.write("========================================================")
                    st.write("========================= NR_RTR_PORT_" + str(i+1) + " ======================")
                    st.write("========================================================")
                    
                    st.write("Performance with Models")
                    st.write("=======================")
                    x=PrettyTable()
                    x.add_column("Classifier Name",["Isolation Forest","Autoencoder","Local Outlier Factor", "One Class SVM", "DBSCAN"])
                    x.add_column("Accuracy Score",[str(round(accuracy_1[i]*100,2)) + " %", str(round(accuracy_2[i]*100,2)) + " %", str(round(accuracy_3[i]*100,2)) + " %", str(round(accuracy_4[i]*100,2)) + " %", str(round(accuracy_5[i]*100,2)) + " %"])
                    st.write(x)

                    x = ["Isolation Forest","Autoencoder","Local Outlier Factor", "One Class SVM", "DBSCAN"]
                    y = [round(accuracy_1[i],2), round(accuracy_2[i],2), round(accuracy_3[i],2), round(accuracy_4[i],2), round(accuracy_5[i],2)]

                    plt.rcParams["figure.figsize"] = (14,5)
                    fig, ax = plt.subplots() 
                    width = 0.25
                    ind = np.arange(len(y))
                    ax.barh(ind, y, width, color="blue")
                    ax.set_yticks((ind+width/2)-0.1)
                    ax.set_yticklabels(x, minor=False)
                    for i, v in enumerate(y):
                        ax.text(v + 0.01, i , str(v), color='blue', fontweight='bold')
                    plt.title('Comparision')
                    plt.xlabel('Accuracy Score')
                    plt.ylabel('Classifiers')      
                    st.pyplot(fig)
                    plt.clf()
        prompt.success('Comparison Complete!')
