import plotly.graph_objects as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import statistics

from io import BytesIO
from plotly.subplots import make_subplots

class AnomalyDataLoader:
    def __init__(self, dataset_name):
        self.supported_datasets = {'IP_Link_Jitter': 'data/Jitter_15min_15days_30nodes.csv',
                                'IP_Router_Port_Total_Traffic_Rate': 'data/Total_Traffic_Rate_5min_7days_20nodes.csv'}
        self.dataset_name=dataset_name
    def __str__(self):
        return self.supported_models

    @st.cache
    def get_raw_df(self):
        return pd.read_csv(self.supported_datasets[self.dataset_name])

    @st.cache
    def get_output_labels(self):
        if self.dataset_name == 'IP_Link_Jitter':
            return ["IP_Link_1",
                    "IP_Link_2",
                    "IP_Link_3",
                    "IP_Link_4",
                    "IP_Link_5",
                    "IP_Link_6",
                    "IP_Link_7",
                    "IP_Link_8",
                    "IP_Link_9",
                    "IP_Link_10",
                    "IP_Link_11",
                    "IP_Link_12",
                    "IP_Link_13",
                    "IP_Link_14",
                    "IP_Link_15",
                    "IP_Link_16",
                    "IP_Link_17",
                    "IP_Link_18",
                    "IP_Link_19",
                    "IP_Link_20",
                    "IP_Link_21",
                    "IP_Link_22",
                    "IP_Link_23",
                    "IP_Link_24",
                    "IP_Link_25",
                    "IP_Link_26",
                    "IP_Link_27",
                    "IP_Link_28",
                    "IP_Link_29",
                    "IP_Link_30",
                ]
        elif self.dataset_name == 'IP_Router_Port_Total_Traffic_Rate':
            return ["NR_RTR_PORT_1",
                    "NR_RTR_PORT_2",
                    "NR_RTR_PORT_3",
                    "NR_RTR_PORT_4",
                    "NR_RTR_PORT_5",
                    "NR_RTR_PORT_6",
                    "NR_RTR_PORT_7",
                    "NR_RTR_PORT_8",
                    "NR_RTR_PORT_9",
                    "NR_RTR_PORT_10",
                    "NR_RTR_PORT_11",
                    "NR_RTR_PORT_12",
                    "NR_RTR_PORT_13",
                    "NR_RTR_PORT_14",
                    "NR_RTR_PORT_15",
                    "NR_RTR_PORT_16",
                    "NR_RTR_PORT_17",
                    "NR_RTR_PORT_18",
                    "NR_RTR_PORT_19",
                    "NR_RTR_PORT_20"
                ]
    
    def get_preprocessed_df(self, raw_df):
        sample_data = raw_df
        sample_data = sample_data.T[1:]
        sample_data = sample_data.reset_index()
        sample_data = sample_data.rename({'index': 'date'}, axis=1)
        
        if self.dataset_name == 'IP_Link_Jitter':
            sample_data['date'] = pd.to_datetime(sample_data['date'], dayfirst = True, errors='coerce')
        elif self.dataset_name == 'IP_Router_Port_Total_Traffic_Rate':
            sample_data['date'] = pd.to_datetime(sample_data['date'], errors='coerce')
        return sample_data

    def fill_null_values(self, preprocessed_df):
        sample_data = preprocessed_df
        for i in range(len(sample_data.columns)-1):
            sample_data[i] = sample_data[i].groupby(sample_data.date.dt.hour).transform(lambda x: x.fillna(x.median()))
        return sample_data

    def get_df_day(self, processed_df):
        sample_data = processed_df
        df_day = pd.DataFrame()
        for i in range(len(sample_data.columns)-1):
            df_day[i] = sample_data.groupby(sample_data.date)[i].sum()
        df_day.index = pd.DatetimeIndex(df_day.index.values, freq=df_day.index.inferred_freq)
        return df_day

    def get_test_day(self, X):
        X_test = X.copy()
        for i in range(len(X_test)):
            condition = np.average(X_test.loc[i]) + 3 * (statistics.stdev(X_test.loc[i]))
            X_test.loc[i] = np.where(X_test.loc[i] > condition, -1, X_test.loc[i])

        for i in range(len(X_test)):
            X_test.loc[i] = np.where(X_test.loc[i] != -1, 1, X_test.loc[i])
            
        X_test = X_test.astype("int")
        return X_test

    def visualise_data(self, processed_df, df_day):
        sample_data = processed_df
        
        if self.dataset_name == 'IP_Link_Jitter':
            plt.figure(figsize=(14, 10), dpi=80)

            
            params = {'legend.fontsize': 10,'legend.handlelength': 2}
            plt.rcParams.update(params)
            plt.plot(df_day[0], label="IP_LINK_1")
            plt.plot(df_day[1], label="IP_LINK_2")
            plt.plot(df_day[2], label="IP_LINK_3")
            plt.plot(df_day[3], label="IP_LINK_4")
            plt.plot(df_day[4], label="IP_LINK_5")
            plt.plot(df_day[5], label="IP_LINK_6")
            plt.plot(df_day[6], label="IP_LINK_7")
            plt.plot(df_day[7], label="IP_LINK_8")
            plt.plot(df_day[8], label="IP_LINK_9")
            plt.plot(df_day[9], label="IP_LINK_10")
            plt.plot(df_day[10], label="IP_LINK_11")
            plt.plot(df_day[11], label="IP_LINK_12")
            plt.plot(df_day[12], label="IP_LINK_13")
            plt.plot(df_day[13], label="IP_LINK_14")
            plt.plot(df_day[14], label="IP_LINK_15")
            plt.plot(df_day[15], label="IP_LINK_16")
            plt.plot(df_day[16], label="IP_LINK_17")
            plt.plot(df_day[17], label="IP_LINK_18")
            plt.plot(df_day[18], label="IP_LINK_19")
            plt.plot(df_day[19], label="IP_LINK_20")
            plt.plot(df_day[20], label="IP_LINK_21")
            plt.plot(df_day[21], label="IP_LINK_22")
            plt.plot(df_day[22], label="IP_LINK_23")
            plt.plot(df_day[23], label="IP_LINK_24")
            plt.plot(df_day[24], label="IP_LINK_25")
            plt.plot(df_day[25], label="IP_LINK_26")
            plt.plot(df_day[26], label="IP_LINK_27")
            plt.plot(df_day[27], label="IP_LINK_28")
            plt.plot(df_day[28], label="IP_LINK_29")
            plt.plot(df_day[29], label="IP_LINK_30")

            plt.title("Data Visualization")
            plt.legend(loc="upper left")
        elif self.dataset_name == 'IP_Router_Port_Total_Traffic_Rate':
            plt.figure(figsize=(14, 10), dpi=80)

            params = {'legend.fontsize': 10,'legend.handlelength': 2}
            plt.rcParams.update(params)
            plt.plot(df_day[0], label="NR_RTR_PORT_1")
            plt.plot(df_day[1], label="NR_RTR_PORT_2")
            plt.plot(df_day[2], label="NR_RTR_PORT_3")
            plt.plot(df_day[3], label="NR_RTR_PORT_4")
            plt.plot(df_day[4], label="NR_RTR_PORT_5")
            plt.plot(df_day[5], label="NR_RTR_PORT_6")
            plt.plot(df_day[6], label="NR_RTR_PORT_7")
            plt.plot(df_day[7], label="NR_RTR_PORT_8")
            plt.plot(df_day[8], label="NR_RTR_PORT_9")
            plt.plot(df_day[9], label="NR_RTR_PORT_10")
            plt.plot(df_day[10], label="NR_RTR_PORT_11")
            plt.plot(df_day[11], label="NR_RTR_PORT_12")
            plt.plot(df_day[12], label="NR_RTR_PORT_13")
            plt.plot(df_day[13], label="NR_RTR_PORT_14")
            plt.plot(df_day[14], label="NR_RTR_PORT_15")
            plt.plot(df_day[15], label="NR_RTR_PORT_16")
            plt.plot(df_day[16], label="NR_RTR_PORT_17")
            plt.plot(df_day[17], label="NR_RTR_PORT_18")
            plt.plot(df_day[18], label="NR_RTR_PORT_19")
            plt.plot(df_day[19], label="NR_RTR_PORT_20")

            plt.title("Data Visualization")
            plt.legend(loc="upper left")
        st.pyplot(plt.gcf())

    def visualise_all_instances(self, df_day):

        if self.dataset_name == 'IP_Link_Jitter':
            name="IP_Link_"
        elif self.dataset_name == 'IP_Router_Port_Total_Traffic_Rate':
            name="NR_RTR_PORT_"
        fig = make_subplots(rows=len(df_day.columns), cols=1)
        for i in range(len(df_day.columns)):
            fig.add_trace(go.Scatter(x=df_day.index, y=df_day[i], name = name + str(i+1)),row=i+1, col=1)
        fig.update_layout(height=10000, width=800, title_text="Visualize all Instances with plotly")
        st.plotly_chart(fig)
    
    def visualise_all_instances_with_anomaly(self, X, df_day, outlier):
        if self.dataset_name == 'IP_Link_Jitter':
            name="IP_Link_"
        elif self.dataset_name == 'IP_Router_Port_Total_Traffic_Rate':
            name="NR_RTR_PORT_"
        fig = make_subplots(rows=len(df_day.columns), cols=1)
        for i in range(len(df_day.columns)):
            fig.add_trace(go.Scatter(x=X.loc[i].index, y=X.loc[i], name=name + str(i+1)),row=i+1, col=1)
            fig.add_trace (go.Scatter (x=X.loc[i][outlier.loc[i][outlier.loc[i]==-1].index].index, y=X.loc[i][outlier.loc[i][outlier.loc[i]==-1].index], 
                                    mode = 'markers', name = 'normal', marker=dict (color='red', size=5)),row=i+1, col=1)
        fig.update_layout(height=10000, width=800, title_text="Visualize all Instances with Anomlies")
        st.plotly_chart(fig)

    def visualise_boxplot(self, df_day):
        if self.dataset_name == 'IP_Link_Jitter':
            plot = px.Figure()

            for i in range(len(df_day.columns)):
                plot.add_trace(px.Box(y=df_day[i], name = "IP_Link_" + str(i+1)))
            plot.update_layout(height=5000)
        elif self.dataset_name == 'IP_Router_Port_Total_Traffic_Rate':    
            plot = px.Figure()
            for i in range(len(df_day.columns)):
                plot.add_trace(px.Box(y=df_day[i], name = "IP_Link_" + str(i+1)))

            plot.update_layout(height=5000)
        st.plotly_chart(plot)

    def render_data(self):
        sample_data = pd.read_csv("data/Jitter_15min_15days_30nodes.csv")
        sample_data = sample_data.T[1:]
        sample_data = sample_data.reset_index()
        sample_data = sample_data.rename({'index': 'date'}, axis=1)
        sample_data['date'] = pd.to_datetime(sample_data['date'], dayfirst = True, errors='coerce') 
        for i in range(len(sample_data.columns)-1):
            sample_data[i] = sample_data[i].groupby(sample_data.date.dt.hour).transform(lambda x: x.fillna(x.median()))

        df_day = pd.DataFrame()
        for i in range(len(sample_data.columns)-1):
            df_day[i] = sample_data.groupby(sample_data.date)[i].sum()
        df_day.index = pd.DatetimeIndex(df_day.index.values, freq=df_day.index.inferred_freq)
        st.write(df_day)
        from matplotlib.pyplot import figure
        figure(figsize=(14, 10), dpi=80)

        params = {'legend.fontsize': 10,'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.plot(df_day[0], label="IP_LINK_1")
        plt.plot(df_day[1], label="IP_LINK_2")
        plt.plot(df_day[2], label="IP_LINK_3")
        plt.plot(df_day[3], label="IP_LINK_4")
        plt.plot(df_day[4], label="IP_LINK_5")
        plt.plot(df_day[5], label="IP_LINK_6")
        plt.plot(df_day[6], label="IP_LINK_7")
        plt.plot(df_day[7], label="IP_LINK_8")
        plt.plot(df_day[8], label="IP_LINK_9")
        plt.plot(df_day[9], label="IP_LINK_10")
        plt.plot(df_day[10], label="IP_LINK_11")
        plt.plot(df_day[11], label="IP_LINK_12")
        plt.plot(df_day[12], label="IP_LINK_13")
        plt.plot(df_day[13], label="IP_LINK_14")
        plt.plot(df_day[14], label="IP_LINK_15")
        plt.plot(df_day[15], label="IP_LINK_16")
        plt.plot(df_day[16], label="IP_LINK_17")
        plt.plot(df_day[17], label="IP_LINK_18")
        plt.plot(df_day[18], label="IP_LINK_19")
        plt.plot(df_day[19], label="IP_LINK_20")
        plt.plot(df_day[20], label="IP_LINK_21")
        plt.plot(df_day[21], label="IP_LINK_22")
        plt.plot(df_day[22], label="IP_LINK_23")
        plt.plot(df_day[23], label="IP_LINK_24")
        plt.plot(df_day[24], label="IP_LINK_25")
        plt.plot(df_day[25], label="IP_LINK_26")
        plt.plot(df_day[26], label="IP_LINK_27")
        plt.plot(df_day[27], label="IP_LINK_28")
        plt.plot(df_day[28], label="IP_LINK_29")
        plt.plot(df_day[29], label="IP_LINK_30")

        plt.title("Data Visualization")
        plt.legend(loc="upper left")
        
        st.pyplot(plt.gcf())