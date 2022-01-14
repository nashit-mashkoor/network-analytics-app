import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.graph_objects as px
import seaborn as sns
from plotly.subplots import make_subplots

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError

# IP Link
def ip_link_visualise_all(df_day):
    fig = plt.figure(figsize=(14, 10), dpi=80)

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
    st.pyplot(fig)
    plt.clf()

def ip_link_visualise_all_instances(df_day):
    fig = make_subplots(rows=len(df_day.columns), cols=1)

    for i in range(len(df_day.columns)):
        fig.add_trace(go.Scatter(x=df_day.index, y=df_day[i], name = "IP_Link_" + str(i+1)),row=i+1, col=1)

    fig.update_layout(height=10000, width=800, title_text="Visualize all Instances with plotly")
    st.plotly_chart(fig)

def ip_link_visualise_all_boxplot(df_day):
    plot = px.Figure()

    for i in range(len(df_day.columns)):
        plot.add_trace(px.Box(y=df_day[i], name = "IP_Link_" + str(i+1)))

    plot.update_layout(height=5000)
    st.plotly_chart(plot)
def ip_link_visualise_all_instances_with_anomaly(X, df_day, outlier):
        name="IP_Link_"
        fig = make_subplots(rows=len(df_day.columns), cols=1)
        for i in range(len(df_day.columns)):
            fig.add_trace(go.Scatter(x=X.loc[i].index, y=X.loc[i], name=name + str(i+1)),row=i+1, col=1)
            fig.add_trace (go.Scatter (x=X.loc[i][outlier.loc[i][outlier.loc[i]==-1].index].index, y=X.loc[i][outlier.loc[i][outlier.loc[i]==-1].index], 
                                    mode = 'markers', name = 'normal', marker=dict (color='red', size=5)),row=i+1, col=1)
        fig.update_layout(height=10000, width=800, title_text="Visualize all Instances with Anomlies")
        st.plotly_chart(fig)
def ip_link_get_labels():
    return  ["IP_Link_1",
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

def ip_link_test( X_test, outlier, verbose):
    accuracy = dict()
    for i in range(30):
        if verbose:
            st.write("========================================================")
            st.write("========================= IP_LINK_" + str(i+1) + " ======================")
            st.write("========================================================")
        
        # Evalution
        y_test = X_test.loc[i].to_numpy().flatten()
        y_pred = outlier.loc[i].to_numpy().flatten()

        # Accuracy Score
        accuracy[i] = accuracy_score(y_test, y_pred)
        if verbose:
            st.write("======================")
            st.write("Accuracy Score:")
            st.write("======================")
            st.write("  Accuracy: ", round(accuracy[i]*100,2), "%")

        # Classification Report
        report = classification_report(y_test, y_pred)
        if verbose:
            st.write("======================")
            st.write("Classification Report:")
            st.write("======================")
            st.text(report)

        # Confusion Matrix
        matrix = confusion_matrix(y_test, y_pred)
        if verbose:
            st.write("=================")
            st.write("Confusion Matrix:")
            st.write("=================")
            st.write(matrix)

        # Heat Map
        if verbose:
            st.write("===============================")
            st.write("Confusion Matrix with Heat MAP:")
            st.write("===============================")
            confusion_matrix_heatmap = confusion_matrix(y_test, y_pred, normalize = 'true')
            sns.set(rc={'figure.figsize':(7,5)})
            sns.heatmap(confusion_matrix_heatmap, annot=True)
            st.pyplot(plt.gcf())
            plt.clf()
    return accuracy

# IP Router
def ip_router_visualise_all(df_day):
    fig = plt.figure(figsize=(14, 10), dpi=80)

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
    st.pyplot(fig)
    plt.clf()

def ip_router_visualise_all_instances(df_day):
    fig = make_subplots(rows=len(df_day.columns), cols=1)

    for i in range(len(df_day.columns)):
        fig.add_trace(go.Scatter(x=df_day.index, y=df_day[i], name = "NR_RTR_PORT_" + str(i+1)),row=i+1, col=1)

    fig.update_layout(height=10000, width=800, title_text="Visualize all Instances with plotly")
    st.plotly_chart(fig)
    plt.clf()

def ip_router_visualise_all_boxplot(df_day):
    plot = px.Figure()

    for i in range(len(df_day.columns)):
        plot.add_trace(px.Box(y=df_day[i], name = "NR_RTR_PORT_" + str(i+1)))

    plot.update_layout(height=5000)
    st.plotly_chart(plot)
    plt.clf()

def ip_router_get_labels():
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

def ip_router_visualise_all_instances_with_anomaly(X, df_day, outlier):
    name="NR_RTR_PORT_"
    fig = make_subplots(rows=len(df_day.columns), cols=1)
    for i in range(len(df_day.columns)):
        fig.add_trace(go.Scatter(x=X.loc[i].index, y=X.loc[i], name=name + str(i+1)),row=i+1, col=1)
        fig.add_trace (go.Scatter (x=X.loc[i][outlier.loc[i][outlier.loc[i]==-1].index].index, y=X.loc[i][outlier.loc[i][outlier.loc[i]==-1].index], 
                                mode = 'markers', name = 'normal', marker=dict (color='red', size=5)),row=i+1, col=1)
    fig.update_layout(height=10000, width=800, title_text="Visualize all Instances with Anomlies")
    st.plotly_chart(fig)

def ip_router_test( X_test, outlier, verbose):
    accuracy = dict()
    for i in range(20):
        if verbose:
            st.write("========================================================")
            st.write("========================= NR_RTR_PORT_" + str(i+1) + " ======================")
            st.write("========================================================")
        
        # Evalution
        y_test = X_test.loc[i].to_numpy().flatten()
        y_pred = outlier.loc[i].to_numpy().flatten()

        # Accuracy Score
        accuracy[i] = accuracy_score(y_test, y_pred)
        if verbose:
            st.write("======================")
            st.write("Accuracy Score:")
            st.write("======================")
            st.write("  Accuracy: ", round(accuracy[i]*100,2), "%")

        # Classification Report
        report = classification_report(y_test, y_pred)
        if verbose:
            st.write("======================")
            st.write("Classification Report:")
            st.write("======================")
            st.text(report)

        # Confusion Matrix
        matrix = confusion_matrix(y_test, y_pred)
        if verbose:
            st.write("=================")
            st.write("Confusion Matrix:")
            st.write("=================")
            st.write(matrix)

        # Heat Map
        if verbose:
            st.write("===============================")
            st.write("Confusion Matrix with Heat MAP:")
            st.write("===============================")
            confusion_matrix_heatmap = confusion_matrix(y_test, y_pred, normalize = 'true')
            sns.set(rc={'figure.figsize':(7,5)})
            sns.heatmap(confusion_matrix_heatmap, annot=True)
            st.pyplot(plt.gcf())
            plt.clf()
    return accuracy
# Auto Encoders
class AutoEncoder(Model):
  def __init__(self, output_units, code_size=8):
    super().__init__()
    self.encoder = Sequential([
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(code_size, activation='relu')
    ])
    self.decoder = Sequential([
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(output_units, activation='sigmoid')
    ])
  
  def call(self, inputs):
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded
  
def auto_encoder_find_threshold(model, x_train_scaled):
    reconstructions = model.predict(x_train_scaled)
    reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)
    threshold = np.mean(reconstruction_errors.numpy()) + np.std(reconstruction_errors.numpy())
    return threshold

def auto_encoder_get_predictions(model, x_test_scaled, threshold):
    predictions = model.predict(x_test_scaled)
    errors = tf.keras.losses.msle(predictions, x_test_scaled)
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
    return preds