# Import Libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns

from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


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


  def find_threshold(model, x_train_scaled):
      reconstructions = model.predict(x_train_scaled)
      reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)
      threshold = np.mean(reconstruction_errors.numpy()) + np.std(reconstruction_errors.numpy())
      return threshold

  def get_predictions(model, x_test_scaled, threshold):
    predictions = model.predict(x_test_scaled)
    errors = tf.keras.losses.msle(predictions, x_test_scaled)
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
    return preds

class AnomalyModel:
  def __init__(self):
      self.supported_models = {'Isolation_Forest':IsolationForest(),
                              'Local_Outlier_Factor': LocalOutlierFactor(),
                              'One_Class_SVM': OneClassSVM(), 
                              'DBSCAN':DBSCAN(), 
                              'Autoencoder': AutoEncoder(output_units=1)}
      
  def __str__(self):
      return self.supported_models

  def train_predict(self, selected_model, X, verbose=True, output_labels=[]):
    model = self.supported_models[selected_model]
    outlier1 = pd.DataFrame()
    list_1 = []
    for i in range(len(X)):
        a = X.loc[i].to_numpy().reshape((len(X.loc[i]), 1))
        if selected_model=='Autoencoder':
          threshold = AutoEncoder.find_threshold(model, a)
          predictions = AutoEncoder.get_predictions(model, a, threshold)
          c = pd.DataFrame(predictions)
          c = c.replace({0: -1}).astype("int")
        else:
          b = model.fit_predict(a)
          c = pd.DataFrame(b)
        if verbose:
          st.write("Outliers Detected in IP_Link_"+str(i+1)+":", c[c == -1].count().sum())
        outlier1 = outlier1.append(c.T)
        list_1.append(c[c == -1].count().sum())
    if verbose:
      st.write("Total Outliers Detected: ", outlier1[outlier1 == -1].count().sum())

    
    if verbose:
      x = output_labels
      y = list_1
      fig = px.bar(x=x, y=y)
      st.plotly_chart(fig)
      st.write("Outliers Detected by {}:".format(selected_model))
      st.write(outlier1)
    outlier1 = outlier1.reset_index(drop=True)
    return outlier1

  def test(self, dataset_name, X_test, outlier, verbose=True):
    accuracy_1 = dict()
    if dataset_name == 'IP_Link_Jitter':
      name="IP_Link_"
      pd_key=30
    elif dataset_name == 'IP_Router_Port_Total_Traffic_Rate':
      name="NR_RTR_PORT_"
      pd_key=20
    for i in range(pd_key):     
      # Evalution
      y_test = X_test.loc[i].to_numpy().flatten()
      y_pred = outlier.loc[i].to_numpy().flatten()

      # Accuracy Score
      accuracy_1[i] = accuracy_score(y_test, y_pred)
      # Classification Report
      report = classification_report(y_test, y_pred)
      # Confusion Matrix
      matrix = confusion_matrix(y_test, y_pred)
      confusion_matrix_heatmap = confusion_matrix(y_test, y_pred, normalize = 'true')

      if verbose:
        st.write("========================={}".format(name) + str(i+1) + " ======================")
        st.write("  Accuracy: ", round(accuracy_1[i]*100,2), "%")

        st.write("Classification Report:")
        st.write(report)

        st.write("Confusion Matrix:")
        st.table(matrix)

        st.write("Confusion Matrix with Heat MAP:")
        fig = plt.figure(figsize=(10, 4))
        sns.heatmap(confusion_matrix_heatmap, annot=True)
        st.pyplot(fig)
    return accuracy_1

  def compare_all(self, dataset_name, X, X_test):
    if dataset_name == 'IP_Link_Jitter':
      name="IP_Link_"
      pd_key=30
    elif dataset_name == 'IP_Router_Port_Total_Traffic_Rate':
      name="NR_RTR_PORT_"
      pd_key=20

    accuracy = {}
    for model in self.supported_models:
      accuracy[model]=self.test(dataset_name, X_test, self.train_predict(model, X, verbose=False, output_labels=[]), verbose=False)
    
    for i in range(pd_key):
      st.write("========================= {}".format(name) + str(i+1) + " ======================")      
      st.write("Performance with Models")
      data=data = {'Classifier Name': [accur for accur in accuracy], 'Accuracy Score': [str(round(accuracy[accur][i]*100,2))+ " %" for accur in accuracy]}  
      st.table(data)

      # x = ["Isolation Forest","Autoencoder","Local Outlier Factor", "One Class SVM", "DBSCAN"]
      # y = [round(accuracy_1[i],2), round(accuracy_2[i],2), round(accuracy_3[i],2), round(accuracy_4[i],2), round(accuracy_5[i],2)]

      # plt.rcParams["figure.figsize"] = (14,5)
      # fig, ax = plt.subplots() 
      # width = 0.25
      # ind = np.arange(len(y))
      # ax.barh(ind, y, width, color="blue")
      # ax.set_yticks((ind+width/2)-0.1)
      # ax.set_yticklabels(x, minor=False)
      # for i, v in enumerate(y):
      #     ax.text(v + 0.01, i , str(v), color='blue', fontweight='bold')
      # plt.title('Comparision')
      # plt.xlabel('Accuracy Score')
      # plt.ylabel('Classifiers')      
      # plt.show()