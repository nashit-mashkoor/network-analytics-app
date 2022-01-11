import matplotlib.pyplot as plt
import streamlit as st
#'Linear_Regression'
#'Decision_Tree_Regression'
#'Gradient_Boosting_Regression'
#'AdaBoost_Regression'
#'Support_Vector_Machine'
#'Regression_Using_Neural_Networks'
def nn_plot_history(history, key):
  plt.plot(history.history[key])
  plt.plot(history.history['val_'+key])
  plt.xlabel("Epochs")
  plt.ylabel(key)
  plt.legend([key, 'val_'+key])
  st.pyplot(plt.gcf())
  plt.clf()