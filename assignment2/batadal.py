#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:00:28 2018

@author: smto
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
from saxpy import SAX

def plot():
    # read in the data to a pandas dataframe
    signals = pd.read_csv('BATADAL_dataset03.csv')
    signals2 = pd.read_csv('BATADAL_dataset04.csv')
    # plot the heatmap with correlations
    #plt.subplots(figsize=(13,10))
    #sns.heatmap(data=signals.corr(), xticklabels=True, yticklabels=True, linewidths=1.0, cbar = True, cmap = 'coolwarm')
    
    plt.subplots(figsize=(13,10))
    sns.heatmap(data=signals2.corr(), xticklabels=True, yticklabels=True, linewidths=1.0, cbar = True, cmap = 'coolwarm')

def predict():
    signals = pd.read_csv('BATADAL_dataset03.csv')
    signals_test = pd.read_csv('BATADAL_test_dataset.csv')
    
    training = signals[['DATETIME', 'L_T1']]
    testing = signals_test[['DATETIME', 'L_T1']]
    print(training.head(5))
    print(training.dtypes)
    
    #train auto regression
    model = AR(training)
    model_fit = model.fit()
    
    # make predictions
    predictions = model_fit.predict(start=len(training), end=len(training)+len(testing)-1)
    # evaluate predictions
    error = mean_squared_error(testing, predictions)
    print("mse = " + error)
    
    #plot results
    pyplot.plot(testing)
    pyplot.plot(predictions, color='red')
    pyplot.show()
    
def pca_task():
    # read in the data to a pandas dataframe
    signals = pd.read_csv('BATADAL_dataset03.csv', parse_dates = True, index_col='DATETIME')
    signals2 = pd.read_csv('BATADAL_dataset04.csv', parse_dates = True, index_col='DATETIME')
    
    labels = signals2[' ATT_FLAG']
    
    #print(signals.head(5))
    
    # standardize the data
    # fit on the training set and transform on the training + test set
    scaler = StandardScaler()
    scaler.fit(signals)
    
    training = scaler.transform(signals)
    testing = scaler.transform(signals2)
    
    #print(training)
    #print(training.shape)
    
    # perform pca
    pca = PCA(n_components=23)
    training = pca.fit_transform(training)
    
    print(pca.explained_variance_ratio_)
    
def discrete_models_task():
    sensors = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2', 'F_PU4', 'F_PU6', 'F_PU7',
                   'F_PU8', 'F_PU10', 'F_PU11', 'F_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                   'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']

    nan = ['F_PU3', 'F_PU5', 'F_PU9']

    # read in the data to a pandas dataframe
    signals = pd.read_csv('BATADAL_dataset03.csv', parse_dates = True, index_col='DATETIME')
    
    sensor_data = signals[sensors]
    
    # perform SAX
    s = SAX()
    # perform PAA on training data
    paa_sensors, original_indices = s.to_PAA(sensor_data)
    # normalize
    normalized_paa_sensors = s.normalize(paa_sensors)
    # convert PAA of training data to series of letters
    letters = s.alphabetize(normalized_paa_sensors)
    
    print(sensor_data.head(5))
    print(paa_sensors)
    print(normalized_paa_sensors)
    print(original_indices)
    print(letters)
    
    # plot discretization
    
    
    
#plot()
#pca_task()
#predict()
discrete_models_task()