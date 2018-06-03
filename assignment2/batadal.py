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
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
from saxpy import SAX

def plot():
    # read in the data to a pandas dataframe
    signals = pd.read_csv('BATADAL_dataset03.csv', parse_dates = True, index_col='DATETIME')
    
    # plot the heatmap with correlations
    plt.subplots(figsize=(13,10))
    sns.heatmap(data=signals.corr(), xticklabels=True, yticklabels=True, linewidths=1.0, cbar = True, cmap = 'coolwarm')

    # plot behavior of P_J280 and F_PU1
    normalized_signals_1 = normalize(signals['P_J280'][:300].values.reshape(1, -1))
    normalized_signals_2 = normalize(signals['F_PU1'][:300].values.reshape(1, -1))
    sns.tsplot(data=normalized_signals_1, color="red")
    sns.tsplot(data=normalized_signals_2)
    
    # plot behavior of P_J269 and F_PU2
    normalized_signals_1 = normalize(signals['P_J269'][:300].values.reshape(1, -1))
    normalized_signals_2 = normalize(signals['F_PU2'][:300].values.reshape(1, -1))
    #normalized_signals_3 = normalize(signals['S_PU2'][:300].values.reshape(1, -1))
    sns.tsplot(data=normalized_signals_1, color="red")
    sns.tsplot(data=normalized_signals_2)
    #sns.tsplot(data=normalized_signals_2, color="green")

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
    
    #sensor_data = signals[sensors]
    
    # perform SAX (check: https://github.com/nphoff/saxpy)
    s = SAX(8, 7, 1e-6)
    # normalize the training data first
    normalized_signals = s.normalize(signals)
    # perform PAA on training data
    paa_signals, original_indices = s.to_PAA(normalized_signals)
    # normalize
    #normalized_paa_signals = s.normalize(paa_signals)
    # convert PAA of training data to series of letters
    letters = s.alphabetize(paa_signals)
    
    #print(sensor_data.head(5))
    print(normalized_signals)
    print(paa_signals)
    print(original_indices)
    print(letters)
    
    # plot discretization
    #sns.tsplot(data=normalized_signals)
    sns.tsplot(data=paa_signals, color="red")
    
    
    # create sliding windows
    #s.sliding_window(letters, 7, 0.9)
    
    # use n-grams
    
    
#plot()
#pca_task()
#predict()
discrete_models_task()