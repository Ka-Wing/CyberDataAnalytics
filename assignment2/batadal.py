#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:00:28 2018

@author: smto
"""

import warnings
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
from saxpy import SAX

class batadal(object):

    batadal3 = None
    batadal4 = None
    batadaltest = None
    sensors = None

    def parser(self, x):
            return pd.datetime.strptime(x, '%d/%m/%y %H')

    def __init__(self, batadal3, batadal4, batadaltest):
        self.batadal3 = pd.read_csv(batadal3, header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)
        self.batadal4 = pd.read_csv(batadal4, header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)
        self.batadaltest = pd.read_csv(batadaltest, header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)

        self.sensors = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2', 'F_PU4', 'F_PU6', 'F_PU7',
                   'F_PU8', 'F_PU10', 'F_PU11', 'F_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                   'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
        
    def plot(self):
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
    
    def predict(self):
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
        plt.plot(testing)
        plt.plot(predictions, color="red")
        plt.show()
    
    def pca_task(self):
        # read in the data to a pandas dataframe
        signals = pd.read_csv('BATADAL_dataset03.csv', parse_dates = True, index_col='DATETIME')
        signals2 = pd.read_csv('BATADAL_dataset04.csv', parse_dates = True, index_col='DATETIME')
        
        labels = signals2[' ATT_FLAG']
        
        # preprocess the data
        signals = signals.drop('ATT_FLAG', axis=1)
        signals2 = signals2.drop(' ATT_FLAG', axis=1)
        
        # standardize the data to have zero mean and unit variance
        scaler1 = StandardScaler()
        scaler1.fit(signals)
        training = scaler1.transform(signals)
        
        scaler2 = StandardScaler()
        scaler2.fit(signals2)
        testing = scaler2.transform(signals2)
        
        # perform pca to determine the normal and anomalous subspace
        pca = PCA()
        pca.fit(training)
        transformed_training = pca.transform(training)
        transformed_testing = pca.transform(testing)

        # print cumulative variance
        print(pca.explained_variance_ratio_.cumsum())
        #output:
        #[0.21494218 0.34874316 0.47677729 0.57654339 0.64867854 0.71628148
        # 0.77181216 0.82719161 0.8697186  0.90220261 0.9278643  0.95244475
        # 0.97108338 0.98616798 0.99348279 0.99639277 0.99748605 0.99827377
        # 0.99893749 0.99934407 0.99959747 0.9998036  0.99987097 0.99992423
        # 0.99995238 0.9999675  0.999976   0.99998424 0.99998976 0.99999518
        # 0.99999829 0.99999997 1.         1.         1.         1.
        # 1.         1.         1.         1.         1.         1.
        # 1.        ]
        
        # we select n_components=10 for the normal subspace, as this would give us 90% variance which is decent
        pca2 = PCA(n_components=10)
        pca2.fit(training)
        
        # apply magic from the paper "Diagnosing Network-Wide Traffic Anomalies" (page 223)
        components = pca2.components_
        P = np.transpose(components)
        P_T = components
        I = np.identity(43)
        
        C = np.matmul(P, P_T)
        C_anomaly = I - C
        
        y = transformed_training
        
        # project training data to the anomalous subspace
        y_residual = np.matmul(C_anomaly, np.transpose(y))
        
        # calculate SPE (training data)        
        spe = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            spe[i] = np.sum(np.square(np.subtract(np.transpose(y_residual)[i], y)[i]))

        # plot to determine threshold
        plt.hist(spe, bins="auto")
        plt.xlim(0, 100)
        plt.show()
        
        # set threshold based on the plot on 30 and detect anomalies in the testing data
        threshold = 30
        
        # project testing data to the anomalous subspace
        y_residual2 = np.matmul(C_anomaly, np.transpose(transformed_testing))
        
        # calculate SPE (testing data)
        spe2 = np.zeros(transformed_testing.shape[0])
        for i in range(transformed_testing.shape[0]):
            spe2[i] = np.sum(np.square(np.subtract(np.transpose(y_residual2)[i], transformed_testing)[i]))
        
        # determine what data is anomalous
        anomalous = np.zeros(transformed_testing.shape[0])
        for i in range(transformed_testing.shape[0]):
            # when spe > threshold then classify as anomalous
            if spe2[i] > threshold:
                anomalous[i] = 1
                
        # plot the anomalous datapoints in the testing set as classified
        plt.plot(anomalous)
        plt.show()
        
        # plot the spe of the testing set
        plt.plot(spe2)
        plt.show()
                
        # compute true negatives, true positives, false negatives, true positives,
        # and preicison and recall
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        for i in range(transformed_testing.shape[0]):
            if labels[i] == -999 and anomalous[i] == 0:
                tn = tn + 1
            if labels[i] == -999 and anomalous[i] == 1:
                fp = fp + 1
            if labels[i] == 1 and anomalous[i] == 0:
                fn = fn + 1
            if labels[i] == 1 and anomalous[i] == 1:
                tp = tp + 1
                
        print("tn: ", tn)
        print("fp: ", fp)
        print("fn: ", fn)
        print("tp: ", tp)
        print("precision: ", tp/(tp+fp))
        print("recall: ", tp/(tp+fn))
        
    def discrete_models_task(self):
        # read in the data to a pandas dataframe
        signals = pd.read_csv('BATADAL_dataset03.csv', parse_dates = True, index_col='DATETIME')
        
        sensor = signals['L_T1']
        # perform SAX (check: https://github.com/nphoff/saxpy)
        s = SAX(200, 8, 1e-6)
        # normalize the training data first
        normalized_sensor = s.normalize(sensor)
        # perform PAA on training data
        paa_sensor, original_indices = s.to_PAA(normalized_sensor)
        # convert PAA of training data to series of letters
        letters = s.alphabetize(paa_sensor)
        
        #print(sensor_data.head(5))
        print(normalized_sensor)
        print(paa_sensor)
        print(original_indices)
        print(letters)
        
        # plot discretization
        sns.tsplot(data=normalized_sensor)
        plt.plot([0, 1000],[0, 0], color="red")
        #sns.tsplot(data=paa_sensor, color="red")
        
        # create sliding windows
        s.sliding_window(letters, 49, 0.9)
        
        # use n-grams
        
    def pca_for_comparison_task(self):
        # read in the data to a pandas dataframe
        signals = pd.read_csv('BATADAL_dataset03.csv', parse_dates = True, index_col='DATETIME')
        signals2 = pd.read_csv('BATADAL_test_dataset_new.csv', parse_dates = True, index_col='DATETIME')
        
        labels = signals2['ATT_FLAG']
        
        # preprocess the data
        signals = signals.drop('ATT_FLAG', axis=1)
        signals2 = signals2.drop('ATT_FLAG', axis=1)
        
        # standardize the data to have zero mean and unit variance
        scaler1 = StandardScaler()
        scaler1.fit(signals)
        training = scaler1.transform(signals)
        
        scaler2 = StandardScaler()
        scaler2.fit(signals2)
        testing = scaler2.transform(signals2)
        
        # perform pca to determine the normal and anomalous subspace
        pca = PCA()
        pca.fit(training)
        transformed_training = pca.transform(training)
        transformed_testing = pca.transform(testing)

        # print cumulative variance
        print(pca.explained_variance_ratio_.cumsum())
        #output:
        #[0.21494218 0.34874316 0.47677729 0.57654339 0.64867854 0.71628148
        # 0.77181216 0.82719161 0.8697186  0.90220261 0.9278643  0.95244475
        # 0.97108338 0.98616798 0.99348279 0.99639277 0.99748605 0.99827377
        # 0.99893749 0.99934407 0.99959747 0.9998036  0.99987097 0.99992423
        # 0.99995238 0.9999675  0.999976   0.99998424 0.99998976 0.99999518
        # 0.99999829 0.99999997 1.         1.         1.         1.
        # 1.         1.         1.         1.         1.         1.
        # 1.        ]
        
        # we select n_components=10 for the normal subspace, as this would give us 90% variance which is decent
        pca2 = PCA(n_components=10)
        pca2.fit(training)
        
        # apply magic from the paper "Diagnosing Network-Wide Traffic Anomalies" (page 223)
        components = pca2.components_
        P = np.transpose(components)
        P_T = components
        I = np.identity(43)
        
        C = np.matmul(P, P_T)
        C_anomaly = I - C
        
        y = transformed_training
        
        # project training data to the anomalous subspace
        y_residual = np.matmul(C_anomaly, np.transpose(y))
        
        # calculate SPE (training data)        
        spe = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            spe[i] = np.sum(np.square(np.subtract(np.transpose(y_residual)[i], y)[i]))

        # plot to determine threshold
        plt.hist(spe, bins="auto")
        plt.xlim(0, 100)
        plt.show()
        
        # set threshold based on the plot on 30 and detect anomalies in the testing data
        threshold = 30
        
        # project testing data to the anomalous subspace
        y_residual2 = np.matmul(C_anomaly, np.transpose(transformed_testing))
        
        # calculate SPE (testing data)
        spe2 = np.zeros(transformed_testing.shape[0])
        for i in range(transformed_testing.shape[0]):
            spe2[i] = np.sum(np.square(np.subtract(np.transpose(y_residual2)[i], transformed_testing)[i]))
        
        # determine what data is anomalous
        anomalous = np.zeros(transformed_testing.shape[0])
        for i in range(transformed_testing.shape[0]):
            # when spe > threshold then classify as anomalous
            if spe2[i] > threshold:
                anomalous[i] = 1
                
        # plot the anomalous datapoints in the testing set as classified
        plt.plot(anomalous)
        plt.show()
        
        # plot the spe of the testing set
        plt.plot(spe2)
        plt.show()
                
        # compute true negatives, true positives, false negatives, true positives,
        # and preicison and recall
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        for i in range(transformed_testing.shape[0]):
            if labels[i] == 0 and anomalous[i] == 0:
                tn = tn + 1
            if labels[i] == 0 and anomalous[i] == 1:
                fp = fp + 1
            if labels[i] == 1 and anomalous[i] == 0:
                fn = fn + 1
            if labels[i] == 1 and anomalous[i] == 1:
                tp = tp + 1
                
        print("tn: ", tn)
        print("fp: ", fp)
        print("fn: ", fn)
        print("tp: ", tp)
        print("precision: ", tp/(tp+fp))
        print("recall: ", tp/(tp+fn))

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Fill in the right path of the dataset.
    b = batadal("BATADAL_dataset03.csv", "BATADAL_dataset04.csv", "BATADAL_test_dataset_new.csv")
    
    #plot()
    #b.pca_task()
    b.pca_for_comparison_task()
    #predict()
    #discrete_models_task()