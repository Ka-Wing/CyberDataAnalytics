import pandas as pd

from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import StandardScaler, normalize
from statsmodels.tsa.stattools import pacf
from sklearn.decomposition import PCA
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.tree import DecisionTreeRegressor
from pandas import DataFrame
from assignment2.saxpy import SAX
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
import numpy as np

from statsmodels.tsa.stattools import adfuller

class batadal(object):

    batadal3 = None
    batadal4 = None
    batadaltest = None
    batadaltest_new = None
    sensors = None

    def parser(self, x):
            return pd.datetime.strptime(x, '%d/%m/%y %H')

    def __init__(self, batadal3, batadal4, batadaltest, batadaltest_new):
        self.batadal3 = pd.read_csv(batadal3, header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)
        self.batadal4 = pd.read_csv(batadal4, header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)
        self.batadaltest = pd.read_csv(batadaltest, header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)
        self.batadaltest_new = pd.read_csv(batadaltest_new, header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)

        self.sensors = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2', 'F_PU4', 'F_PU6', 'F_PU7',
                   'F_PU8', 'F_PU10', 'F_PU11', 'F_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                   'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']

    def plots(self):
        # read in the data to a pandas dataframe
        signals = self.batadal3
        
        # plot the heatmap with correlations
        plt.subplots(figsize=(13,10))
        sns.heatmap(data=signals.corr(), xticklabels=True, yticklabels=True, linewidths=1.0, cbar = True, cmap = 'coolwarm')
        plt.show()
    
        # plot behavior of P_J280 and F_PU1
        normalized_signals_1 = normalize(signals['P_J280'][:300].values.reshape(1, -1))
        normalized_signals_2 = normalize(signals['F_PU1'][:300].values.reshape(1, -1))
        sns.tsplot(data=normalized_signals_1, color="red")
        sns.tsplot(data=normalized_signals_2)
        plt.show()
        
        # plot behavior of P_J269 and F_PU2
        normalized_signals_1 = normalize(signals['P_J269'][:300].values.reshape(1, -1))
        normalized_signals_2 = normalize(signals['F_PU2'][:300].values.reshape(1, -1))
        #normalized_signals_3 = normalize(signals['S_PU2'][:300].values.reshape(1, -1))
        sns.tsplot(data=normalized_signals_1, color="red")
        sns.tsplot(data=normalized_signals_2)
        #sns.tsplot(data=normalized_signals_2, color="green")
        plt.show()


    def water_level_prediction(self):
        signals = self.batadal3
        test = self.batadaltest

        # Get only L_T1 til L_T7 columns
        signals = signals.ix[:, :'L_T7']
        test = test.ix[:, :'L_T7']

        # List of temporary data
        list = []

        # Drop another column every loop and try to predict that dropped column.
        for i in range(len(signals.columns)):
            dropped_column = "L_T" + str(i + 1)

            s = signals
            t = test

            signals_X = s.drop(s.columns[i], axis=1)
            signals_y = s[s.columns[i]]

            test_X = t.drop(t.columns[i], axis=1)
            test_y = t[t.columns[i]]

            # Use DecisionTreeRegressor to predict the dropped column values in the test set.
            classifier = DecisionTreeRegressor()
            classifier = classifier.fit(signals_X, signals_y)
            test_predicted_y = classifier.predict(test_X)

            test_y = test_y.values
            difference = test_predicted_y - test_y

            list.append([difference, dropped_column])

            amount = 0 # Number of off predictions
            accumulated_diff_percentage = 0 # Percentage of difference in actual and prediction value.

            # Calculate the difference and change in percentage for each corresponding actual and prediction value
            for j in range(len(difference)):
                a = test_y[j]
                b = test_predicted_y[j]
                diff = b - a
                percentage = (test_predicted_y[j]-test_y[j])/test_y[j]
                if diff != 0.0:
                    amount += 1
                    accumulated_diff_percentage += abs((test_predicted_y[j]-test_y[j])/test_y[j])

            title = "Dropped column: " + dropped_column
            print(title)
            print("Amount of differences:", amount)
            print("Average in percentage: ", accumulated_diff_percentage/amount)
            print("")

        # Each line in a plot.
        for i in range(len(list)):
            pyplot.plot(list[i][0], label=list[i][1])
            pyplot.legend(loc='upper left')
            pyplot.title("Differences of prediction and actual in absolute numbers")
            pyplot.show()

        # All lines in a plot.
        for i in range(len(list)):
            pyplot.plot(list[i][0], label=list[i][1])
        pyplot.legend(loc='upper left')
        pyplot.title("Differences of prediction and actual in absolute numbers")
        pyplot.show()




    def arma(self, signal='L_T1'):
        sensors = self.sensors
        signals = self.batadal3
        test = self.batadaltest
        # These orders were found using the bruteforce method, and corresponds with the sensors.
        orders = [(10, 0, 2), # L_T1
                  (8, 0, 2), # L_T2
                  (3, 0, 4), # L_T3
                  (6, 0, 2), # L_T4
                  (4, 0, 3), # L_T5
                  (2, 0, 4), # L_T6
                  (2, 0, 4), # L_T7
                  (9, 0, 3), # F_PU1
                  (11, 0, 3), # F_PU2
                  (4, 0, 2), # F_PU4
                  (1, 0, 2), # F_PU6
                  (4, 0, 3), # F_PU7
                  (3, 0, 3), # F_PU8
                  (1, 0, 1), # F_PU10
                  (5, 0, 3), # F_PU11
                  (5, 0, 3), # F_V2
                  (2, 0, 9), # P_J280
                  (9, 0, 3), # P_J269
                  (6, 0, 2), # P_J300
                  (4, 0, 3), # P_J256
                  (6, 0, 2), # P_J289
                  (3, 0, 3), # P_J415
                  (5, 0, 3), # P_J302
                  (4, 0, 3), # P_J306
                  (5, 0, 3), # P_J307
                  (3, 0, 3), # P_J317
                  (6, 0, 3), # P_J14
                  (6, 0, 2) # P_J422
                 ]

        for i in range(len(sensors)):
            if sensors[i] != signal:
                continue
            signals2 = signals[[sensors[i]]]

            # Plot signal
            pyplot.plot(signals2)
            pyplot.title("Data " + sensors[i])
            pyplot.show()

            # ACF
            plot_acf(signals2)
            pyplot.title("autocorrelation " + sensors[i])
            pyplot.show()

            # PACF
            pyplot.plot(pacf(signals2))
            pyplot.title("partial autocorrelation " + sensors[i])
            pyplot.show()

            # Fit model
            model = ARIMA(signals2, order=orders[i])
            model_fit = model.fit(disp=0)

            #Plot residuals
            residuals = DataFrame(model_fit.resid)
            pyplot.title("Residuals " + sensors[i])
            sns.tsplot(model_fit.resid)
            pyplot.show()
            residuals.plot()
            pyplot.title("ARMA Fit Residual Error Line Plot " + sensors[i])
            pyplot.show()
            residuals.plot(kind='kde')
            pyplot.title("ARMA Fit Residual Error Density Plot " + sensors[i])
            pyplot.show()
            print(residuals.describe())

            predictions = [] # List of predictions used to plotting the graph.
            history = signals2[sensors[i]].tolist()[-100:] # Last 100 entries of the  training set.

            test2 = test[sensors[i]].tolist() # Change from Series to list for compatability reasons.

            for t in range(len(test2)):
                if t < 5:
                    history.append(test2[t])
                    continue

                # Run ARIMA and give prediction
                predicted_value=0 # Used when ARIMA fails.
                try:
                    model = ARIMA(history, order=orders[i])
                    model_fit = model.fit(disp=0)
                    output = model_fit.forecast()
                    predicted_value = output[0]
                except:
                    pass
                predictions.append(predicted_value)

                # Effectively add one element to the end of the list, and taking one out in the beginning
                actual_value = test2[t]
                history.append(actual_value)
                history = history[-105:]

                print(str(t) + '/' + str(len(test2)) +' predicted=%f, expected=%f' % (predicted_value, actual_value))

            # Take out the first 5 that was used as history.
            test2 = test2[5:]
            error = mean_squared_error(test2, predictions)
            print('Test MSE: %.3f' % error)

            # Plot
            pyplot.title("Predictions for " + str(sensors[i]) )
            pyplot.plot(test2)
            pyplot.plot(predictions, color='red')
            pyplot.show()



            break


    # Augmented Dickey Fuller Test
    def dftest(self):
        sensors = self.sensors
        signals = self.batadal3


        for sensor in sensors:
            print(sensor)
            try:
                signals2 = signals[sensor]
                result = adfuller(signals2)
                print('ADF Statistic: %f' % result[0])
                print('p-value: %f' % result[1])
                print('Lags: %f' % result[2])
                print('Observations: %f' % result[3])
                print('Critical Values:')
                for key, value in result[4].items():
                    print('\t%s: %.3f' % (key, value))
            except:
                pass
            print("\n\n")

    # Plotting the SAX/PAA
    def discrete_models_task(self, plot_alphabet=False, signal='L_T1', w=100, a=8):
        signals = self.batadal3

        signals = signals[signal]
        sax = SAX(wordSize=w, alphabetSize=a, epsilon=1e-6)
        normalized_signals = sax.normalize(signals)
        paa, _ = sax.to_PAA(normalized_signals)

        alphabet = sax.alphabetize(paa)
        alphabet_string = self._numbers_to_letter(alphabet)
        print(alphabet_string)

        _, ax = pyplot.subplots()
        ax.set_color_cycle(['blue', 'blue', 'green'])
        #sns.tsplot(signals, color="red")
        sns.tsplot(normalized_signals, color="lightblue")
        x, y = self._paa_plot(paa, signals.shape[0], w)
        # pyplot.plot(paa)

        if plot_alphabet:
            self._alphabet_plot(alphabet_string, x, y)

        pyplot.plot(x, y)
        pyplot.show()

    def _numbers_to_letter(self, alphabet_string):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        string = ""
        for i in alphabet_string:
            number = int(i)
            string += alphabet[number]
        return string

    # Plots the alphabet on the PAA line.
    def _alphabet_plot(self, alphabet, x, y):
        j = 0

        for i in range(len(x)):
            if i % 2 != 0:
                continue
            xvalue = (x[i+1]-x[i])/2 + x[i]
            yvalue = y[i] + 0.05
            pyplot.text(xvalue, yvalue, alphabet[j])
            j = j + 1

    # Returns the correct PAA plot out of the PAA returned by the code of Qin Lin.
    def _paa_plot(self, data, n, w):
        hallo = n/w
        x = []
        y = []
        for i in range(0, w):
            x.append(hallo*i)
            x.append(hallo*(i+1))

        for i in range(len(data)):
            y.append(data[i])
            y.append(data[i])

        return x, y

    # PCA
    def pca_task(self):
        # read in the data to a pandas dataframe
        signals = self.batadal3
        signals2 = self.batadal4
        
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
        
    # Comparison task: PCA method
    def pca_for_comparison_task(self):
        # read in the data to a pandas dataframe
        signals = self.batadal3
        signals2 = self.batadaltest_new
        
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
    b = batadal("BATADAL_dataset03.csv", "BATADAL_dataset04.csv", "BATADAL_test_dataset.csv", "BATADAL_test_dataset new.csv")

    # Familirization Task
    # b.plots()
    # b.water_level_prediction()

    # ARMA Task
    # b.arma(signal='L_T1') # Method that makes an ARIMA model.
    # b.dftest() # Augmented Dickey Fuller Test

    # Discrete models task
    # b.discrete_models_task(plot_alphabet=True, signal='L_T1', w=70, a=8) # parameter is for plotting the alphabet of
    #  the graph.

    # PCA
    # b.pca_task()
    
    # Comparison task: PCA method
    # b.pca_for_comparison_task()