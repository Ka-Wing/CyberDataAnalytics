import itertools
import pandas as pd
import time

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA, ARMAResults
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.stattools import acf, pacf
from sklearn.decomposition import PCA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
import math
from assignment2.saxpy import SAX
import seaborn as sns
from matplotlib import pyplot
from sklearn.metrics import recall_score, roc_curve, auc, confusion_matrix, mean_squared_error
import warnings
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation
import numpy

from statsmodels.tsa.stattools import adfuller

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

    def plots(self):
        signals = pd.read_csv("BATADAL_dataset03.csv")
        with open("a.txt", "w") as hallo:
            hallo.write(str(signals.corr()))
        sns.heatmap(data=signals.corr(), cbar=True, cmap='coolwarm')


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




    def arma(self):
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
            if sensors[i] != 'L_T7':
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
    def discrete_models_task(self, plot_alphabet=False):
        signals = self.batadal3

        signals = signals['L_T2']
        w = 500
        sax = SAX(wordSize=w, alphabetSize=8, epsilon=1e-6)
        normalized_signals = sax.normalize(signals)
        paa, _ = sax.to_PAA(normalized_signals)

        alphabet = sax.alphabetize(paa)
        print(len(alphabet))
        print(type(alphabet))
        print(alphabet)

        _, ax = pyplot.subplots()
        ax.set_color_cycle(['blue', 'blue', 'green'])
        #sns.tsplot(signals, color="red")
        sns.tsplot(normalized_signals, color="lightblue")
        x, y = self._paa_plot(paa, signals.shape[0], w)
        print("x = ", x)
        print("y = ", y)
        print(y)
        # pyplot.plot(paa)

        if plot_alphabet:
            self._alphabet_plot(alphabet, x, y)

        pyplot.plot(x, y)
        pyplot.show()


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




if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Fill in the right path of the dataset.
    b = batadal("BATADAL_dataset03.csv", "BATADAL_dataset04.csv", "BATADAL_test_dataset.csv")

    #Familirization Task
    b.plots()
    # b.water_level_prediction()

    # ARMA Task
    # b.arma() # This method plots the (p)acf and other graphs. Later, the order was added after looking at the graphs.
    # b.dftest(self): # Augmented Dickey Fuller Test

    # Discrete models task
    # b.discrete_models_task(plot_alphabet=True) # parameter is for plotting the alphabet of the graph.

    # PCA