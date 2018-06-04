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

    def water_flow(self):
        #list[['L_T1', 'F_PU1', 'S_PU1'], ['L_T2', 'F_PU2', 'S_PU2']]

        sensors = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2', 'F_PU4', 'F_PU6', 'F_PU7',
                   'F_PU8', 'F_PU10', 'F_PU11', 'F_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                   'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']

        signals = self.batadal3
        test = self.batadaltest

        signals = signals.drop('ATT_FLAG', axis=1)

        print(len(signals.columns))
        print(len(test.columns))

        print(signals.iloc[0])
        print(test.iloc[0])

        # List of temporary data
        list = []
        percentages = []

        # Drop another column every loop and try to predict that dropped column.
        for i in range(len(sensors)):
            dropped_column = signals.columns[i]
            l = [dropped_column]

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
            pi = []

            # Calculate the difference and change in percentage for each corresponding actual and prediction value
            for j in range(len(difference)):
                a = test_y[j]
                b = test_predicted_y[j]
                diff = b - a
                if test_y[j] != 0.0:
                    percentage = abs((test_predicted_y[j]-test_y[j])/test_y[j])
                    if diff != 0.0:
                        pi.append(percentage)
                        amount += 1
                        if math.isnan(percentage):
                            accumulated_diff_percentage += percentage
            l.append(pi)

            title = "Dropped column: " + dropped_column
            print(title)
            print("Amount of differences:", amount)
            if amount == 0:
                print("Average in percentage: 0.0")
            else:
                average = (accumulated_diff_percentage/amount)
                print("Average in percentage: ", average)
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


    def count(self):
        signals = pd.read_csv("BATADAL_dataset03.csv")
        headers = ['DATETIME', 'L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2',
                   'F_PU3', 'S_PU3', 'F_PU4', 'S_PU4', 'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6', 'F_PU7', 'S_PU7', 'F_PU8',
                   'S_PU8', 'F_PU9', 'S_PU9', 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2', 'S_V2', 'P_J280', 'P_J269',
                   'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422',
                   'ATT_FLAG']


        print(signals.ATT_FLAG.value_counts())



    def create_csv(self):
        sensors = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2', 'F_PU4', 'F_PU6', 'F_PU7',
                   'F_PU8', 'F_PU10', 'F_PU11', 'F_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                   'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']

        signals = pd.read_csv("BATADAL_dataset03.csv")

        for sensor in sensors:
            signals = signals[["DATETIME", sensor]]
            signals.to_csv("a/" + str(sensor) + ".csv", sep=',', index=False)


    def arma(self):
        sensors = self.sensors
        signals = self.batadal3
        test = self.batadaltest
        # These orders were found using the bruteforce method, and corresponds with the sensors.
        orders = [(11, 0, 3),  # L_T1
                  (11, 0, 12), # L_T2
                  (3, 0, 4),#(17, 0, 18), # L_T3
                  (18, 0, 13), # L_T4
                  (11, 0, 17), # L_T5
                  (18, 0, 15), # L_T6
                  (18, 0, 13), # L_T7
                  (18, 0, 3), # F_PU1
                  (11, 0, 3), # F_PU2
                  (18, 0, 17), # F_PU4
                  (1, 0, 2), # F_PU6
                  (12, 0, 13), # F_PU7
                  (18, 0, 12), # F_PU8
                  (18, 0, 11), # F_PU10
                  (0, 0, 1), # F_PU11
                  (15, 0, 10), # F_V2
                  (18, 0, 2), # P_J280
                  (18, 0, 3), # P_J269
                  ]

        for i in range(len(sensors)):
            if sensors[i] != 'L_T3':
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
            # plot_pacf(signals2)
            # pyplot.title("partial autocorrelation " + sensors[i])
            # pyplot.show()

            # Fit model
            model = ARIMA(signals2, order=orders[i])
            model_fit = model.fit(disp=0)


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

            predictions = []
            history = []

            test2 = test[[sensors[i]]]

            history = signals2[sensors[i]].tolist()

            for t in range(len(test2)):
                if t < 5:
                    history.append(test2.iloc[t])
                    continue

                model = ARIMA(history, order=orders[i])
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test2.iloc[t]
                history.append(obs)
                print(str(t) + '/' + str(len(test2)) +' predicted=%f, expected=%f' % (yhat, obs))
            error = mean_squared_error(test, predictions)
            print('Test MSE: %.3f' % error)
            # plot
            pyplot.plot(test)
            pyplot.plot(predictions, color='red')
            pyplot.show()



            break


    # Method to bruteforce the order. The workings of this algorithms is not that important, as this method is a tool.
    def bruteforce_arma(self):
        sensors = self.sensors
        signals = self.batadal3

        for sensor in sensors:
            print(sensor)
            signals2 = signals[[sensor]]


            P = Q = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            D = [0, 1]

            # Make combination of each p, d and q. Then remove duplicates
            orders = [(a, b, c) for a in P for b in D for c in Q]
            orders = list(set(orders))

            model = None

            # These variables are used as final results.
            lowest = 9999999999999999
            lowest_order = (0,0,0)

            # Orders that has been tried already.
            tried = []


            # Try al combinations up til 20 by increasing p, after three iterations of p without any lower AIC,
            # then break.
            for pie in range(0, 20):
                tries = 3
                for pee in range(0, pie):
                    for kuu in range(0, pee):
                        # Set order
                        order = (pee, 0, kuu)

                        # See if order is already tried. Break if yes, continue if no.
                        if order not in tried:
                            tried.append(order)
                        else:
                            continue

                        # Try to make the model and check if AIC is lower than the lowest seen up til now.
                        try:
                            model = ARIMA(signals2, order=order)
                            model_fit = model.fit(disp=0)
                            print(order, str(model_fit.aic))
                            if(model_fit.aic < lowest):
                                tries == 3
                                lowest_order = order
                                lowest = model_fit.aic

                        except Exception as e:
                            print(str(e))
                            continue
                tries = tries - 1
                # Breaks if after three iteration no lower AIC is found.
                if tries == 0:
                    break

            # Try al combinations up til 20 by increasing q, after three iterations of q without any lower AIC,
            # then break.
            for pie in range(0, 20):
                tries = 3
                for pee in range(0, pie):
                    for kuu in range(0, pee):
                        order = (kuu, 0, pee) # Note that only this line is different than the loop before.
                        if order not in tried:
                            tried.append(order)
                        else:
                            continue

                        try:
                            model = ARIMA(signals2, order=order)
                            model_fit = model.fit(disp=0)
                            print(order, model_fit.aic)
                            if (model_fit.aic < lowest):
                                tries == 3
                                lowest_order = order
                                lowest = model_fit.aic


                        except Exception as e:
                            print(str(e))
                            pass
                tries = tries - 1
                if tries == 0:
                    break

            print("lowest_aic: ", lowest)
            print("lowest_order: ", lowest_order)



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

    def payseae(self):
        # Loading training data
        signals_training = pd.read_csv("BATADAL_dataset04.csv", header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)

        # Replacing all -999 ATT_FLAGS to 0 for compatibility purposes.
        signals_training[' ATT_FLAG'] = signals_training[' ATT_FLAG'].replace(-999, 0)

        # Loading test data
        signals_test = pd.read_csv("BATADAL_test_dataset new.csv", header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)

        # Dividing dataset into features and labels.
        signals_training_features = signals_training.drop([' ATT_FLAG'], axis=1)
        signals_training_labels = signals_training[' ATT_FLAG']
        signals_test_features = signals_test.drop(['ATT_FLAG'], axis=1)
        signals_test_labels = signals_test['ATT_FLAG']

        # Performing PCA on training features.
        pca = PCA(n_components=26) # 26 is bruteforced
        signals_training_pca = pca.fit_transform(signals_training_features)

        # Running classifier on PCA'ed dataset.
        classifier = RandomForestClassifier()
        classifier.fit(signals_training_pca, signals_training_labels)

        # Performing PCA on test features.
        signals_test_pca = pca.transform(signals_test_features)
        signals_test_pca

        # Do pedictions.
        predict_labels = classifier.predict(signals_test_pca)

        conf_matrix = confusion_matrix(signals_test_labels, predict_labels)
        tn, fp, fn, tp = conf_matrix.ravel()
        print("tn: ", tn)
        print("fp: ", fp)
        print("fn: ", fn)
        print("tp: ", tp)
        print("\n")

        print(conf_matrix)

        print(type(signals_training_features))
        print(signals_training_features.shape[0])
        print(signals_training_features.shape[1])
        print(type(signals_training_pca))
        print(signals_training_pca.shape[0])
        print(signals_training_pca.shape[1])
        print(type(signals_training_labels))
        print(signals_training_labels.shape[0])
        #print(signals_training_labels.shape[1])

        print("Hallo")

        residual = signals_training_labels.values-predict_labels
        print(type(residual))
        print(len(residual))
        pyplot.plot(residual)
        pyplot.show()

        sns.residplot(signals_training_pca, signals_training_labels)
        pyplot.show()


        print(type(predict_labels))
        print(len(predict_labels))


    def paysea(self):
        pca = PCA()

        # Loading training data
        signals_training = pd.read_csv("BATADAL_dataset04.csv", header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)

        # Replacing all -999 ATT_FLAGS to 0 for compatibility purposes.
        signals_training[' ATT_FLAG'] = signals_training[' ATT_FLAG'].replace(-999, 0)

        # Loading test data
        signals_test = pd.read_csv("BATADAL_test_dataset.csv", header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)

        pca.fit(signals_training)
        print(pca.n_components_)

        training_pca = pca.transform(signals_training)
        test_pca = pca.transform(signals_test)


    def p(self):
        # Load the data
        signals = self.batadal3

        # Scale the data so that the columns have a zero mean.
        scaled_training_set  = StandardScaler().fit_transform(signals)
        print(scaled_training_set.shape[0], scaled_training_set.shape[1])

        # Apply PCA
        pca_training_set = PCA().fit_transform(scaled_training_set)
        print(pca_training_set.shape[0], pca_training_set.shape[1])





        exit()
        pca_training_set = PCA().fit_transform(scaled_training_set)

    def discrete_models_task(self, plot_alphabet=False):
        sensors = self.sensors
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
        x, y = self.paa_plot(paa, signals.shape[0], w)
        print("x = ", x)
        print("y = ", y)
        print(y)
        # pyplot.plot(paa)

        if plot_alphabet:
            self.alphabet_plot(alphabet, x, y)

        pyplot.plot(x, y)
        pyplot.show()

        sax.windo

    # Plots the alphabet on the PAA line.
    def alphabet_plot(self, alphabet, x, y):
        j = 0

        for i in range(len(x)):
            if i % 2 != 0:
                continue
            xvalue = (x[i+1]-x[i])/2 + x[i]
            yvalue = y[i] + 0.05
            pyplot.text(xvalue, yvalue, alphabet[j])
            j = j + 1

    # Returns the correct PAA plot out of the PAA returned by the code of Qin Lin.
    def paa_plot(self, data, n, w):
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

    def tslearnn(self):
        numpy.random.seed(0)
        # Generate a random walk time series
        n_ts, sz, d = 1, 100, 1
        dataset = random_walks(n_ts=n_ts, sz=sz, d=d)
        dataset = pd.read_csv("BATADAL_dataset03.csv", header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)
        dataset = dataset['L_T1']
        print(type(dataset))

        scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
        dataset = scaler.fit_transform(dataset)

        # PAA transform (and inverse transform) of the data
        n_paa_segments = 11
        paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
        paa_dataset_inv = paa.inverse_transform(paa.fit_transform(dataset))

        # SAX transform
        n_sax_symbols = 8
        sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
        sax_dataset_inv = sax.inverse_transform(sax.fit_transform(dataset))

        # 1d-SAX transform
        n_sax_symbols_avg = 8
        n_sax_symbols_slope = 8
        one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                        alphabet_size_slope=n_sax_symbols_slope)
        one_d_sax_dataset_inv = one_d_sax.inverse_transform(one_d_sax.fit_transform(dataset))

        pyplot.figure()
        pyplot.subplot(2, 2, 1)  # First, raw time series
        pyplot.plot(dataset[0].ravel(), "b-")
        pyplot.title("Raw time series")

        pyplot.subplot(2, 2, 2)  # Second, PAA
        pyplot.plot(dataset[0].ravel(), "b-", alpha=0.4)
        pyplot.plot(paa_dataset_inv[0].ravel(), "b-")
        pyplot.title("PAA")

        pyplot.subplot(2, 2, 3)  # Then SAX
        pyplot.plot(dataset[0].ravel(), "b-", alpha=0.4)
        pyplot.plot(sax_dataset_inv[0].ravel(), "b-")
        pyplot.title("SAX, %d symbols" % n_sax_symbols)

        pyplot.subplot(2, 2, 4)  # Finally, 1d-SAX
        pyplot.plot(dataset[0].ravel(), "b-", alpha=0.4)
        pyplot.plot(one_d_sax_dataset_inv[0].ravel(), "b-")
        pyplot.title("1d-SAX, %d symbols (%dx%d)" % (n_sax_symbols_avg * n_sax_symbols_slope,
                                                  n_sax_symbols_avg,
                                                  n_sax_symbols_slope))

        pyplot.tight_layout()
        pyplot.show()



if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Fill in the right path of the dataset.
    b = batadal("BATADAL_dataset03.csv", "BATADAL_dataset04.csv", "BATADAL_test_dataset.csv")

    # ARMA Task
    # b.bruteforce_arma()
    b.arma()
    # auto()
    # def dftest(self): # Augmented Dickey Fuller Test

    # Discrete models task
    # b.discrete_models_task(plot_alphabet=False)
    # b.tslearnn()

    # PCA
    # payseae()
    # paysea()
    # b.water_level_prediction()
    # b.water_flow()
    # b.p()