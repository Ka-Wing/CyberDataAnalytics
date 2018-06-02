import itertools
import pandas as pd
import time

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA, ARMAResults
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
import seaborn as sns
from matplotlib import pyplot
from sklearn.metrics import recall_score, roc_curve, auc, confusion_matrix
import warnings

from statsmodels.tsa.stattools import adfuller

class batadal(object):

    batadal3 = None
    batadal4 = None
    batadaltest = None

    def parser(self, x):
            return pd.datetime.strptime(x, '%d/%m/%y %H')

    def __init__(self, batadal3, batadal4, batadaltest):
        self.batadal3 = pd.read_csv(batadal3, header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)
        self.batadal4 = pd.read_csv(batadal4, header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)
        self.batadaltest = pd.read_csv(batadaltest, header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)

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
        sensors = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2', 'F_PU4', 'F_PU6', 'F_PU7',
                   'F_PU8', 'F_PU10', 'F_PU11', 'F_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                   'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']

        sensors = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2', 'F_PU4', 'F_PU6', 'F_PU7',
                   'F_PU8', 'F_PU10', 'F_PU11', 'F_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                   'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
        #sensors = ['L_T2']

        signals = self.batadal3

        # print(signals.iloc[0])


        for sensor in sensors:
            print(sensor)
            signals2 = signals[[sensor]]
            #print(signals2.head())
            #signals2.plot()
            #pyplot.title(sensor)
            #pyplot.get_current_fig_manager().window.showMaximized()
            #pyplot.show()

            p = 11
            d = 0
            q = 2

            plot_acf(signals2)
            aaseeef = acf(signals2)
            for i in range(len(aaseeef)):
                if aaseeef[i] < 0:
                    p = i
                    break

            peeaaseeef = pacf(signals2)
            for j in range(len(peeaaseeef)):

                if peeaaseeef[j] < 0:
                    q = j
                    break


            P = Q = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            D = [0, 1]

            orders = []#(a, b, c) for a in P for b in D for c in Q]
            print(len(orders))
            orders = list(set(orders))

            print(len(orders))
            print(orders)

            #pyplot.title("autocorrelation")
            #pyplot.show()

            #plot_pacf(signals2, lags=100)
            #pyplot.title("partial autocorrelation")
            #pyplot.show()

            model = None

            lowest = 9999999999999999
            lowest_order = (0,0,0)

            print(time.ctime())

            tried = []

            for pie in range(0, 20):
                tries = 3
                for pee in range(0, pie):
                    for kuu in range(0, pee):
                        order = (pee, 0, kuu)
                        if order not in tried:
                            tried.append(order)
                        else:
                            continue

                        try:
                            if True:
                                model = ARIMA(signals2, order=order)
                            elif(sensor == 'L_T1'):
                                model = ARIMA(signals2, order=(11, 0, 3))
                            elif(sensor == 'L_T2'):
                                model = ARIMA(signals2, order=(8, 0, 2))
                            elif (sensor == 'L_T3'):
                                model = ARIMA(signals2, order=(3, 0, 4))
                            elif (sensor == 'L_T4'):
                                model = ARIMA(signals2, order=(6, 0, 2))
                            elif (sensor == 'L_T5'):
                                model = ARIMA(signals2, order=(4, 0, 3))
                            elif (sensor == 'L_T6'):
                                model = ARIMA(signals2, order=(2, 0, 4))
                            elif (sensor == 'L_T7'):
                                model = ARIMA(signals2, order=(2, 0, 4))
                            elif (sensor == 'F_PI1'):
                                model = ARIMA(signals2, order=(9, 0, 3))
                            elif (sensor == 'F_PU2'):
                                model = ARIMA(signals2, order=(10, 0, 3))
                            elif (sensor == 'F_PU4'):
                                model = ARIMA(signals2, order=(4, 0, 2))
                            elif (sensor == 'F_PU6'):
                                model = ARIMA(signals2, order=(1, 0, 2))
                            elif (sensor == 'F_PU7'):
                                model = ARIMA(signals2, order=(3, 0, 3))
                            elif (sensor == 'F_PU8'):
                                model = ARIMA(signals2, order=(4, 0, 3))
                            elif (sensor == 'F_PU10'):
                                model = ARIMA(signals2, order=(3, 0, 3))
                            elif (sensor == 'F_PU11'):
                                model = ARIMA(signals2, order=(1, 0, 1))
                            elif (sensor == 'F_V2'):
                                model = ARIMA(signals2, order=(5, 0, 3))
                            elif (sensor == 'P_J280'):
                                model = ARIMA(signals2, order=(2, 0, 9))
                            elif (sensor == 'P_J269'):
                                model = ARIMA(signals2, order=(9, 0, 3))
                            elif (sensor == 'P_J300'):
                                model = ARIMA(signals2, order=(6, 0, 2))
                            elif (sensor == 'P_J256'):
                                model = ARIMA(signals2, order=(4, 0, 3))
                            elif (sensor == 'P_J289'):
                                model = ARIMA(signals2, order=(6, 0, 2))
                            elif (sensor == 'P_J415'):
                                model = ARIMA(signals2, order=(3, 0, 3))
                            elif (sensor == 'P_J302'):
                                model = ARIMA(signals2, order=(5, 0, 3))
                            elif (sensor == 'P_J306'):
                                model = ARIMA(signals2, order=(4, 0, 3))
                            elif (sensor == 'P_J307'):
                                model = ARIMA(signals2, order=(5, 0, 3))
                            elif (sensor == 'P_J317'):
                                model = ARIMA(signals2, order=(6, 0, 3))
                            elif (sensor == 'P_J422'):
                                model = ARIMA(signals2, order=(6, 0, 2))
                            else:
                                model = ARIMA(signals2, order=o)

                            model_fit = model.fit(disp=0)
                            print(order, str(model_fit.aic))
                            if(model_fit.aic < lowest):
                                tries == 3
                                lowest_order = order
                                lowest = model_fit.aic


                        #print(model_fit.summary())
                        except Exception as e:
                            print(str(e))
                            continue
                tries = tries - 1
                if tries == 0:
                    break

            for pie in range(0, 20):
                tries = 3
                for pee in range(0, pie):
                    for kuu in range(0, pee):
                        order = (kuu, 0, pee)
                        if order not in tried:
                            tried.append(order)
                        else:
                            continue

                        try:
                            if True:
                                model = ARIMA(signals2, order=order)
                            elif (sensor == 'L_T1'):
                                model = ARIMA(signals2, order=(11, 0, 3))
                            elif (sensor == 'L_T2'):
                                model = ARIMA(signals2, order=(8, 0, 2))
                            elif (sensor == 'L_T3'):
                                model = ARIMA(signals2, order=(3, 0, 4))
                            elif (sensor == 'L_T4'):
                                model = ARIMA(signals2, order=(6, 0, 2))
                            elif (sensor == 'L_T5'):
                                model = ARIMA(signals2, order=(4, 0, 3))
                            elif (sensor == 'L_T6'):
                                model = ARIMA(signals2, order=(2, 0, 4))
                            elif (sensor == 'L_T7'):
                                model = ARIMA(signals2, order=(2, 0, 4))
                            elif (sensor == 'F_PI1'):
                                model = ARIMA(signals2, order=(9, 0, 3))
                            elif (sensor == 'F_PU2'):
                                model = ARIMA(signals2, order=(10, 0, 3))
                            elif (sensor == 'F_PU4'):
                                model = ARIMA(signals2, order=(4, 0, 2))
                            elif (sensor == 'F_PU6'):
                                model = ARIMA(signals2, order=(1, 0, 2))
                            elif (sensor == 'F_PU7'):
                                model = ARIMA(signals2, order=(3, 0, 3))
                            elif (sensor == 'F_PU8'):
                                model = ARIMA(signals2, order=(4, 0, 3))
                            elif (sensor == 'F_PU10'):
                                model = ARIMA(signals2, order=(3, 0, 3))
                            elif (sensor == 'F_PU11'):
                                model = ARIMA(signals2, order=(1, 0, 1))
                            elif (sensor == 'F_V2'):
                                model = ARIMA(signals2, order=(5, 0, 3))
                            elif (sensor == 'P_J280'):
                                model = ARIMA(signals2, order=(2, 0, 9))
                            elif (sensor == 'P_J269'):
                                model = ARIMA(signals2, order=(9, 0, 3))
                            elif (sensor == 'P_J300'):
                                model = ARIMA(signals2, order=(6, 0, 2))
                            elif (sensor == 'P_J256'):
                                model = ARIMA(signals2, order=(4, 0, 3))
                            elif (sensor == 'P_J289'):
                                model = ARIMA(signals2, order=(6, 0, 2))
                            elif (sensor == 'P_J415'):
                                model = ARIMA(signals2, order=(3, 0, 3))
                            elif (sensor == 'P_J302'):
                                model = ARIMA(signals2, order=(5, 0, 3))
                            elif (sensor == 'P_J306'):
                                model = ARIMA(signals2, order=(4, 0, 3))
                            elif (sensor == 'P_J307'):
                                model = ARIMA(signals2, order=(5, 0, 3))
                            elif (sensor == 'P_J317'):
                                model = ARIMA(signals2, order=(6, 0, 3))
                            elif (sensor == 'P_J422'):
                                model = ARIMA(signals2, order=(6, 0, 2))
                            else:
                                model = ARIMA(signals2, order=o)

                            model_fit = model.fit(disp=0)
                            print(order, model_fit.aic)
                            if (model_fit.aic < lowest):
                                tries == 3
                                lowest_order = order
                                lowest = model_fit.aic


                        # print(model_fit.summary())
                        except Exception as e:
                            print(str(e))
                            pass
                tries = tries - 1
                if tries == 0:
                    break

            print("lowest_aic: ", lowest)
            print("lowest_order: ", lowest_order)

            continue

            residuals = DataFrame(model_fit.resid)
            pyplot.title("Hello")
            sns.tsplot(model_fit.resid)
            pyplot.show()

            residuals.plot()
            pyplot.title("ARMA Fit Residual Error Line Plot")
            pyplot.show()
            residuals.plot(kind='kde')
            pyplot.title("ARMA Fit Residual Error Density Plot")
            pyplot.show()
            print(residuals.describe())

            print("AIC: ", ARMAResults.aic(model_fit))
            print("BIC: ", ARMAResults.bic(model_fit))

            #break



    def dftest(self):
        sensors = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2', 'F_PU4', 'F_PU6', 'F_PU7',
                   'F_PU8', 'F_PU10', 'F_PU11', 'F_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                   'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']

        nan = ['F_PU3', 'F_PU5', 'F_PU9']

        signals = pd.read_csv("BATADAL_dataset03.csv", header=0, parse_dates=[0], index_col=0, squeeze=True,
                              date_parser=self.parser)

        for sensor in sensors:
            print(sensor)
            try:
                signals2 = signals[sensor]
                result = adfuller(signals2)
                print('ADF Statistic: %f' % result[0])
                print('p-value: %f' % result[1])
                print('lags: %f' % result[2])
                print('observations: %f' % result[3])
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





if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Fill in the right path of the dataset.
    b = batadal("BATADAL_dataset03.csv", "BATADAL_dataset04.csv", "BATADAL_test_dataset.csv")


    # create_csv()
    # count()
    b.arma()
    # auto()
    # dftest()
    # payseae()
    # paysea()
    # b.water_level_prediction()
    # b.water_flow()