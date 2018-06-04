import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import itertools
import seaborn as sns
import datetime

from sklearn.tree import DecisionTreeClassifier


class FraudDetection:

    reader = None
    currencies = {'GBP': 1.550, 'AUD': 0.735, 'MXN': 0.062, 'SEK': 0.118, 'NZD': 0.677}
    list = []
    csv = None

    def read_csv(self):
        transactions = pd.read_csv("C:\\Users\\kw\\Dropbox\\TU Delft\\Y2\\Q4\\CS4035 Cyber Data Analytics\\Week 1 - "
                                "Credit Card Fraud\\data_for_student_case.csv(1)\\data_for_student_case.csv")
        return transactions

    # Preprocesses the dataset to get better results.
    def preprocess_csv(self):

        csv = self.read_csv()

        # Remove all "Refused" and changes "Settled" and "Chargeback" to "0" and "1" respectively
        csv = csv[csv.simple_journal != 'Refused']
        csv['simple_journal'] = csv['simple_journal'].replace('Settled', 0).replace('Chargeback', 1)
        # print(csv.iloc[1])

        # Change all 4, 5, 6 to 3
        csv['cvcresponsecode'] = csv['cvcresponsecode'].replace(4, 3).replace(5, 3).replace(6, 3)

        # Change all currencies to USD for better comparison between amount of different currencies.
        # for index, row in csv.iterrows():
        #     if index == csv.shape[0]:
        #         break
        #
        #     csv.iat[index, 5] = math.ceil(row['amount'] * self.currencies[row['currencycode']])

        return csv

    def get_dataset(self):
        if self.csv is None:
            self.csv = self.preprocess_csv()
        transactions = self.csv

        # all
        model_variables = ['issuercountrycode', 'txvariantcode', 'bin', 'amount', 'currencycode',
                           'shoppercountrycode', 'simple_journal', 'cardverificationcodesupplied' , 'cvcresponsecode',
                           'accountcode']

        # Transactions with only the relevant columns/columsn in model_variables
        transactions_data_relevant = transactions[model_variables]

        # Get the one hot encoded thingy.
        return pd.get_dummies(transactions_data_relevant)


    def get_classifiers(self):
        return [KNeighborsClassifier(), RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier()]

    def run_classifier(self, training_features, training_target, validation_features, validation_target, test_features,
                       test_target, list_of_classifiers=None, label="Original"):

        list_roc = [] # List for false positive rate, true positve rate and auc.
        list_scores = [] # List for accuracy and recall, TP, FP, FN and TN

        # Get list of classifiers
        if list_of_classifiers is None:
            list_of_classifiers = self.get_classifiers()

        for clf in list_of_classifiers:
            # print("Name = " + clf.__class__.__name__)
            clf.fit(training_features, training_target)
            # print(label)
            if not (validation_features is None or validation_target is None):
                print('Validation Results')
                print(clf.score(validation_features, validation_target))
                print(recall_score(validation_target, clf.predict(validation_features)))
            print('\nTest Results')
            accuracy = clf.score(test_features, test_target)
            recall = recall_score(test_target, clf.predict(test_features))
            print(accuracy)
            print(recall)



            actual = test_target
            predictions = clf.predict(test_features)
            conf_matrix = confusion_matrix(actual, predictions)
            tn, fp, fn, tp = confusion_matrix(actual, predictions).ravel()
            print(tn, fp, fn, tp)
            print(conf_matrix)

            Y_score = clf.predict_proba(test_features)[:, 1]
            fpr, tpr, _ = roc_curve(test_target, Y_score)

            roc_auc = auc(fpr, tpr)

            list_roc.append([fpr, tpr, roc_auc])
            list_scores.append([accuracy, recall, conf_matrix[0][0], conf_matrix[0][1],
                                conf_matrix[1][0], conf_matrix[1][1]])


        return list_roc, list_scores


    def imbalance_task(self):
        transactions = self.get_dataset()


        # Split in into two datasets, second being test set.
        tv_features, test_features, \
        tv_target, test_target = train_test_split(transactions.drop(['simple_journal'], axis=1),
                                                  transactions['simple_journal'],
                                                  test_size=0.1)


        #Split the first dataset into training set and validation set.
        training_features, validation_features, \
        training_target, validation_target = train_test_split(tv_features,
                                                              tv_target,
                                                              test_size=0.1)
        row, _ = transactions.shape

        # Smote
        print("Smoting")
        sm = SMOTE()
        training_features_smoted, training_target_smoted = sm.fit_sample(training_features, training_target)

        # Undersample
        print("Undersampling")
        rus = RandomUnderSampler(return_indices=True)
        training_features_undersampled, training_target_undersampled, _ = rus.fit_sample(training_features,
                                                                                         training_target)

        # Smote Tomek
        print("Tomek Smoting")
        smt = SMOTETomek()
        training_features_tomek, training_target_tomek = smt.fit_sample(training_features, training_target)

        print("Length dataset: " + str(row))
        print("Length training set: " + str(len(training_features)))
        print("Length smoted training set: " + str(len(training_features_smoted)))
        print("Difference original - smoted: " + str((len(training_features_smoted) - len(training_features)) / len(
            training_features)))
        print("Length undersampled training set: " + str(len(training_features_undersampled)))
        print("Difference original - undersampled: " + str((len(training_features_undersampled) - len(
            training_features)) / len(training_features)))
        print("Length Tomeksmoted training set: " + str(len(training_features_tomek)))
        print("Difference Original - Tomeksmoted: " + str((len(training_features_tomek) - len(training_features)) / len(
            training_features)))
        print("Length validation set: " + str(len(validation_features)))
        print("Length testing set: " + str(len(test_features)))
        print("\n\n")


        list_unsmoted, _ = self.run_classifier(training_features, training_target, validation_features,
                                              validation_target,
                                       test_features, test_target, label="Unsmoted")

        list_smoted, _ = self.run_classifier(training_features_smoted, training_target_smoted, validation_features,
                                          validation_target, test_features, test_target, label="Smoted")

        list_tomek, _ = self.run_classifier(training_features_tomek, training_target_tomek, validation_features,
                                         validation_target, test_features, test_target, label="Tomek")

        list_undersampled, _ = self.run_classifier(training_features_undersampled, training_target_undersampled,
                                                validation_features, validation_target, test_features, test_target,
                                                label="Undersampled")

        # Make the plots
        list_of_classifiers = self.get_classifiers()
        for i in range(0, len(list_of_classifiers)):
            plt.figure(figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([-0.01, 1.0])
            plt.ylim([0.0, 1.01])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.grid(True)
            plt.plot(list_smoted[i][0], list_smoted[i][1], label='AUC smoted = {0}'.format(list_smoted[i][2]))
            plt.plot(list_unsmoted[i][0], list_unsmoted[i][1], label='AUC unsmoted = {0}'.format(list_unsmoted[i][2]))
            plt.plot(list_undersampled[i][0], list_undersampled[i][1], label='AUC undersampled = {0}'.format(
                list_undersampled[i][2]))
            plt.plot(list_tomek[i][0], list_tomek[i][1], label='AUC Tomek = {0}'.format(list_tomek[i][2]))
            plt.legend(loc="lower right", shadow=True, fancybox=True)
            plt.title(list_of_classifiers[i].__class__.__name__)
            plt.show()

        pass

    sns.set()

    def plotClassificationData(x, y, title=""):
        palette = sns.color_palette()
        plt.scatter(x[y == 0, 0], x[y == 0, 1], label="Class #0", alpha=0.5,
                    facecolor=palette[0], linewidth=0.15)
        plt.scatter(x[y == 1, 0], x[y == 1, 1], label="Class #1", alpha=0.5,
                    facecolor=palette[2], linewidth=0.15)
        plt.title(title)
        plt.legend()
        plt.show()

    def linePlot(x, title=""):
        palette = sns.color_palette()
        plt.plot(x, alpha=0.5, label=title, linewidth=0.2)
        plt.legend()
        plt.show()

    def savePlotClassificationData(x, y):
        palette = sns.color_palette()
        plt.scatter(x[y == 0, 0], x[y == 0, 1], label="Class #0", alpha=0.5,
                    facecolor=palette[0], linewidth=0.15)
        plt.scatter(x[y == 1, 0], x[y == 1, 1], label="Class #1", alpha=0.5,
                    facecolor=palette[2], linewidth=0.15)

        plt.legend()
        # plt.show()
        filePath = "C://Users//kw//Desktop" + str(datetime.datetime.now(datetime.timezone.utc).timestamp()) + ".png"
        plt.savefig(filePath)

    def plotHistogram(x, bins=10):
        plt.hist(x, bins=bins)
        plt.show()


    def classification_task(self, white_box=True, black_box=False):
        if white_box == black_box:
            return False


        transactions = self.get_dataset()
        dataset_features = transactions.drop(['simple_journal'], axis=1)
        dataset_target = transactions['simple_journal']
        kfold = KFold(n_splits=10, shuffle=True)
        i = 0

        list_accuracy = []
        list_recall = []
        list_TP = []
        list_FP = []
        list_FN = []
        list_TN = []

        for train_index, test_index in kfold.split(dataset_features, dataset_target):
            training_features = dataset_features.iloc[train_index]
            test_features = dataset_features.iloc[test_index]
            training_target = dataset_target.iloc[train_index]
            test_target = dataset_target.iloc[test_index]

            classifier = []

            if white_box:
                classifier.append(DecisionTreeClassifier())

            if black_box:
                # Smote
                print("Smote")
                sm = SMOTE()
                training_features, training_target = sm.fit_sample(training_features, training_target)

                classifier.append(RandomForestClassifier())

            _, scores = self.run_classifier(training_features, training_target, None, None,
                                            test_features, test_target, classifier, label="Whitebox classification")
            list_accuracy.append(scores[0][0])
            list_recall.append(scores[0][1])
            list_TP.append(scores[0][2])
            list_FP.append(scores[0][3])
            list_FN.append(scores[0][4])
            list_TN.append(scores[0][5])

        average_accuracy = sum(list_accuracy)/len(list_accuracy)
        average_recall = sum(list_recall) / len(list_recall)
        sum_TP = sum(list_TP)
        sum_FP = sum(list_FP)
        sum_FN = sum(list_FN)
        sum_TN = sum(list_TN)

        print(average_accuracy)
        print(average_recall)
        print(sum_TP)
        print(sum_FP)
        print(sum_FN)
        print(sum_TN)




if __name__ == "__main__":
    a = FraudDetection()

    a.imbalance_task()

    # One should be true and the other should be false.
    a.classification_task(white_box=False, black_box=True)