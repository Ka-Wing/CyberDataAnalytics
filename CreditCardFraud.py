import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, roc_curve, auc
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import datetime

from sklearn.tree import DecisionTreeClassifier


class FraudDetection:

    reader = None
    mxn_to_usd_multiplier = 0.062
    aud_to_usd_multiplier = 0.735
    gbp_to_usd_multiplier = 1.550
    nzd_to_usd_multiplier = 0.677
    sek_to_usd_multiplier = 0.118
    list = []

    def smote(self):
        transactions = pd.read_csv("C:\\Users\\kw\\Dropbox\\TU Delft\\Y2\\Q4\\CS4035 Cyber Data Analytics\\Week 1 - "
                                "Credit Card Fraud\\data_for_student_case.csv(1)\\hallo.csv")
        model_variables = ['issuercountrycode', 'txvariantcode', 'bin', 'amount', 'currencycode',
                           'shoppercountrycode', 'shopperinteraction', 'simple_journal',
                           'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']

        # Transactions with only the relevant columns/columsn in model_variables
        transactions_data_relevant = transactions[model_variables]

        # Get the one hot encoded thingy.
        transactions_relevant_encoded = pd.get_dummies(transactions_data_relevant)

        # Split in into two datasets, second being test set.
        tv_features, test_features, \
        tv_target, test_target = train_test_split(transactions_relevant_encoded.drop(['simple_journal'], axis=1),
                                                        transactions_relevant_encoded['simple_journal'],
                                                        test_size=0.1)


        #Split the first dataset into training set and validation set.
        training_features, validation_features, \
        training_target, validation_target = train_test_split(tv_features,
                                                              tv_target,
                                                              test_size=0.1)

        print(type(transactions_relevant_encoded))
        row, _ = transactions_relevant_encoded.shape

        # Smote
        sm = SMOTE()
        training_features_smoted, training_target_smoted = sm.fit_sample(training_features, training_target)

        # Undersample
        rus = RandomUnderSampler(return_indices=True)
        training_features_undersampled, training_target_undersampled, _ = rus.fit_sample(training_features,
                                                                                         training_target)

        print("Length dataset: " + str(row))
        print("Length training set: " + str(len(training_features)))
        print("Difference: " + str((len(training_features_smoted) - len(training_features)) / len(training_features)))
        print("Length smoted training set: " + str(len(training_features_smoted)))
        print("Length undersampled training set: " + str(len(training_features_undersampled)))
        print("Difference: " + str((len(training_features_undersampled) - len(training_features)) / len(training_features)))
        print("Length validation set: " + str(len(validation_features)))
        print("Length testing set: " + str(len(test_features)))


        classifiers = [KNeighborsClassifier(n_neighbors=1), RandomForestClassifier(),
                       LogisticRegression(), DecisionTreeClassifier(), AdaBoostClassifier()]

        list = []

        for clf_rf in classifiers:
            print("Name = " + clf_rf.__class__.__name__)
            clf_rf.fit(training_features, training_target)
            print("UNSMOTED!")
            print('Validation Results')
            print(clf_rf.score(validation_features, validation_target))
            print(recall_score(validation_target, clf_rf.predict(validation_features)))
            print('\nTest Results')
            print(clf_rf.score(test_features, test_target))
            print(recall_score(test_target, clf_rf.predict(test_features)))

            actual = validation_target
            predictions = clf_rf.predict(validation_features)
            # print(type(actual))
            # print(type(predictions))
            # print(len(actual))
            # print(len(predictions))
            # print(actual)
            # print(predictions)
            print(confusion_matrix(actual, predictions))

            actual = test_target
            predictions = clf_rf.predict(test_features)

            Y_score = clf_rf.predict_proba(test_features)[:, 1]
            fpr, tpr, _ = roc_curve(test_target, Y_score)

            roc_auc = auc(fpr, tpr)

            list.append([fpr, tpr, roc_auc])

            print(confusion_matrix(actual, predictions))

        classifiers = [KNeighborsClassifier(n_neighbors=1), RandomForestClassifier(),
                       LogisticRegression(), DecisionTreeClassifier(), AdaBoostClassifier()]

        list2 = []

        for clf_rf in classifiers:
            print("Name = " + clf_rf.__class__.__name__)
            clf_rf.fit(training_features_undersampled, training_target_undersampled)
            print("UNDERSAMPLED!")
            print('Validation Results')
            print(clf_rf.score(validation_features, validation_target))
            print(recall_score(validation_target, clf_rf.predict(validation_features)))
            print('\nTest Results')
            print(clf_rf.score(test_features, test_target))
            print(recall_score(test_target, clf_rf.predict(test_features)))

            actual = validation_target
            predictions = clf_rf.predict(validation_features)
            # print(type(actual))
            # print(type(predictions))
            # print(len(actual))
            # print(len(predictions))
            # print(actual)
            # print(predictions)
            print(confusion_matrix(actual, predictions))

            actual = test_target
            predictions = clf_rf.predict(test_features)

            Y_score = clf_rf.predict_proba(test_features)[:, 1]
            fpr, tpr, _ = roc_curve(test_target, Y_score)

            roc_auc = auc(fpr, tpr)

            list2.append([fpr, tpr, roc_auc])

            print(confusion_matrix(actual, predictions))

        classifiers3 = [KNeighborsClassifier(n_neighbors=1), RandomForestClassifier(),
                       LogisticRegression(), DecisionTreeClassifier(), AdaBoostClassifier()]

        i = 0
        for clf_rf in classifiers3:
            print("Name = " + clf_rf.__class__.__name__)

            clf_rf.fit(training_features_smoted, training_target_smoted)
            print("SMOTED!")
            print('Validation Results')
            score = clf_rf.score(validation_features, validation_target)
            print(score)
            print(recall_score(validation_target, clf_rf.predict(validation_features)))
            print('\nTest Results')
            print(clf_rf.score(test_features, test_target))
            print(recall_score(test_target, clf_rf.predict(test_features)))

            actual = test_target
            predictions = clf_rf.predict(test_features)
            print(confusion_matrix(actual, predictions))

            false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)

            # get roc/auc info
            Y_score = clf_rf.predict_proba(test_features)[:, 1]
            fpr, tpr, _ = roc_curve(test_target, Y_score)

            roc_auc = auc(fpr, tpr)

            # make the plot
            plt.figure(figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.grid(True)
            plt.plot(fpr, tpr, label='AUC smoted = {0}'.format(roc_auc))
            plt.plot(list[i][0], list[i][1], label='AUC unsmoted = {0}'.format(list[i][2]))
            plt.plot(list2[i][0], list2[i][1], label='AUC undersampled = {0}'.format(list2[i][2]))
            plt.legend(loc="lower right", shadow=True, fancybox=True)
            plt.title(clf_rf.__class__.__name__)
            plt.show()

            # roc_auc = auc(false_positive_rate, true_positive_rate)
            #
            # plt.title('ROC')
            # plt.plot(false_positive_rate, true_positive_rate, 'b',
            #          label='AUC = %0.2f' % roc_auc)
            # plt.legend(loc='lower right')
            # plt.plot([0, 1], [0, 1], 'r--')
            # plt.xlim([-0.1, 1.2])
            # plt.ylim([-0.1, 1.2])
            # plt.ylabel('True Positive Rate')
            # plt.xlabel('False Positive Rate')
            # plt.show()

            actual = test_target
            predictions = clf_rf.predict(test_features)
            print(confusion_matrix(actual, predictions))
            i = i + 1



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


    def load_data(self, path):
        self.reader = csv.reader(open(path, 'rt', encoding="ascii"), delimiter=',', quotechar='|')

    def load_data_in_list(self):
        if self.reader is None:
            return False
        else:
            for row in self.reader:
                self.list.append(row)


    def print_list(self):
        if self.list == []:
            return False
        else:
            i = 0
            for row in self.reader:
                print(row[6])
                #print(', '.join(row))

    def percentage_chargeback_per_country(self, row_number):
        countries = []
        amount = []
        i = 0
        for row in self.list:
            if row[9] != "Refused":
                if not row[row_number] in countries:
                    countries.append(row[row_number])
                    if row[9] == "Chargeback":
                        amount.append([1, 1])
                    elif row[9] == "Settled":
                        amount.append([0, 1])
                elif row[row_number] in countries:
                    index = countries.index(row[row_number])
                    list = amount[index]
                    if row[9] == "Chargeback":
                        list = [list[0] + 1, list[1] + 1]
                    elif row[9] == "Settled":
                        list = [list[0], list[1] + 1]
                    amount[index] = list
                    pass

            i = i + 1
            if i % 10000 == 0:
                print(i)

        if len(countries) != len(amount):
            print("WRONG!")
            return False
        else:
            with open("a.txt", "w") as file:
                for i in range(len(countries)):
                    file.write(countries[i] + " " + str(float(amount[i][0])/float(amount[i][1])))
                    file.write("\n")

    def changecurrency(self):
        with open('changedcurrency.csv', 'w') as file:
            file.write("txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,currencycode,shoppercountrycode,"
                       "shopperinteraction,simple_journal,cardverificationcodesupplied,cvcresponsecode,creationdate,"
                       "accountcode,mail_id,ip_id,card_id")
            file.write('\n')

            if self.list == []:
                return False
            else:
                for row in self.list:
                    conversion = 0
                    if row[6] == "SEK":
                        conversion = float(row[5]) * self.sek_to_usd_multiplier
                    elif row[6] == "NZD":
                        conversion = float(row[5]) * self.nzd_to_usd_multiplier
                    elif row[6] == "AUD":
                        conversion = float(row[5]) * self.aud_to_usd_multiplier
                    elif row[6] == "GBP":
                        conversion = float(row[5]) * self.gbp_to_usd_multiplier
                    elif row[6] == "MXN":
                        conversion = float(row[5]) * self.mxn_to_usd_multiplier

                    row[5] = str(math.floor(conversion))
                    line = ', '.join(row)
                    file.write(line)
                    file.write('\n')

    def print_data(self):
        if self.reader is None:
            return False
        else:
            i = 0
            for row in self.reader:
                print(', '.join(row))

if __name__ == "__main__":
    a = FraudDetection()
    # a.load_data('C:\\Users\\kw\\Dropbox\\TU Delft\\Y2\\Q4\\CS4035 Cyber Data Analytics\\Week 1 - Credit Card '
    #               'Fraud\\data_for_student_case.csv(1)\\data_for_student_case.csv')
    # a.load_data_in_list()
    # a.percentage_chargeback_per_country(4)
    a.smote()
    # a.hallo()