import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, roc_curve, auc
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

    def read_csv(self):
        transactions = pd.read_csv("C:\\Users\\kw\\Dropbox\\TU Delft\\Y2\\Q4\\CS4035 Cyber Data Analytics\\Week 1 - "
                                "Credit Card Fraud\\data_for_student_case.csv(1)\\data_for_student_case - Copy.csv")
        return transactions

    def filter_csv(self):
        csv = self.read_csv()
        csv = csv[csv.simple_journal != 'Refused']
        csv['simple_journal'] = csv['simple_journal'].replace('Settled', 0).replace('Chargeback', 1)
        return csv


    def get_classifiers(self):
        return [KNeighborsClassifier(), RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier()]

    def run_classifier(self, training_features, training_target, validation_features, validation_target, test_features,
                       test_target, list_of_classifiers=None, label="Original"):
        list = []
        if list_of_classifiers is None:
            list_of_classifiers = self.get_classifiers()

        for clf in list_of_classifiers:
            print("Name = " + clf.__class__.__name__)
            clf.fit(training_features, training_target)
            print(label)
            print('Validation Results')
            print(clf.score(validation_features, validation_target))
            print(recall_score(validation_target, clf.predict(validation_features)))
            print('\nTest Results')
            print(clf.score(test_features, test_target))
            print(recall_score(test_target, clf.predict(test_features)))

            actual = test_target
            predictions = clf.predict(test_features)
            print(confusion_matrix(actual, predictions))

            Y_score = clf.predict_proba(test_features)[:, 1]
            fpr, tpr, _ = roc_curve(test_target, Y_score)

            roc_auc = auc(fpr, tpr)

            list.append([fpr, tpr, roc_auc])

        return list


    def smote(self):
        transactions = self.filter_csv()

        # print(transactions.issuercountrycode.value_counts())
        # print(transactions.txvariantcode.value_counts())
        # print(transactions.currencycode.value_counts())
        # print(transactions.shoppercountrycode.value_counts())
        # print(transactions.shopperinteraction.value_counts())
        # print(transactions.simple_journal.value_counts())
        # print(transactions.cardverificationcodesupplied.value_counts())
        # print(transactions.cvcresponsecode.value_counts())
        # print(transactions.accountcode.value_counts())

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
        row, _ = transactions_relevant_encoded.shape

        # Smote
        print("Smoting")
        sm = SMOTE(random_state=12)
        training_features_smoted, training_target_smoted = sm.fit_sample(training_features, training_target)

        # Undersample
        print("Undersampling")
        rus = RandomUnderSampler(return_indices=True)
        training_features_undersampled, training_target_undersampled, _ = rus.fit_sample(training_features,
                                                                                         training_target)

        # Smote Tomek
        print("Tomek Smoting")
        smt = SMOTETomek(random_state=12)
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


        list_unsmoted = self.run_classifier(training_features, training_target, validation_features, validation_target,
                                       test_features, test_target, label="Unsmoted")

        list_smoted = self.run_classifier(training_features_smoted, training_target_smoted, validation_features,
                                          validation_target, test_features, test_target, label="Smoted")

        list_tomek = self.run_classifier(training_features_tomek, training_target_tomek, validation_features,
                                         validation_target, test_features, test_target, label="Tomek")

        list_undersampled = self.run_classifier(training_features_undersampled, training_target_undersampled,
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