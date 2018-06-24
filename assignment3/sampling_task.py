import numpy as np

from assignment3.task import task
from assignment3.misc import MinWiseSampling
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import time


class sampling_task(task):
    # Top ten most frequent in descending order
    ip_addresses = ["208.88.186.6", "78.175.28.225", "82.113.63.230", "195.168.45.2", "82.150.185.24",
                    "213.137.179.195", "62.180.140.208", "81.208.118.74", "88.255.232.197", "62.168.4.186"]

    # Distributions of the top ten most frequent IP addresses
    ip_frequencies = {"208.88.186.6": 0.221587156,
                      "78.175.28.225": 0.198860268,
                      "82.113.63.230": 0.198150053,
                      "195.168.45.2": 0.164223617,
                      "82.150.185.24": 0.168976596,
                      "213.137.179.195": 0.166736687,
                      "62.180.140.208": 0.143572743,
                      "81.208.118.74": 0.14122357,
                      "88.255.232.197": 0.129860126,
                      "62.168.4.186": 0.130187917}

    def __init__(self, fileName):
        self.load_df(fileName)

    # Get all file names that is generated the minwise sampling.
    def __get_all_file_names(self, path="./mws/20"):
        return [f for f in listdir(path) if isfile(join(path, f))]

    # Sample the dataset using Min-Wise Sampling
    def minwise_sampling(self, k):
        mws = MinWiseSampling(self.df.shape[0], k)

        size = self.df.shape[0]
        for i in range(size):
            print(i + 1, "/", self.df.shape[0])
            mws.input(self.df.iloc[i])

        return mws.get_dataframe()

    # This method creates the dataset, then calculates the distribution per file found, and calculates the difference
    # w.r.t. original dataset. It then shows a heatmap of it.
    def task(self, create_minwise_sampling_dataset=False):
        if create_minwise_sampling_dataset:
            print("Creates dataset, might take long (one hour). Consider using our provided dataset.")
            self.create_datasets()
            print("Creating dataset done.")


        files_20 = self.__get_all_file_names("./mws/20")
        files_10 = self.__get_all_file_names("./mws/10")

        differences_20 = []
        differences_10 = []


        for i in range(0, len(files_20)):
            csv = pd.read_csv("./mws/20/" + files_20[i], header=0, index_col=0)
            print("Calculating for file " + str(i+1) + "/" + str(len(files_20)))
            size = csv.shape[0]

            list_of_distributions = []

            # Calculates the difference in distribution
            for ip in self.ip_addresses:
                difference = math.fabs(csv['ip'].value_counts()[ip]/size * 100 - self.ip_frequencies[ip])
                list_of_distributions.append(difference)

            differences_20.append(list_of_distributions)

        for i in range(0, len(files_10)):
            csv = pd.read_csv("./mws/10/" + files_10[i], header=0, index_col=0)
            print("Calculating for file " + str(i + 1) + "/" + str(len(files_10)))
            size = csv.shape[0]

            list_of_distributions = []

            # Calculates the difference in distribution
            for ip in self.ip_addresses:
                difference = self.ip_frequencies[ip]
                try:
                    difference = math.fabs(csv['ip'].value_counts()[ip]/size * 100 - self.ip_frequencies[ip])
                except:
                    pass # If there are any 0 values (and there are), then division by 0 will not work.
                list_of_distributions.append(difference)

            differences_10.append(list_of_distributions)


        array_20 = np.array(differences_20).transpose()
        array_10 = np.array(differences_10).transpose()
        column_names_20 = ["1/20", "2/20", "3/20", "4/20", "5/20", "6/20", "7/20", "8/20", "9/20", "10/20", "11/20",
                      "12/20", "13/20", "14/20", "15/20", "16/20", "17/20", "18/20", "19/20"]
        column_names_10 = ["1000", "10000", "100000", "10000000"]
        df_20 = pd.DataFrame(array_20, columns=column_names_20, index=self.ip_addresses, dtype=float)
        df_10 = pd.DataFrame(array_10, columns=column_names_10, index=self.ip_addresses, dtype=float)

        self.heatmap(df_20)
        self.heatmap(df_10)


    def heatmap(self, dataframe):
        sns.set(font_scale=1.8)
        sns.heatmap(data=dataframe, linewidths=1.0, vmax=0.075, cmap='coolwarm')
        plt.show()

    def create_datasets(self):
        size = self.df.shape[0]
        print(size)
        i = math.ceil(size / 20)

        while not i > size:
            k = int(i)
            df = self.minwise_sampling(k)
            df.to_csv('datasets/mws/20/mws_sample_' + str(i) + '.csv', sep=',', index=False)


        list_of_size = [1000, 10000, 100000, 1000000]

        for i in list_of_size:
            k = int(i)
            df = self.minwise_sampling(k)
            df.to_csv('datasets/mws/10/mws_sample_' + str(i) + '.csv', sep=',', index=False)

    @staticmethod
    def run_task(preprocessing=False, create_minwise_sampling_dataset=False):
        print("Preprocessing. Wait til it says it is done.")
        if preprocessing:
            sampling_task.preprocess(input="datasets/capture20110817.pcap.netflow.labeled",
                                     output="datasets/preprocessed_task_1.csv",
                                     list_of_ips=["147.32.84.229"], task_name="sampling")
        print("Done.")
        time.sleep(3)



        sampling = sampling_task("datasets/preprocessed_task_1.csv")
        sampling.task(create_minwise_sampling_dataset=create_minwise_sampling_dataset)

if __name__ == "__main__":
    # Set 'preprocessing' to True if you want to create the dataset, set to False to use the provided dataset.

    # Set create_'minwise_sampling_dataset' to True on if you want to create the minwise-sampling datasets,
    # Set it to False use the provided dataset.
    sampling_task.run_task(preprocessing=True, create_minwise_sampling_dataset=True)