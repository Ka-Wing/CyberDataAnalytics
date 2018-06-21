import ipaddress
import math

import numpy
import pandas as pd
import random
from os import listdir
from os.path import isfile, join
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import time
from assignment3.CountMin import Sketch


class CountMinSketch():
    def __init__(self, delta, epsilon):
        self.w = int(np.ceil(2 / epsilon))
        self.d = int(np.ceil(np.log(1 / delta)))
        self.count_array = np.zeros((self.d, self.w))
        self.hash_functions = []

        print("w: ", self.w)
        print("d: ", self.d)

        time.sleep(2)

        for i in range(0, self.d):
            self.hash_functions.append(self.pairwise_indep_hash())

    # returns a hash function from a family of pairwise independent hash functions
    def pairwise_indep_hash(self):
        # the formula: h(x) = ((ax+b) % p) % m with p = prime;, a > 0; a, b < p
        p = pow(2, 61) - 1  # some big random prime
        a = random.randrange(0, p)
        b = random.randrange(0, p)

        # returns a hash function
        return lambda x: ((a * x + b) % p) % self.w

    # updates the counter array
    def update(self, ip):
        # [i, h_i(element)]

        # Convert IP to integer for hash function compatibility
        ip_int = int(ipaddress.IPv4Address(ip))

        for j in range(len(self.hash_functions)):
            k = self.hash_functions[j](ip_int)
            self.count_array[j][k] += 1

    # estimates the number of occurrences of ip
    def estimate(self, ip):
        # min(h_1(item), h_2(item), ...) = the estimate

        # Convert IP to integer for hash function compatibility
        ip_int = int(ipaddress.IPv4Address(ip))

        # Find min_j{ CM[j, h_j(ip)]]}
        list = []
        for j in range(0, self.d):
            list.append(self.count_array[j][self.hash_functions[j](ip_int)])

        return min(list)


class MinWiseSampling():
    k = 0  # The size that the final subset should have.
    i = 0  # The count of arrivals.
    n = 0  # After how many arrivals the algorithm should reset. n should be a float so decimals will not be neglected.
    temp = 0  # Used to determine when to start a new batch of arrivals.
    item = None  # Used to store the item with the lowest rank.
    item_value = 2  # Used to store the lowest rank
    list_of_items = []  # The subset of data.

    def __init__(self, size, k):
        self.k = k
        self.n = size / k
        self.list_of_items = []
        self.temp = 0

    # For every item in the stream. use input(item).
    def input(self, item):
        random_number = random.random()
        if random_number < self.item_value:
            self.item = item
            self.item_value = random_number

        self.i = self.i + 1

        if self.i >= self.temp:
            self.list_of_items.append(item)

            # Resets
            self.temp = self.temp + self.n
            self.item_value = 2
            item = None

    # Retrieving the subset dataset.
    def get_dataframe(self):
        print("Length: ", len(self.list_of_items))
        df = pd.concat(self.list_of_items, axis=1)
        print("Transposing")
        df = df.transpose()
        print(df.iloc[0])
        print(df.iloc[1])
        return df


class sketching_task():
    df = None
    # top ten most frequent in descending order
    ip_addresses = ["208.88.186.6", "78.175.28.225", "82.113.63.230", "195.168.45.2", "82.150.185.24",
                    "213.137.179.195", "62.180.140.208", "81.208.118.74", "88.255.232.197", "62.168.4.186"]

    def load_df(self, fileName):
        self.df = pd.read_csv(fileName)

    def cmsketch(self, delta=0.01, epsilon=0.0000001):
        cms = CountMinSketch(delta, epsilon)
        t = time.time()
        for i in range(self.df.shape[0]):
            print(i + 1, "/", self.df.shape[0])
            cms.update(self.df.ip.iloc[i])

        print("Time: ", time.time() - t)
        for ip in self.ip_addresses:
            print(ip + ":", int(cms.estimate(ip)))


class sampling_task():
    df = None

    # top ten most frequent in descending order
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

    # Get all file names that is generated the minwise sampling.
    def __get_all_file_names(self, path="./mws/20"):
        return [f for f in listdir(path) if isfile(join(path, f))]

    # def hoi(self):
    #     files = self.__get_all_file_names()
    #     # files = ["mws_sample_274565.1.csv"]
    #     for file in files:
    #         print(file)
    #         file = "mws/" + file
    #         df = pd.read_csv(file)
    #         print(df.ip.value_counts())
    #         print(df.shape[0])

    # Parsing the line.
    def __parse_line(self, line):
        # First replace all double tabs to one tabs.
        line = line.replace('\t\t', '\t')

        # Split on space
        line = line.split(' ')
        date = line[0]

        # Then split on tabs
        line = line[1].split('\t')
        try:
            flow_start = line[0]
            duration = line[1]
            protocol = line[2]
            src_ip_port = line[3].split(':')
            src_ip = src_ip_port[0]

            # If there is no port given.
            src_port = "NA"
            if len(src_ip_port) == 2:
                src_port = src_ip_port[1]

            dst_ip_port = line[5].split(':')
            dst_ip = dst_ip_port[0]

            # If there is no port given.
            dst_port = "NA"
            if len(dst_ip_port) == 2:
                dst_port = dst_ip_port[1]

            flags = line[6]
            tos = line[7]
            packets = line[8]
            bytes = line[9]
            flows = line[10]
            label = line[11].replace('\n', '')
        except Exception as e:
            print(e)
            print(line)
            exit(0)

        return [date, flow_start, duration, protocol, src_ip, src_port, dst_ip, dst_port, flags, tos, packets, bytes, \
                flows, label]

    def load_df(self, fileName):
        self.df = pd.read_csv(fileName)

    def preprocess(self):
        HOST_IP = "147.32.84.229"

        dataframe_list = []
        headers = ['date', 'flow start', 'durat', 'prot', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'flags', 'tos',
                   'packets', 'bytes', 'flows', 'label']

        with open("capture20110817.pcap.netflow.labeled", "r") as file:
            i = 0
            a = file.readlines()
            for line in a:
                if i == 0:
                    i = i + 1
                else:
                    dataframe_list.append(self.__parse_line(line))
            self.df = pd.DataFrame(dataframe_list, columns=headers)
            self.df = self.df[(self.df['src_ip'] == HOST_IP) | (self.df['dst_ip'] == HOST_IP)]

            self.df['ip'] = self.df['src_ip'].map(str) + self.df['dst_ip']
            self.df['ip'] = self.df['ip'].map(lambda x: x.replace(HOST_IP, ""))

            self.df.to_csv('preprocessed2.csv', sep=',', index=False)

    # Sample the dataset using Min-Wise Sampling
    def minwise_sampling(self, size, k):
        mws = MinWiseSampling(size, k)

        for i in range(self.df.shape[0]):
            print(i + 1, "/", self.df.shape[0])
            mws.input(self.df.iloc[i])

        return mws.get_dataframe()

    def task(self):
        files_20 = self.__get_all_file_names("./mws/20")
        files_10 = self.__get_all_file_names("./mws/10")

        differences_20 = []
        differences_10 = []

        for file in files_20:
            csv = pd.read_csv("./mws/20/" + file, header=0, index_col=0)
            print(file)
            size = csv.shape[0]

            list_of_distributions = []

            for ip in self.ip_addresses:
                difference = math.fabs(csv['ip'].value_counts()[ip]/size * 100 - self.ip_frequencies[ip])
                list_of_distributions.append(difference)

            differences_20.append(list_of_distributions)

        for file in files_10:
            csv = pd.read_csv("./mws/10/" + file, header=0, index_col=0)
            print(file)
            size = csv.shape[0]

            list_of_distributions = []

            for ip in self.ip_addresses:
                difference = self.ip_frequencies[ip]
                try:
                    difference = math.fabs(csv['ip'].value_counts()[ip]/size * 100 - self.ip_frequencies[ip])
                except:
                    pass
                list_of_distributions.append(difference)

            differences_10.append(list_of_distributions)


        array_20 = numpy.array(differences_20).transpose()
        array_10 = numpy.array(differences_10).transpose()
        columns_20 = ["1/20", "2/20", "3/20", "4/20", "5/20", "6/20", "7/20", "8/20", "9/20", "10/20", "11/20",
                      "12/20", "13/20", "14/20", "15/20", "16/20", "17/20", "18/20", "19/20"]
        columns_10 = ["1000", "10000", "100000", "10000000"]
        df_20 = pd.DataFrame(array_20, columns=columns_20, index=self.ip_addresses, dtype=float)
        df_10 = pd.DataFrame(array_10, columns=columns_10, index=self.ip_addresses, dtype=float)


        self.heatmap(df_20)
        self.heatmap(df_10)




    def heatmap(self, dataframe):
        sns.set(font_scale=1.8)
        sns.heatmap(data=dataframe, linewidths=1.0, vmax=0.075, cmap='coolwarm')
        plt.show()




if __name__ == "__main__":
    # Sampling Task
    sampling_task().task()

    exit(0)

    # Sketching task
    sketching = sketching_task()
    sketching.load_df("preprocessed2.csv")
    sketching.cmsketch(delta=0.01, epsilon=0.0000001)


    # s.hoi()
    #
    # exit(0)
    #
    # size = s.df.shape[0]
    # print(size)
    # i = math.ceil(size / 20)
    #
    # while not i > size:
    #     k = int(i)
    #     df = s.minwise_sampling(size, k)
    #     df.to_csv('mws/mws_sample_' + str(i) + '.csv', sep=',', index=False)
    #
    #     print(df.shape[0])


    # s.heatmap()