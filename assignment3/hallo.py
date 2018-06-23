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
#from assignment3.CountMin import Sketch


class task():
    df = None

    def load_df(self, fileName):
        self.df = pd.read_csv(fileName)

    @staticmethod
    def preprocess(input="", output="", list_of_ips=[], task_name=""):
        dataframe_list = []
        headers = ['date', 'flow start', 'durat', 'prot', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'flags', 'tos',
                   'packets', 'bytes', 'flows', 'label']

        with open(input, "r") as file:
            row_counter = 0
            lines = file.readlines()
            for line in lines:
                if row_counter == 0:
                    row_counter = row_counter + 1
                else:
                    dataframe_list.append(task.__parse_line(line))

            df = pd.DataFrame(dataframe_list, columns=headers)
            df = df[(df['src_ip'].isin(list_of_ips)) | (df['dst_ip'].isin(list_of_ips))]

            if(task_name == "sampling" or task_name == "sketching"):
                df['ip'] = df['src_ip'].map(str) + df['dst_ip']
                df['ip'] = df['ip'].map(lambda x: x.replace(list_of_ips[0], ""))
            elif(task_name== "discretization"):
                df = df[df['label'] != "Background"]

            df.to_csv(output, sep=',', index=False)

    # Parsing the line.
    @staticmethod
    def __parse_line(line):
        # First replace all double tabs to one tabs.
        line = line.replace('\t\t', '\t')
        line = line.replace('\t', " ")

        # Split on space
        line = line.split(' ')

        try:
            date = line[0]
            flow_start = line[1]
            duration = line[2]
            protocol = line[3]
            src_ip_port = line[4].split(':')
            src_ip = src_ip_port[0]

            # If there is no port given.
            src_port = "NA"
            if len(src_ip_port) == 2:
                src_port = src_ip_port[1]

            dst_ip_port = line[6].split(':')
            dst_ip = dst_ip_port[0]

            # If there is no port given.
            dst_port = "NA"
            if len(dst_ip_port) == 2:
                dst_port = dst_ip_port[1]

            flags = line[7]
            tos = line[8]
            packets = line[9]
            bytes = line[10]
            flows = line[11]
            label = line[12].replace('\n', '')
        except Exception as e:
            print(e)
            print(line)
            exit(0)

        return [date, flow_start, duration, protocol, src_ip, src_port, dst_ip, dst_port, flags, tos, packets, bytes, \
                flows, label]


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
        #random_number = 1#random.random()
        #if random_number < self.item_value:
            #self.item = item
            #self.item_value = random_number
            #pass

        #self.i = self.i + 1

        #if self.i >= self.temp:
            # self.list_of_items.append(item)

            # Resets
            #self.temp = self.temp + self.n
            #self.item_value = 2
            #self.item = None
        pass

    # Retrieving the subset dataset.
    def get_dataframe(self):
        print("Length: ", len(self.list_of_items))
        df = pd.concat(self.list_of_items, axis=1)
        print("Transposing")
        df = df.transpose()
        print(df.iloc[0])
        print(df.iloc[1])
        return df


class sketching_task(task):
    # Top ten most frequent in descending order
    ip_addresses = ["208.88.186.6", "78.175.28.225", "82.113.63.230", "195.168.45.2", "82.150.185.24",
                    "213.137.179.195", "62.180.140.208", "81.208.118.74", "88.255.232.197", "62.168.4.186"]

    def __init__(self, fileName):
        self. load_df(fileName)

    def cmsketch(self, delta=0.01, epsilon=0.0000001):
        cms = CountMinSketch(delta, epsilon)
        t = time.time()
        for i in range(self.df.shape[0]):
            print(i + 1, "/", self.df.shape[0])
            cms.update(self.df.ip.iloc[i])

        print("Time: ", time.time() - t)
        for ip in self.ip_addresses:
            print(ip + ":", int(cms.estimate(ip)))


class sampling_task(task):
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

    def __init__(self, fileName):
        self.load_df(fileName)

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

    # Sample the dataset using Min-Wise Sampling
    def minwise_sampling(self, k):
        mws = MinWiseSampling(self.df.shape[0], k)

        t = time.time()

        size = self.df.shape[0]
        for i in range(size):
            print(i + 1, "/", self.df.shape[0])
            mws.input(self.df.iloc[i])

        diff = time.time() - t
        print("Time", diff)



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


class discretization_task(task):
    packets_values = []
    duration_values = []
    bytes_values = []
    protocol_boolean = False
    packets_boolean = False
    bytes_boolean = False
    duration_boolean = False
    bins = 0

    def __init__(self, fileName, bins=4, protocol=True, packets=True, bytes=True, duration=True):
        self.load_df(fileName)
        self.protocol_boolean = protocol
        self.bins = bins

        if packets:
            self.packets_boolean = packets
            self.packets_values = self.__column_occurences_to_list('packets')

        if duration:
            self.duration_boolean = duration
            self.duration_values = self.__column_occurences_to_list('durat')

        if bytes:
            self.bytes_boolean = bytes
            self.bytes_values = self.__column_occurences_to_list('bytes')


    def get_ordinal_rank(self, p):
        return math.ceil(p/100 * self.df.shape[0])

    def get_nth_packets_percentile(self, n):
        return self.packets_values[n-1]

    def get_nth_duration_percentile(self, n):
        return self.duration_values[n-1]

    def get_nth_bytes_percentile(self, n):
        return self.bytes_values[n-1]

    def get_packets_mapping(self, v):
        percentile = 100 / self.bins

        for i in range(self.bins):
            if v <= self.get_nth_packets_percentile(self.get_ordinal_rank((i + 1) * percentile)):
                return i

        return self.bins - 1

    def get_protocol_mapping(self, v):
        attribute_mapping_protocol = {'TCP': 0, 'ICMP': 1, 'UDP': 2}
        return attribute_mapping_protocol[v]

    def get_duration_mapping(self, v):
        percentile = 100 / self.bins

        for i in range(self.bins):
            if v <= self.get_nth_duration_percentile(self.get_ordinal_rank((i+1) * percentile)):
                return i

        return self.bins - 1


    def get_bytes_mapping(self, v):
        percentile = 100 / self.bins

        for i in range(self.bins):
            if v <= self.get_nth_bytes_percentile(self.get_ordinal_rank((i + 1) * percentile)):
                return i

        return self.bins - 1

    def __column_occurences_to_list(self, column_name):
        list = []
        list_of_indices = self.df[column_name].value_counts().sort_index().index

        for amount in list_of_indices:
            occurences = self.df[column_name].value_counts()[amount]

            for i in range(occurences):
                list.append(amount)

        return list


    def netflow_encoding(self, netflow):
        code = 0

        # Each tupple is in the form of <function_name, column_name in dataframe, size>
        attributes_tuples = []

        if self.protocol_boolean:
            attributes_tuples.append([self.get_protocol_mapping, 'prot', 3])

        if self.packets_boolean:
            attributes_tuples.append([self.get_packets_mapping, 'packets', 3])

        if self.duration_boolean:
            attributes_tuples.append([self.get_duration_mapping, 'durat', 3])

        if self.bytes_boolean:
            attributes_tuples.append([self.get_bytes_mapping, 'bytes', 3])

        space_size = 1
        for tuple in attributes_tuples:
            space_size = space_size * tuple[2]


        for tuple in attributes_tuples:
            code = code + tuple[0](netflow[tuple[1]]) * (space_size / tuple[2])
            space_size = space_size / tuple[2]

        return code

    def add_netflow_encoding_column(self):
        self.df['encoding'] = self.df.apply(lambda x: self.netflow_encoding(x), axis=1)

    def add_protocol_encoding_column(self):
        self.df['protocol'] = self.df['prot'].map(lambda x: self.get_protocol_mapping(x))

    def scatterplot(self):
        #sns.regplot(data=self.df, x="protocol", y="packets", fit_reg=False)
        sns.lmplot(x='prot', y='packets', data=self.df, fit_reg=False, hue='label')
        plt.show()
        
    def compare_hosts(self, infected_host, normal_host):        
        print("encoding", self.df['encoding'].unique())
        infected = self.df[(self.df['src_ip'] == infected_host) | (self.df['dst_ip'] == infected_host)]
        normal = self.df[(self.df['src_ip'] == normal_host) | (self.df['dst_ip'] == normal_host)]

        plt.title("Infected (" + infected_host + ") vs. Normal (" + normal_host + ")")
        plt.plot(infected['encoding'], label='infected')
        plt.plot(normal['encoding'], label='normal')
        plt.xlabel('netflow')
        plt.ylabel('encoding')
        plt.legend()
        plt.show()

    def print_encodings(self):
        for i in range(self.df.shape[0]):
            print("<" +
                  self.df.iloc[i].prot + ", " +
                  str(self.df.iloc[i].packets) + ", " +
                  str(self.df.iloc[i].durat) + ", " +
                  str(self.df.iloc[i].bytes) +
                  "> =",
                  self.netflow_encoding(self.df.iloc[i]))





if __name__ == "__main__":
    # Sampling Task
    # sampling_task.preprocess(input="capture20110817.pcap.netflow.labeled", output="preprocessed2.csv",
    #                    list_of_ips=["47.32.84.229"], task="sampling")
    # sampling = sampling_task("preprocessed2.csv")
    # sampling.minwise_sampling()

    # exit(0)

    # Sketching task
    # sketching_task.preprocess("capture20110817.pcap.netflow.labeled", "preprocessed2.csv",
    #                         list_of_ips=["147.32.84.229"], task_name="sketching")
    # sketching = sketching_task("preprocessed2.csv")
    # sketching.cmsketch(delta=0.01, epsilon=0.0001)
    # exit(0)

    # Botnet flow data discretization task
    discretization = discretization_task("preprocessed2_scen10_2.csv",
                                         bins=3,
                                         protocol=True,
                                         packets=True,
                                         duration=False,
                                         bytes=False)
    # discretization.preprocess(input="capture20110818.pcap.netflow.labeled", output="preprocessed2_scen10_2.csv",
    #                          list_of_ips=["147.32.84.205", "147.32.84.170", "147.32.84.134", "147.32.84.164",
    #                                  "147.32.87.36", "147.32.80.9", "147.32.87.11"], task="discretization")
    discretization.add_netflow_encoding_column()

    print("Swarm")
    discretization.compare_hosts("147.32.84.205", "147.32.84.170")
    discretization.compare_hosts("147.32.84.205", "147.32.84.134")
    discretization.compare_hosts("147.32.84.205", "147.32.84.164")
    discretization.compare_hosts("147.32.84.205", "147.32.87.36")
    discretization.compare_hosts("147.32.84.205", "147.32.80.9")
    discretization.compare_hosts("147.32.84.205", "147.32.87.11")
    print("Swarm")
    exit(0)

    print(discretization.df.iloc[1])
    print()
    print()
    print()
    print()
    print("encoding =", discretization.netflow_encoding(discretization.df.iloc[1]))
    discretization.add_netflow_encoding_column()
    print(discretization.df.iloc[1])

    exit(0)

    exit(0)
    print(discretization.packets_values)
    print(discretization.get_ordinal_rank(99.99))
    print(discretization.get_nth_packet_percentile(discretization.get_ordinal_rank(99.99)))

    exit(0)

    discretization.load_df("preprocessed2_scen10.csv")
    print(discretization.df['label'].value_counts())



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