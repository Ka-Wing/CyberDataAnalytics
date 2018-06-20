import ipaddress
import math
import pandas as pd
import random
from os import listdir
from os.path import isfile, join
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import time

class CountMinSketch():
    def __init__(self, delta, epsilon):
        self.w = int(np.ceil(2/epsilon))
        self.d = int(np.ceil(np.log(1/delta)))

        self.count_array = np.zeros((self.d, self.w))
        self.hash_functions = []

        print("w: ", self.w)
        print("d: ", self.d)

        time.sleep(5)

        for i in range(0, self.d):
            self.hash_functions.append(self.pairwise_indep_hash())
       
    # returns a hash function from a family of pairwise independent hash functions
    def pairwise_indep_hash(self):
        # the formula: h(x) = ((ax+b) % p) % m with p = prim;, a > 0; a, b < p
        p = pow(2, 61) - 1 #some big random prime
        a = random.randrange(0, p)
        b = random.randrange(0, p)

        # returns a hash function
        return lambda x: ((a * x + b) % p) % self.w

    # updates the counter array
    def update(self, ip):
        #[i, h_i(element)]

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
        minimum = sys.maxsize
        for j in range(0, self.d):
            a = self.count_array[j][self.hash_functions[j](ip_int)]

            if a < minimum:
                minimum = a

        return minimum


class MinWiseSampling():

    k = 0 # The size that the final subset should have.
    i = 0 # The count of arrivals.
    n = 0 # After how many arrivals the algorithm should reset. n should be a float so decimals will not be neglected.
    temp = 0 # Used to determine when to start a new batch of arrivals.
    item = None # Used to store the item with the lowest rank.
    item_value = 2 # Used to store the lowest rank
    list_of_items = [] # The subset of data.

    def __init__(self, size, k):
        self.k = k
        self.n = size/k
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



class sampling_task():
    df = None

    # top ten most frequent in descending order
    ip_addresses = ["208.88.186.6", "78.175.28.225", "82.113.63.230", "195.168.45.2",
                                          "82.150.185.24", "213.137.179.195", "62.180.140.208", "81.208.118.74",
                                          "88.255.232.197", "62.168.4.186"]



    # Get all file names that is generated the minwise sampling.
    def __get_all_file_names(self):
        print("a")
        mypath = "./mws"
        return [f for f in listdir(mypath) if isfile(join(mypath, f))]

    def hoi(self):
        files = self.__get_all_file_names()
        # files = ["mws_sample_274565.1.csv"]
        for file in files:
            print(file)
            file = "mws/" + file
            df = pd.read_csv(file)
            print(df.ip.value_counts())
            print(df.shape[0])

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
            print(i+1, "/", self.df.shape[0])
            mws.input(self.df.iloc[i])

        return mws.get_dataframe()

    def heatmap(self):
        heatmap = pd.read_csv("heatmap2.csv", header=0, index_col=0)
        columns = list(heatmap.columns)
        heatmap = heatmap[columns].astype(float)

        print(heatmap.iloc[3])

        sns.set(font_scale=1.8)
        sns.heatmap(data=heatmap, linewidths=1.0, vmax=0.075, cmap='coolwarm')
        plt.show()

    def cmsketch(self, delta, epsilon):
        cms = CountMinSketch(delta, epsilon)
        t = time.time()
        for i in range(self.df.shape[0]):
            print(i + 1, "/", self.df.shape[0])
            cms.update(self.df.iloc[i]['ip'])

        print("Time: ", time.time()-t)

        for ip in range(len(self.ip_addresses)):
            print(ip, cms.estimate(ip))


if __name__ == "__main__":
    s = sampling_task()
    s.load_df("preprocessed2.csv")

    s.cmsketch(0.00005, 0.0000005463194187598)

    exit()
    # s.hoi()
    #
    # exit(0)
    #
    # size = s.df.shape[0]
    # print(size)
    # i = math.ceil(size / 20)
    #
    # for i in [1000, 10000, 100000, 1000000]:
    #     k = int(i)
    #     df = s.minwise_sampling(size, k)
    #     df.to_csv('mws/mws_sample_' + str(i) + '.csv', sep=',', index=False)
    #
    #     print(df.shape[0])


    # s.heatmap()