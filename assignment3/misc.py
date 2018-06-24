import ipaddress
import pandas as pd
import random
import numpy as np
import time



# Class initiated to perform count min sketch.
class CountMinSketch():
    def __init__(self, delta, epsilon):
        self.w = int(np.ceil(2 / epsilon))
        self.d = int(np.ceil(np.log(1 / delta)))
        self.count_array = np.zeros((self.d, self.w)) # The table used to store the values.
        self.hash_functions = []

        print("w: ", self.w)
        print("d: ", self.d)

        time.sleep(2)

        for i in range(0, self.d):
            self.hash_functions.append(self.pairwise_indep_hash())

    # Returns a hash function from a family of pairwise independent hash functions
    def pairwise_indep_hash(self):
        # The formula: h(x) = ((ax+b) % p) % m with p = prime;, a > 0; a, b < p
        p = pow(2, 61) - 1  # some big random prime
        a = random.randrange(0, p)
        b = random.randrange(0, p)

        # Returns a hash function
        return lambda x: ((a * x + b) % p) % self.w

    # Updates the counter array
    def update(self, ip):
        # Convert IP to integer for hash function compatibility
        ip_int = int(ipaddress.IPv4Address(ip))

        for j in range(len(self.hash_functions)):
            k = self.hash_functions[j](ip_int)
            self.count_array[j][k] += 1

    # Estimates the number of occurrences of ip
    def estimate(self, ip):
        # Convert IP to integer for hash function compatibility
        ip_int = int(ipaddress.IPv4Address(ip))

        # Find min_j{ CM[j, h_j(ip)]]}
        list = []
        for j in range(0, self.d):
            list.append(self.count_array[j][self.hash_functions[j](ip_int)])

        return min(list)


# Class initiated to perform count min sketch.
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
            pass

        self.i = self.i + 1

        if self.i >= self.temp:
            self.list_of_items.append(item)

            # Resets
            self.temp = self.temp + self.n
            self.item_value = 2
            self.item = None
        pass

    # Retrieving the subset dataset.
    def get_dataframe(self):
        df = pd.concat(self.list_of_items, axis=1)
        df = df.transpose()
        return df