from assignment3.task import task
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import matplotlib.patches as mpatches
import numpy as np

class discretization_task(task):
    packets_values = [] # All values of the 'packet' column
    duration_values = [] # All values of the 'durat' column
    bytes_values = [] # All values of the 'bytes' column
    protocol_boolean = False # Whether the 'protocol' should be chosen as feature.
    packets_boolean = False # Whether the 'packets' should be chosen as feature.
    bytes_boolean = False # Whether the 'bytes' should be chosen as feature.
    duration_boolean = False # Whether the 'duration' should be chosen as feature.
    bins = 0 # Amount of bins used
    attribute_mapping_protocol = {} # The attribute mapping of the column 'prot'

    def __init__(self, fileName, bins=3, protocol=False, packets=False, bytes=False, duration=False):
        self.load_df(fileName)
        self.protocol_boolean = protocol
        self.bins = bins
        self.attribute_mapping_protocol = {'TCP': 0, 'UDP': 1, 'ICMP': 2}

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

    # Returns the attribute mapping of the column 'prot'.
    def get_protocol_mapping(self, v):
        return self.attribute_mapping_protocol[v]

    # Returns the attribute mapping of the column 'packets'.
    def get_packets_mapping(self, v):
        percentile = 100 / self.bins

        for i in range(self.bins):
            if v <= self.get_nth_packets_percentile(self.get_ordinal_rank((i + 1) * percentile)):
                return i

        return self.bins - 1

    # Returns the attribute mapping of the column 'durat'.
    def get_duration_mapping(self, v):
        percentile = 100 / self.bins

        for i in range(self.bins):
            if v <= self.get_nth_duration_percentile(self.get_ordinal_rank((i+1) * percentile)):
                return i

        return self.bins - 1

    # Returns the attribute mapping of the column 'bytes'.
    def get_bytes_mapping(self, v):
        percentile = 100 / self.bins

        for i in range(self.bins):
            if v <= self.get_nth_bytes_percentile(self.get_ordinal_rank((i + 1) * percentile)):
                return i

        return self.bins - 1

    # Add values of all rows in a certain column to a list.
    def __column_occurences_to_list(self, column_name):
        list = []
        list_of_indices = self.df[column_name].value_counts().sort_index().index

        for amount in list_of_indices:
            occurences = self.df[column_name].value_counts()[amount]

            for i in range(occurences):
                list.append(amount)

        return list

    # Plot the protocol vs packet amounts, with legitimate and botnet flows as different colors.
    def packets_visualization(self):
        sns.stripplot(data=self.df, y='packets', x='prot', hue='label', jitter=True)
        plt.show()

    # Plot the ICMP netflows bytes
    def imcp_visualization(self):
        # Only ICMP and Botnet netflows.
        df_icmp = self.df[(self.df['prot'] == "ICMP") & (self.df['label'] == "Botnet")]

        # Sort bytes on ascending order.
        df_icmp = df_icmp.sort_values('bytes', axis=0)

        # Plot
        nump = np.array(df_icmp['bytes'])
        plt.plot(nump)
        plt.axhline(y=65535, color='r', linestyle='dotted')
        plt.ylabel("Bytes")
        plt.title("ICMP packets in ascending bytes order.")
        plt.show()

    # Plot a barchart of percentages legitimate/botnet per protocol.
    def protocol_visualization(self):
        df_tcp = self.df[self.df['prot'] == "TCP"]
        df_icmp = self.df[self.df['prot'] == "ICMP"]
        df_udp = self.df[self.df['prot'] == "UDP"]

        # Count all netflows per label per protocol.
        try:
            tcp_botnet = df_tcp['label'].value_counts()['Botnet']
        except:
            tcp_botnet = 0.000001  # Set to this number because code cannot divide zero when making the plots.

        try:
            tcp_legitimate = df_tcp['label'].value_counts()['LEGITIMATE']
        except:
            tcp_legitimate = 0.000001

        try:
            icmp_botnet = df_icmp['label'].value_counts()['Botnet']
        except:
            icmp_botnet = 0.000001

        try:
            icmp_legitimate = df_icmp['label'].value_counts()['LEGITIMATE']
        except:
            icmp_legitimate = 0.000001

        try:
            udp_botnet = df_udp['label'].value_counts()['Botnet']
        except:
            udp_botnet = 0.000001

        try:
            udp_legitimate = df_udp['label'].value_counts()['LEGITIMATE']
        except:
            udp_legitimate = 0.000001

        print("tcp legitimate: ", tcp_legitimate)
        print("tcp botnet: ", tcp_botnet)
        print("icmp legitimate: ", icmp_legitimate)
        print("icmp botnet: ", icmp_botnet)
        print("udp legitimate: ", udp_legitimate)
        print("udp botnet: ", udp_botnet)

        # Data construction
        bar_index = [0, 1, 2]
        raw_data = {'greenBars': [tcp_legitimate, icmp_legitimate, udp_legitimate],
                    'redBars': [tcp_botnet, icmp_botnet, udp_botnet],
                    }
        df = pd.DataFrame(raw_data)

        # From raw value to percentage
        totals = [i + j for i, j in zip(df['greenBars'], df['redBars'])]
        greenBars = [i / j * 100 for i, j in zip(df['greenBars'], totals)]
        redBars = [i / j * 100 for i, j in zip(df['redBars'], totals)]

        # Plot
        barWidth = 0.40
        names = ('TCP', 'ICMP', 'UDP')
        plt.bar(bar_index, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
        plt.bar(bar_index, redBars, bottom=greenBars, color='#ff5252', edgecolor='white', width=barWidth)
        plt.xticks(bar_index, names)
        plt.xlabel("protocol")
        plt.ylabel("percentage")
        plt.title("Percentage legitimate/botnet packets per protocol")
        red = mpatches.Patch(color='#ff5252', label='Botnet')
        green = mpatches.Patch(color='#b5ffb9', label='Legitimate')
        plt.legend(handles=[red, green])

        plt.show()

    # Encodes the netflows as stated in the paper.
    def netflow_encoding(self, netflow):
        code = 0

        # Each tuple is in the form of <function_name, column_name in dataframe, size>
        attributes_tuples = []

        if self.protocol_boolean:
            attributes_tuples.append([self.get_protocol_mapping, 'prot', len(self.attribute_mapping_protocol)])

        if self.packets_boolean:
            attributes_tuples.append([self.get_packets_mapping, 'packets', self.bins])

        if self.duration_boolean:
            attributes_tuples.append([self.get_duration_mapping, 'durat', self.bins])

        if self.bytes_boolean:
            attributes_tuples.append([self.get_bytes_mapping, 'bytes', self.bins])

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

    # Checks the correlation of two columns in the data set.
    def correlation(self, first_column, second_column):
        print("Correlation of only ICMP packets: ",
              self.df[self.df.prot == "ICMP"][first_column].corr(self.df[self.df.prot == "ICMP"][second_column]))

        print("Correlation of all packets: ", self.df[first_column].corr(self.df[second_column]))

    @staticmethod
    def run_task(preprocessing=False):
        if (preprocessing):
            print("Preprocessing. Wait til it says it is done.")
            discretization_task.preprocess(input="capture20110818.pcap.netflow.labeled", output="datasets/preprocessed_task_3_4.csv",
                            list_of_ips=["147.32.84.205", "147.32.84.170", "147.32.84.134", "147.32.84.164",
                                         "147.32.87.36", "147.32.80.9", "147.32.87.11"], task_name="discretization")
            print("Done.")

        discretization = discretization_task("datasets/preprocessed_task_3_4.csv",
                                             bins=3,
                                             protocol=True,
                                             packets=True,
                                             duration=False,
                                             bytes=False)

        discretization.protocol_visualization()
        discretization.packets_visualization()
        discretization.imcp_visualization()

        discretization.add_netflow_encoding_column()
        discretization.compare_hosts("147.32.84.205", "147.32.84.170")
        discretization.compare_hosts("147.32.84.205", "147.32.84.134")
        discretization.compare_hosts("147.32.84.205", "147.32.84.164")
        discretization.compare_hosts("147.32.84.205", "147.32.87.36")
        discretization.compare_hosts("147.32.84.205", "147.32.80.9")
        discretization.compare_hosts("147.32.84.205", "147.32.87.11")

if __name__ == "__main__":
    # Set 'preprocessing' to True if you want to create the dataset, set to False to use the provided dataset.
    discretization_task.run_task(preprocessing=False)

#