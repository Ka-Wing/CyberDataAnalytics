import pandas as pd
import random


class MinWiseSampling():

    k = 0
    i = 0
    item = None
    item_value = 2
    list_of_items = []

    def __init__(self, k):
        self.k = k

    def input(self, item):
        random_number = random.random()
        if random_number < self.item_value:
            self.item = item
            self.item_value = random_number

        self.i = self.i + 1

        # Resets after getting k items.
        if self.i == self.k:
            self.i = 0
            self.item_value = 2
            self.list_of_items.append(item)
            item = None

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

    def load_df(self):
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
            df = pd.DataFrame(dataframe_list, columns=headers)
            self.df = df[(df['src_ip'] == "147.32.84.229") | (df['dst_ip'] == "147.32.84.229")]




    # Sample the dataset using Min-Wise Sampling
    def minwise_sampling(self, k):
        mws = MinWiseSampling(k)

        for i in range(self.df.shape[0]):
            print(i, "/", self.df.shape[0])
            mws.input(self.df.iloc[i])

        return mws.get_dataframe()






if __name__ == "__main__":
    s = sampling_task()
    s.load_df()
    df = s.minwise_sampling(6)
    # df.to_csv('mws_sample.csv', sep=',')

