import pandas as pd

# Supermethod of any tasks.
class task():
    df = None

    # Loading dataframe from a CSV.
    def load_df(self, fileName):
        self.df = pd.read_csv(fileName)

    # Preprocessing the dataset.
    # input: Input file name
    # output: Output file name
    # list_of_ips: IPs to filter the dataset.
    # taskname: "sampling", "sketching", "discretization" or "profiling"
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

            if (task_name == 'sampling' or task_name == 'sketching'):
                df['ip'] = df['src_ip'].map(str) + df['dst_ip']
                df['ip'] = df['ip'].map(lambda x: x.replace(list_of_ips[0], ""))
            elif task_name == 'discretization' or 'profiling':
                df = df[df['label'] != "Background"]

            df.to_csv(output, sep=',', index=False)

    # Parsing the line of the dataset.
    @staticmethod
    def __parse_line(line):
        # Replace all double tabs to one tabs.
        line = line.replace('\t\t', '\t')

        # Replace all tabs to spaces.
        line = line.replace('\t', " ")

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