from .task import task
from .discretization_task import discretization_task
import numpy as np
from hmmlearn import hmm


class profiling_task(task):
    dataframe = None  # The preproccessed dataset
    infected_hosts = None  # IP of all infected hosts minus the chosen infected host (147.32.84.165)
    normal_hosts = None  # IP of all normal hosts

    def __init__(self, dataframe, infected_hosts, normal_hosts):
        self.dataframe = dataframe
        self.infected_hosts = infected_hosts
        self.normal_hosts = normal_hosts

    # Returns array with sliding windows of size window_size
    def sliding_windows(self, ip, window_size):
        new_data = []
        # Obtain sequence data from the netflows from the given ip
        data = self.dataframe[(self.dataframe['src_ip'] == ip) | (self.dataframe['dst_ip'] == ip)]
        data = data['encoding'].tolist()

        if len(data) < window_size:
            return new_data

        for i in range(len(data) - window_size):
            new_data.append(data[i:i + window_size])
        new_data = np.array(new_data)

        return new_data

    # Returns the log probability of all hosts
    def hmm_model(self, data):
        # Learn hmm from the data of infected host 147.32.84.165
        model = hmm.GaussianHMM(n_components=4)
        model.fit(data)
        # Save the log probability
        logprob_infected = model.score(data)

        # Get log probability of the other infected and normal hosts,
        # using the model learned from the data from the chosen infected host
        logprob_others = []

        # Get log probability of the other infected hosts
        for infected in self.infected_hosts:
            new_data = self.sliding_windows(infected, 10)
            if len(new_data) == 0:
                logprob_others.append((infected, 0))
            else:
                logprob_others.append((infected, model.score(new_data)))
        # Get log probability of the normal hosts
        for normal in self.normal_hosts:
            new_data = self.sliding_windows(normal, 10)
            if len(new_data) == 0:
                logprob_others.append((normal, 0))
            else:
                logprob_others.append((normal, model.score(new_data)))

        # Prints the log probability of the infested host and all other hosts
        # This is used to determine the threshold
        print("logprob_infected: ", logprob_infected)
        print("logprob_others: ", logprob_others)

        return logprob_infected, logprob_others

    # Returns a list with ips which are classified as infected and another list,
    # with ips which are classified as normal
    def classification(self, logprob_infected, logprob_others):
        classified_infected = []
        classified_normal = []

        for tup in logprob_others:
            ip, logprob = tup
            # Check whether the difference is below the threshold logprob_infected/2?
            # Classify accordingly
            if abs(logprob - logprob_infected) < (logprob_infected / 2):
                classified_infected.append(ip)
            else:
                classified_normal.append(ip)

        return classified_infected, classified_normal

    # Compute true negatives, true positives, false negatives, true positives,
    # and precision and recall
    def evaluation(self, classified_infected, classified_normal):
        tn = 0
        fp = 0
        fn = 0
        tp = 0

        for ip in classified_infected:
            if ip in self.infected_hosts:
                tp = tp + 1
            else:
                fp = fp + 1

        for ip in classified_normal:
            if ip in self.normal_hosts:
                tn = tn + 1
            else:
                fn = fn + 1

        print("tp: ", tp)
        print("tn: ", tn)
        print("fp: ", fp)
        print("fn: ", fn)
        print("precision: ", tp / (tp + fp))
        print("recall: ", tp / (tp + fn))

    @staticmethod
    def run_task(self, preprocessing=False):
        if preprocessing:
            if (preprocessing):
                self.preprocess(input="datasets/capture20110818.pcap.netflow.labeled",
                                output="datasets/preprocessed_task_3_4.csv",
                                list_of_ips=["147.32.84.205", "147.32.84.170", "147.32.84.134", "147.32.84.164",
                                             "147.32.87.36", "147.32.80.9", "147.32.87.11"], task="profiling")

        discretization = discretization_task("datasets/preprocessed_task_3_4.csv",
                                             bins=3,
                                             protocol=True,
                                             packets=True,
                                             duration=False,
                                             bytes=False)
        discretization.add_netflow_encoding_column()

        profiling = profiling_task(discretization.df, ["147.32.84.191", "147.32.84.192", "147.32.84.193",
                                                       "147.32.84.204", "147.32.84.205", "147.32.84.206",
                                                       "147.32.84.207", "147.32.84.208", "147.32.84.209"],
                                   ["147.32.84.170", "147.32.84.134", "147.32.84.164",
                                    "147.32.87.36", "147.32.80.9", "147.32.87.11"])
        data = profiling.sliding_windows("147.32.84.165", 10)
        logprob_infected, logprob_others = profiling.hmm_model(data)
        classified_infected, classified_normal = profiling.classification(logprob_infected, logprob_others)
        profiling.evaluation(classified_infected, classified_normal)


if __name__ == "__main__":
    # Set 'preprocessing' to True if you want to create the dataset, set to False to use the provided dataset.
    profiling_task.run_task(preprocessing=False)