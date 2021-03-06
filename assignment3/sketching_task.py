import time

from assignment3.task import task
from assignment3.misc import CountMinSketch

class sketching_task(task):
    # Top ten most frequent in descending order
    ip_addresses = ["208.88.186.6", "78.175.28.225", "82.113.63.230", "195.168.45.2", "82.150.185.24",
                    "213.137.179.195", "62.180.140.208", "81.208.118.74", "88.255.232.197", "62.168.4.186"]

    def __init__(self, fileName):
        self. load_df(fileName)

    # Perform Count-Min Sketch
    def cmsketch(self, delta=0.01, epsilon=0.0000001):
        cms = CountMinSketch(delta, epsilon)
        for i in range(self.df.shape[0]):
            print(i + 1, "/", self.df.shape[0])
            cms.update(self.df.ip.iloc[i])

        print()
        print("delta:", delta, ", epsilon:", epsilon)
        for ip in self.ip_addresses:
            print(ip + ":", int(cms.estimate(ip)))

    @staticmethod
    def run_task(preprocessing=False):
        if preprocessing:
            print("Preprocessing. Wait til it says it is done.")
            sketching_task.preprocess("capture20110817.pcap.netflow.labeled", "datasets/preprocessed_task_2.csv",
                                      list_of_ips=["147.32.84.229"], task_name="sketching")
            print("Done.")

        sketching = sketching_task("datasets/preprocessed_task_2.csv")

        for epsilon in [0.01, 0.001, 0.0001, 0.00001]:
            sketching.cmsketch(delta=0.01, epsilon=epsilon)
            print("These results will stay here 15 seconds on screen before calculating CM-sketch with another "
                  "epsilon.")
            time.sleep(15)

if __name__ == "__main__":
    # Set 'preprocessing' to True if you want to create the dataset, set to False to use the provided dataset.
    sketching_task.run_task(preprocessing=False)

#