# Assignment 3 

## Reproducing steps

### Please install the following packages for Python 3

* scikit-learn
* seaborn
* statsmodels
* pandas
* matplotlib
* numpy
* hmmlearn

### Sampling task

* Run the functions provided in the ```main``` of ```sampling_task.py```.
* ```sampling_task``` uses the ```MinWiseSampling class``` in ```misc.py```.

### Sketching task

* Run the functions provided in the ```main``` of ```sketching_task.py```.
* ```sketching_task``` uses the ```CountMinSketch class``` in ```misc.py```.

### Botnet flow data discretization task

* Run the functions provided in the ```main``` of ```discretization_task.py```.

### Botnet profiling task

* Run the functions provided in the ```main``` of ```profiling.py``` below the comment stating ```Botnet Profiling Task```.

## Important notes
* Every Task uses the ```task class``` in ```task.py```.
* For Tasks which take quite some time to run, they print out the progress of the task, so that you know that the program is busy doing something and is not stuck.
* Use the preprocessed datasets provided by us or you can generate them yourselves. Be sure to use the correct datasets to process with as stated in the report. When you generate the preprocessed datasets yourself, note that this may take a very long time.
* We have split the tasks into several files because one group decided to deduct points for readability of the code
due to code that was clustered together in one file. Therefore we decided to split the tasks to several files, but
it seems some systems fails to import local files. So we also provided the file ```all.py``` with all the classes in it, which you can run if your system fails to import the files.
* Feel free to ask us questions if you are running into small problems. Look for us at Slack.
