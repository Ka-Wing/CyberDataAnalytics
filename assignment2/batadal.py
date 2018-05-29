#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:00:28 2018

@author: smto
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot():
    # read in the data to a pandas dataframe
    signals = pd.read_csv('BATADAL_dataset03.csv')
    # plot the heatmap with correlations
    plt.subplots(figsize=(13,10))
    sns.heatmap(data=signals.corr(), xticklabels=True, yticklabels=True, linewidths=1.0, cbar = True, cmap = 'coolwarm')
    
plot()