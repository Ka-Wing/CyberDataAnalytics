#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:55:53 2018

@author: smto
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read in the data to a pandas dataframe
data = pd.read_csv('data_for_student_case.csv')

#remove 'Refused' transactions as we do not know whether they're fraud or beneign
data = data[data['simple_journal'] != 'Refused']

def currency_converter(row):
    #GBP: 1.5618545948 USD 1.5433200000
    #AUD: 0.7663613939 USD 0.7135750000
    #MXN: 0.0633811416 USD 0.0606053637
    #SEK: 0.1197737051 USD 0.1174262284
    #NZD: 0.6754660315 USD 0.6791000000
    #Took the average of these for the currency converter
    currencies = {'GBP': 1.550, 'AUD': 0.735, 'MXN': 0.062, 'SEK': 0.118, 'NZD': 0.677}
    
    return row['amount'] * currencies[row['currencycode']]
    
def plot():
    #add new column with the amount converted to amount in USD
    data['usd_amount'] = data.apply(lambda x : currency_converter(x), axis=1)
    
    #print(data.cvcresponsecode.unique())
    
    #sns.stripplot(data=data, x="shopperinteraction", y="usd_amount", hue="simple_journal", jitter=True)
    
    #sns.swarmplot(data=data, x="shopperinteraction", y="usd_amount", hue="simple_journal")
    
    # USED in overleaf: shopperinteraction
    #sns.factorplot(data=data, x="shopperinteraction", y="usd_amount", col="simple_journal", kind="strip", jitter=True)
    
    # USED in overleaf: cvcresponsecode
    #sns.factorplot(data=data, x="cvcresponsecode", y="usd_amount", col="simple_journal", kind="strip", jitter=True)
    
    # USED in overleaf: currencycode
    #sns.factorplot(data=data, x="currencycode", y="usd_amount", col="simple_journal", kind="strip", jitter=True)
    
    # USED in overleaf: currencycode
    #sns.factorplot(data=data, x="currencycode", y="usd_amount", col="simple_journal", kind="box")
    
    # USED in overleaf: accountcode
    #g = sns.factorplot(data=data, x="accountcode", y="usd_amount", col="simple_journal", kind="strip", jitter=True)
    #g.set_xticklabels(rotation=35)
    
    # USED in overleaf: cardverificationcodesupplied
    sns.factorplot(data=data, x="cardverificationcodesupplied", y="usd_amount", col="simple_journal", kind="strip", jitter=True)
    
    ############ FRAUD heatmap ############
# =============================================================================
#     sub_df = data[['bin', 'simple_journal', 'usd_amount', 'cvcresponsecode']].copy()
#     sub_fraud_df = sub_df[sub_df['simple_journal'] != 'Settled']
#     #print(sub_df.head(5))
#     
#     rowdicts = []
#     for l, d in sub_fraud_df.groupby("bin usd_amount cvcresponsecode".split()):
#         d = {"bin": l[0], "usd_amount": l[1], "cvcresponsecode": l[2]}
#         rowdicts.append(d)
#         
#     sorted_fraud = pd.DataFrame.from_dict(rowdicts)
#     
#     #print(sorted_fraud)
#     
#     sorted_fraud = sorted_fraud.pivot_table("usd_amount", "bin", "cvcresponsecode")
#     sns.heatmap(data=sorted_fraud, cbar = True, cmap = 'coolwarm')
# =============================================================================
    #print(df2.simple_journal.values.dtype)
    #print(df2)
    
    ############ BENIGN heatmap ############    
# =============================================================================
#     sub_df2 = data[['bin', 'simple_journal', 'usd_amount', 'cvcresponsecode']].copy()
#     sub_benign_df = sub_df2[sub_df2['simple_journal'] != 'Chargeback']
#     #print(sub_df.head(5))
#     
#     rowdicts_benign = []
#     for l, d in sub_benign_df.groupby("bin usd_amount cvcresponsecode".split()):
#         d = {"bin": l[0], "usd_amount": l[1], "cvcresponsecode": l[2]}
#         rowdicts_benign.append(d)
#         
#     sorted_benign = pd.DataFrame.from_dict(rowdicts_benign)
#     
#     sorted_benign = sorted_benign.pivot_table("usd_amount", "bin", "cvcresponsecode")
#     sns.heatmap(data=sorted_benign, cbar = True, cmap = 'coolwarm')
# =============================================================================
    
def main():
    plot()
    
main()