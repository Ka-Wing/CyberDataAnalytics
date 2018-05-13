import pandas as pd
import seaborn as sns

# read in the data to a pandas dataframe
data = pd.read_csv('C:\\Users\\kw\Dropbox\\TU Delft\\Y2\\Q4\\CS4035 Cyber Data Analytics\\Week 1 - Credit Card '
                   'Fraud\\data_for_student_case.csv(1)\\data_for_student_case.csv')

# remove 'Refused' transactions as we do not know whether they're fraud or beneign
data = data[data['simple_journal'] != 'Refused']


def currency_converter(row):
    # GBP: 1.5618545948 USD 1.5433200000
    # AUD: 0.7663613939 USD 0.7135750000
    # MXN: 0.0633811416 USD 0.0606053637
    # SEK: 0.1197737051 USD 0.1174262284
    # NZD: 0.6754660315 USD 0.6791000000
    # Took the average of these for the currency converter
    currencies = {'GBP': 1.550, 'AUD': 0.735, 'MXN': 0.062, 'SEK': 0.118, 'NZD': 0.677}

    return row['amount'] * currencies[row['currencycode']]


def plot():
    # add new column with the amount converted to amount in USD
    data['usd_amount'] = data.apply(lambda x: currency_converter(x), axis=1)

    # USED in overleaf: shopperinteraction
    sns.factorplot(data=data, x="shopperinteraction", y="usd_amount", col="simple_journal", kind="strip", jitter=True)

    # USED in overleaf: cvcresponsecode
    sns.factorplot(data=data, x="cvcresponsecode", y="usd_amount", col="simple_journal", kind="strip", jitter=True)

    # USED in overleaf: currencycode
    sns.factorplot(data=data, x="currencycode", y="usd_amount", col="simple_journal", kind="box")

    g = sns.factorplot(data=data, x="accountcode", y="usd_amount", col="simple_journal", kind="box")
    g.set_xticklabels(rotation=35)
