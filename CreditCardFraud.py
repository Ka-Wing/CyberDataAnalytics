import csv

import math


class FraudDetection:

    reader = None
    mxn_to_usd_multiplier = 0.6
    aud_to_usd_multiplier = 0.735
    gbp_to_usd_multiplier = 1.54
    nzd_to_usd_multiplier = 0.63
    sek_to_usd_multiplier = 0.115
    list = []

    def load_data(self, path):
        self.reader = csv.reader(open(path, 'rt', encoding="ascii"), delimiter=',', quotechar='|')

    def load_data_in_list(self):
        if self.reader is None:
            return False
        else:
            for row in self.reader:
                self.list.append(row)


    def print_list(self):
        if self.list == []:
            return False
        else:
            i = 0
            for row in self.reader:
                print(row[6])
                #print(', '.join(row))

    def changecurrency(self):
        with open('changedcurrency.csv', 'w') as file:
            file.write("txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,currencycode,shoppercountrycode,"
                       "shopperinteraction,simple_journal,cardverificationcodesupplied,cvcresponsecode,creationdate,"
                       "accountcode,mail_id,ip_id,card_id")
            file.write('\n')

            if self.list == []:
                return False
            else:
                for row in self.list:
                    conversion = 0
                    if row[6] == "SEK":
                        conversion = float(row[5]) * self.sek_to_usd_multiplier
                    elif row[6] == "NZD":
                        conversion = float(row[5]) * self.nzd_to_usd_multiplier
                    elif row[6] == "AUD":
                        conversion = float(row[5]) * self.aud_to_usd_multiplier
                    elif row[6] == "GBP":
                        conversion = float(row[5]) * self.gbp_to_usd_multiplier
                    elif row[6] == "MXN":
                        conversion = float(row[5]) * self.mxn_to_usd_multiplier

                    row[5] = str(math.floor(conversion))
                    line = ', '.join(row)
                    file.write(line)
                    file.write('\n')

    def print_data(self):
        if self.reader is None:
            return False
        else:
            i = 0
            for row in self.reader:
                print(', '.join(row))

if __name__ == "__main__":
    a = FraudDetection()
    a.load_data('C:\\Users\\kw\\Dropbox\\TU Delft\\Y2\\Q4\\CS4035 Cyber Data Analytics\\Week 1 - Credit Card '
                  'Fraud\\data_for_student_case.csv(1)\\data_for_student_case.csv')
    a.load_data_in_list()
    #a.print_list()
    a.changecurrency()