import math
import re
from matplotlib import pyplot as plt

regex = "(.*?)predicted=(.*?), expected=(.*?):"

def a():
    list = []
    with open("C:\\Users\\kw\\Desktop\\a.txt", "r") as txt:
        txt2 = str(txt.read()).split("\n")
        for i in txt2:
            try:
                int(i[0])
                i = i + ":"
                solution = re.match(regex, i)
                if solution[2] is None:
                    continue
                else:
                    list.append(i)

            except:
                continue
    txt.close()

    with open("C:\\Users\\kw\\Desktop\\b.txt", "w+") as text:
        p = 0
        for i in list:
            if p != 0:
                text.write('\n')
            else:
                p = 1

            text.write(i)
    text.close()

def b():
    predicted = []
    actual = []
    with open("C:\\Users\kw\Desktop\\b.txt", "r") as txt:
        txt = str(txt.read()).split("\n")
        for i in txt:
            print(i)
            solution = re.match(regex, i)
            predicted.append(float(solution[2]))
            actual.append(float(solution[3]))

    plt.plot(predicted)
    plt.plot(actual)
    plt.show()


a()
b()