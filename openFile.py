from collections import defaultdict
import numpy as n

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

lengthOfFile = file_len("testData\sample.csv")
f = open("testData\sample.csv", "r")
columns = f.readline().split(",")
for i in range(len(columns)):
    columns[i] = columns[i].strip()

#dataDict = defaultdict(list)
data = {"x":n.empty([lengthOfFile-1,1]), "y":n.empty([lengthOfFile-1,1])}

counter = 0
for i in f:
    values = i.split(",")
    data["x"][counter], data["y"][counter]  = values
    print(data["x"][counter][0], data["y"][counter][0])
    counter +=1

import matplotlib.pyplot as plt

plt.plot(data["x"], data["y"],'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#for line in f:
    #print(line)

f.close()