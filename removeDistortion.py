import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


myFile = np.genfromtxt('C:\\Users\\isaac\\Documents\\FYP\\clean600noise600.txt', delimiter=',')

#print(myFile[1,1])

#its important to state that we are operating on a time series, an evenly spaced vector

trainset = myFile[:750,:]#torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = myFile[750:,:] #torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ##input layer, output layer
        #layer 1
        self.fc1 = nn.Linear(600 , 64)
        #layer 2
        self.fc2 = nn.Linear(64 , 64)
        #layer 3 
        self.fc3 = nn.Linear(64 , 64)
        #layer 4 - 64 from previous, 10 numbers output
        self.fc4 = nn.Linear(64 , 600)
    
    def forward(self, x):
        # runs on *output* of layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x #F.log_softmax(x, dim=1)

net = Net()
print(net)

X = torch.rand((1,600))
X = X.view(-1,600*1)

output = net(X)

print(output)

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr = 0.001)

#iterate over our data (epoch is whole pass over data set)

EPOCHS = 5000

for epoch in range(EPOCHS):
    for row in trainset:
        # data is a batch of featuresets and labels
        X = torch.tensor(row[600:],dtype=torch.float)
        y = torch.tensor(row[:600],dtype=torch.float)
        #X, y = data
        # if you dont zero the gradient, they will just continue getting added together
        net.zero_grad()
        output = net(X.view(-1, 1*600))
        # if our dataset is a set of vectors, we hope they are one-hot vectors (one value on) and we use m-square-err
        loss = F.mse_loss(output, y)#F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0
notbad = 0
# now we test! since we aren't training, we dont need to calculate gradients
with torch.no_grad():
    for row in testset:
        X = torch.tensor(row[600:], dtype=torch.float)
        y = torch.tensor(row[:600], dtype=torch.float)
        output = net(X.view(-1,600))
        #return the index of each element and the element
        for idx, i in enumerate(output):
            #if torch.argmax(i) == y[idx]:
            #    correct += 1
            if np.abs(output[0,idx]-y[idx]) < 0.01:
                notbad += 1
            total += 1
            

#print("Accuracy: ", round(correct/total,3))
print("not bad: ", round(notbad/total,3))

plt.plot(output[0],'r')
plt.plot(y,'y')
plt.show()
print(y.size())
print(output[0,1])

