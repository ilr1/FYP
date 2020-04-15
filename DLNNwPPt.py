#import important modules
import torch
import torchvision
from torchvision import transforms, datasets

#Its really important to seperate out our datasets early

train = datasets.MNIST("", train = True, download = False, 
                        transform = transforms.Compose([transforms.ToTensor()]))

test =  datasets.MNIST("", train = False, download = False,
                        transform = transforms.Compose([transforms.ToTensor()]))

#we shuffle because the name of the game is generalisation (avoid overfitting (memorisation of ds))
#Be aware that we could fit the entire dataset through our model in one go (~150Mb) but we divide into 10 batches
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ##input layer, output layer
        #layer 1
        self.fc1 = nn.Linear(28*28 , 64)
        #layer 2
        self.fc2 = nn.Linear(64 , 64)
        #layer 3 
        self.fc3 = nn.Linear(64 , 64)
        #layer 4 - 64 from previous, 10 numbers output
        self.fc4 = nn.Linear(64 , 10)
    
    def forward(self, x):
        # runs on *output* of layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()
print(net)

X = torch.rand((28,28))
X = X.view(-1,28*28)

output = net(X)

print(output)

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr = 0.001)

#iterate over our data (epoch is whole pass over data set)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of featuresets and labels
        X, y = data
        # if you dont zero the gradient, they will just continue getting added together
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        # if our dataset is a set of vectors, we hope they are one-hot vectors (one value on) and we use m-square-err
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

# now we test! since we aren't training, we dont need to calculate gradients
with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1,28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total,3))

import matplotlib.pyplot as plt
