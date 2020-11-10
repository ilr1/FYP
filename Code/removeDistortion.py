import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

myFile = np.genfromtxt('.\clean600noise600.txt', delimiter=',')
#trainset = myFile[:750,:]#torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
#testset = myFile[750:,:] #torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)
trainset = torch.randn(12800,600)
trainset = torch.cat([trainset, torch.mul(trainset, trainset)], dim=1)

testset = torch.ones(1,1200)*0.5
device = torch.device("cuda")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(600 , 1200),
            nn.Linear(1200 , 1200),
            nn.Linear(1200 , 600),
        )

    def forward(self, x):
        return self.model(x)

net = Net().to(device)

optimizer = optim.Adam(net.parameters(), lr = 1e-4)

EPOCHS = 30
BATCH_SIZE = 32

for epoch in range(EPOCHS):
    for i in range(0, len(trainset), BATCH_SIZE):
        batch = trainset[i:i+BATCH_SIZE]

        # data is a batch of featuresets and labels
        batch_X = batch[:,600:].to(device)
        batch_y = batch[:,:600].to(device)

        net.zero_grad()
        output = net(batch_X)
        loss = F.mse_loss(output, batch_y)

        loss.backward()
        optimizer.step()

    print(loss)

X = trainset[0][600:].to(device)
print(net(X))

correct = 0
total = 0
notbad = 0
# now we test! since we aren't training, we dont need to calculate gradients
with torch.no_grad():
    for row in testset:
        X = row[600:].to(device)
        y = row[:600].to(device)
        output = net(X.view(-1,600))
        
        loss = F.mse_loss(output, y)
        print(loss)
        print(output)

# #print("Accuracy: ", round(correct/total,3))
# print("not bad: ", round(notbad/total,3))
