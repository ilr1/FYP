import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#myFile = np.genfromtxt('.\clean600noise600.txt', delimiter=',')
#trainset = myFile[:750,:]#torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
#testset = myFile[750:,:] #torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

trainset = torch.randn(32000,1) #torch.from_numpy(np.random.uniform(-2,2,32000)).view([32000,1]).type(torch.FloatTensor)
#noiseSet = torch.randn(32000,1)
trainset = torch.cat([trainset + 0.1*trainset.pow(3),trainset], dim=1)

testset = torch.ones(1,1)*0.5
testset = torch.cat([testset + 0.1*testset.pow(3),testset], dim=1)

device = torch.device("cuda")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1 , 32),
            nn.ReLU(),
            nn.Linear(32 , 32),
            nn.ReLU(),
            nn.Linear(32 , 1),
        )

    def forward(self, x):
        return self.model(x)

net = Net().to(device)

optimizer = optim.Adam(net.parameters(), lr = 1e-4)

EPOCHS = 20
BATCH_SIZE = 32

for epoch in range(EPOCHS):
    for i in range(0, len(trainset), BATCH_SIZE):
        batch = trainset[i:i+BATCH_SIZE]

        # data is a batch of featuresets and labels
        batch_X = batch[:,0].to(device).view(-1,1)
        batch_y = batch[:,1].to(device).view(-1,1)

        net.zero_grad()
        output = net(batch_X)
        loss = F.mse_loss(output, batch_y)

        loss.backward()
        optimizer.step()

    print(loss)

print(testset.shape)
print(testset)
print(testset[0][0])
X = testset[0][0].to(device).view(1,1)
print(net(X))

# correct = 0
# total = 0
# notbad = 0
# # now we test! since we aren't training, we dont need to calculate gradients
# with torch.no_grad():
#     for row in testset:
#         print(row.shape)
#         X = row[0].to(device).view(-1,1)
#         y = row[1].to(device).view(-1,1)
#         output = net(X.view(-1,1))
        
#         loss = F.mse_loss(output, y)
#         print(loss)
#         print(output)

# #print("Accuracy: ", round(correct/total,3))
# print("not bad: ", round(notbad/total,3))
