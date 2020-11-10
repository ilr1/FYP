import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt



if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("CPU")
    print("running on the CPU")


# define our data generation function
def data_importer():
        
    inputFileURL = r'C:\Users\Admin\Documents\fyp\From50HzTo1kHzInput.mat'
    expectedFileURL = r'C:\Users\Admin\Documents\fyp\From50HzTo1kHzPureSine.mat'

    fIn = sio.loadmat(inputFileURL)
    fOut = sio.loadmat(expectedFileURL)
    
    inputs = fIn['data']
    expected = fOut['expectedData']

    inputs = inputs[3000:7000]
    expected = expected[3000:7000]

    return inputs, expected

x,y = data_importer()
print(y.shape)

# define the model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1,400,10)
        self.conv2 = nn.Conv1d(400,200,10)
        self.fc1 = Linear(198200, 200)
        self.fc2 = Linear(200, 4000)


    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x,5,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x,5,2)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

net = Net().to(device)

# define the loss function
loss_function = MSELoss()
# define the optimizer
optimizer = SGD(net.parameters(), lr=0.01)

# define the number of epochs and the data set size
nb_epochs = 10

inD, outD = data_importer()

#plt.plot(outD[:,1])
#plt.show()

paired = np.concatenate((inD, outD))
np.random.shuffle(np.transpose(paired))

inD = paired[0:4000]
print(inD.shape)
outD = paired[4000:8000]

#plt.plot(outD[:,1])
#plt.show()

# create our training loop
for epoch in range(nb_epochs):
    for i in tqdm(range(951)):
        X = Variable(Tensor(inD[:,i]).view(-1,1,4000)).to(device)
        y = Variable(Tensor(outD[:,i])).to(device)


        epoch_loss = 0


        y_pred = net(X)

        loss = loss_function(y_pred, y)

        epoch_loss = loss.data
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    print(f"Epoch: {epoch}. Loss: {loss}")

X = Variable(Tensor(inD[:,1]).view(-1,1,4000)).to(device)
var = net(X).cpu()
var = var.detach().numpy()
#plt.plot(var)
#plt.show()

# # test the model
# model.eval()
# test_data = data_generator(1)
# prediction = model(Variable(Tensor(test_data[0][0])))
# print("Prediction: {}".format(prediction.data[0]))
# print("Expected: {}".format(test_data[1][0]))

# print('a & b:', model.fc1.weight)
# print('c:', model.fc1.bias)
