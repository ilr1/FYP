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
        
    inputFileURL = "From50HzTo1kHzInput.mat"
    expectedFileURL = "From50HzTo1kHzPureSine.mat"

    fIn = sio.loadmat(inputFileURL)
    fOut = sio.loadmat(expectedFileURL)
    
    inputs = fIn['data']
    expected = fOut['expectedData']

    inputs = inputs[3000:7000]
    expected = expected[3000:7000]

    print(inputs.shape)

    return inputs, expected

# define the model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv1d(1, 512, 10, 5),
            nn.ReLU(),
            nn.Conv1d(512, 512, 8, 4),
            nn.ReLU(),
            nn.Conv1d(512, 512, 4, 2),
            nn.ReLU(),
            nn.Conv1d(512, 512, 4, 2),
            nn.ReLU(),
            nn.Conv1d(512, 512, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(11776, 4000)
        )

    def forward(self, x):
        return self.model(x)

net = Net().to(device)

# define the loss function
loss_function = MSELoss()
# define the optimizer
optimizer = SGD(net.parameters(), lr=0.01)

# define the number of epochs and the data set size
epochs = 5

#create training and testing data
X_train, y_train = data_importer()

X_train = X_train[:,:800]
y_train = y_train[:,:800]

X_test = X_train[:,801:]
y_test = y_train[:,801:]

# create our training loop
for epoch in range(epochs):
    epoch_loss = 0

    paired = np.concatenate((X_train, y_train))
    np.random.shuffle(np.transpose(paired))
    X_train = paired[0:4000]
    y_train = paired[4000:8000]

    for i in tqdm(range(800)):
        X = Variable(Tensor(X_train[:,i])).view(1,1,-1).to(device)
        y = Variable(Tensor(y_train[:,i])).view(1,-1).to(device)

        net.zero_grad()

        y_pred = net(X)

        loss = loss_function(y_pred, y)
        epoch_loss += loss

        loss.backward()
        optimizer.step()
        
    print(f"Epoch: {epoch}. Loss: {epoch_loss}")

#net.save_state_dict('trainedNetwork.pt')
torch.save(net.state_dict(),'trainedNetwork.pt')

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(800,951,1)):
        real_val = y_test[i].to(device)
        net_out = net(test_X[i].view(-1, 1, -1).to(device))  # returns a list, 
        predicted_val = net_out
        loss = loss_function(predicted_val, real_val)
        if loss < real_val/10:
            correct += 1
        total += 1

print("Accuracy: ", round(correct/total, 3))

#X = Variable(Tensor(X_train[:,1]).view(-1,1,4000)).to(device)
#var = net(X).cpu()
#var = var.detach().numpy()


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
