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
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

for data in trainset:
    print(data)
    break

#data is a [tensor,tensor] where the first tensor is a tensor of tensors containing our data and the second is a tensor of our labels for the first
x,y = data[0][0], data[1][0]

print(y)

import matplotlib.pyplot as plt

print(data[0][0].shape)

#plt.imshow(data[0][0].view(28,28))
#plt.show()

total = 0
counter_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

for data in trainset:
    Xs,Ys = data
    for y in Ys:
        counter_dict[int(y)] += 1

print(counter_dict)

