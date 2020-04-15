import torch

x = torch.Tensor([5,3])

print(x)

y = torch.rand([2,5])

print(y)

#reshaping an array is called view()

y.view([1,10])

print(y.view([1,10]))