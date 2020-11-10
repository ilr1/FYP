# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:24:00 2020

@author: isaac
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

sequence = [[1,2,3,4,5],
            [100,101,102,103,104]]

# Define Autoregressive Network
class GRU_Counter(nn.Module):
    def __init__(self, batch_size):
        super(GRU_Counter, self).__init__()
        
        self.batch_size = batch_size
        self.input_size = 1
        self.hidden_size = 1
        
        self.ar = nn.GRU(self.input_size, self.hidden_size, num_layers=1, bidirectional=False, batch_first=True) # (B, S) = (2,5)
        
        def init_hidden(self):
            # Initialize hidden and cell states
            # (num_layers * num_directions, batch, hidden_size)
            return Variable(torch.zeros(1, self.batch_size, self.hidden_size))

        def forward(self, x):            
            for element in batch:
                out, h = self.ar(element, h) # c = 1 * 1 * 256
                    
            return out
        
        
# Train the model
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(5):
    optimizer.zero_grad()

    output = net(inputs)

    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()

    # print loss and predictions
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
    print("Predicted string: ", ''.join(result_str))
                    
            
