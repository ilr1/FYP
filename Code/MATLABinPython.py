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
import matlab.engine
#reference signal
filePath = "Ludwig_van_Beethoven_-_Concerto_for_Piano_and_Orchestra_no._4_in_G_major_-_3._Rondo_(Wilhelm_Kempff,_Ferdinand_Leitner,_Berliner_Philharmoniker,_1962).flac"

cPath = "pythonData.mat"

#call matlab script to read in signal, play signal, read out signal 
try:
    #start engine
    eng = matlab.engine.start_matlab()
    #play sound step 1: signal 1 becomes signal 2
    eng.readFileToDaq(filePath,1)
    #load in contents
    contents = sio.loadmat(cPath)
    #operate on contents step 2: signal 2 becomes signal 3
    data = contents['data'] + 1
    #play signal 3 and see if its better step 3: signal 3 becomes signal 2/4 (speaker outputs)
    #save altered data
    sio.savemat(cPath,mdict = {'data': data})
finally:
    eng.quit()