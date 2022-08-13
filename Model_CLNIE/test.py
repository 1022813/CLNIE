import pickle

import dgl
import pandas as pd
import numpy as np
import torch
import numpy.ma as ma
from torch import nn
import torch.nn.functional as F
import random

if __name__ == '__main__':
    list = torch.tensor([[5,7,3], [1,2,3],[2,3,4]])
    # b = sorted(enumerate(list), key=lambda x: x[1], reverse=True)
    # print(b[2][0])
    new = torch.Tensor()
    new = (list[0] + list[1]).resize(1, len(list[0]))
    l = torch.cat([list, new], dim=0)
    print(l)
