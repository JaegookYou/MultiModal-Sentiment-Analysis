import os
import pickle
import numpy as np
from random import random

from config import get_config, activation_dict
from data_loader import get_loader

import torch
import torch.nn as nn
from torch.nn import functional as F

# Setting the config for each stage
train_config = get_config(mode='train')
dev_config = get_config(mode='dev')
test_config = get_config(mode='test')

print(train_config)

# # Creating pytorch dataloaders
# train_data_loader = get_loader(train_config, shuffle = True)
# dev_data_loader = get_loader(dev_config, shuffle = False)
# test_data_loader = get_loader(test_config, shuffle = False)