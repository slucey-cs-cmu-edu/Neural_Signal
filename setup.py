import numpy as np
import matplotlib.pyplot as plt
import torch

# Image rescale and I/O
from skimage import io
from skimage.transform import rescale

# stores different optimizors like SGD
import torch.optim as optim

# Some torch functions that are used multiple times
import torch.nn.functional as F
import torch.nn as nn

# Set the class for Sinc activation
class Sinc(nn.Module):
    def __init__(self, beta):
        super(Sinc, self).__init__()
        self.beta = beta

    def forward(self, x):
        return torch.sinc(self.beta*x)