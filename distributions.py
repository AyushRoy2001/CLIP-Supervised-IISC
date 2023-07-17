import os
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin

from configs import transformation, parse_option
from dataloader import IQADataloader
from model import CustomCLIP

# The arguments
args = parse_option()

class Distribution:
    def __init__(self, N):
        # initialising from random uniform distributions
        self._mu_g = torch.rand(size = (1, 1024), device=args.device, requires_grad=True)
        self._A_g = torch.rand(size = (1024, 1024), device=args.device, requires_grad=True)
        self._mu_b = torch.rand(size = (1, 1024), device=args.device, requires_grad=True)
        self._A_b = torch.rand(size = (1024, 1024), device=args.device, requires_grad=True)
        self._num_samples = N
    
    def generateSamples(self, good=True):
        z = torch.randn(self._num_samples, device=args.device, requires_grad=True)
        if good:
            t_g = self._mu_g + torch.matmul(self._A_g, z)
            return t_g
        t_b = self._mu_b + torch.matmul(self._A_b, z)
        return t_b
