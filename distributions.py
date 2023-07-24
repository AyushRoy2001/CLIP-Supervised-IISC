import os
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin

from configs import parse_option

# The arguments
args = parse_option()

class Distribution:
    def __init__(self):
        # initialising from random uniform distributions
        self._mu_g = nn.Parameter(torch.rand(size = (1, args.random_samples), device=args.device, requires_grad=True))
        self._A_g = nn.Parameter(torch.rand(size = (args.random_samples, args.random_samples), device=args.device, requires_grad=True))
        self._mu_b = nn.Parameter(torch.rand(size = (1, args.random_samples), device=args.device, requires_grad=True))
        self._A_b = nn.Parameter(torch.rand(size = (args.random_samples, args.random_samples), device=args.device, requires_grad=True))
    
    def generateSamples(self, projected_tg, projected_tb, good=True): 
        self._mu_g = 
        self._mu_b = 
        z = torch.randn(args.random_samples, args.random_samples, device=args.device, requires_grad=True)
        if good:
            t_g = self._mu_g + torch.matmul(self._A_g, z)
            return t_g
        else:
            t_b = self._mu_b + torch.matmul(self._A_b, z)
            return t_b
