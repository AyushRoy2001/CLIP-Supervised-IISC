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
    def __init__(self, projected_tg, projected_tb):
        # initialising from random uniform distributions
        self._mu_g = nn.Parameter(torch.mean(projected_tg, dim=0, keepdim=True), requires_grad=True).to(args.device)
        self._A_g = nn.Parameter(torch.eye(self._mu_g.size(1), device=args.device)*0.5, requires_grad=True)
        self._mu_b = nn.Parameter(torch.mean(projected_tb, dim=0, keepdim=True), requires_grad=True).to(args.device)
        self._A_b = nn.Parameter(torch.eye(self._mu_g.size(1), device=args.device)*0.5, requires_grad=True)
    
    def generateSamples(self, good=True): 
        z = torch.randn(args.random_samples, self._mu_g.size(1), device=args.device, requires_grad=True)
        if good:
            t = self._mu_g + torch.matmul(z, self._A_g.T)
        else:
            t = self._mu_b + torch.matmul(z, self._A_b.T)
        return t
