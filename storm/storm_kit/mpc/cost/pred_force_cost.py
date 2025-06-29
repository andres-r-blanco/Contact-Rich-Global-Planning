#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch
import torch.nn as nn
# import torch.nn.functional as F
from .gaussian_projection import GaussianProjection
import numpy as np
import time


class PredForceCost(nn.Module):
    def __init__(self, weight, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float64},
                 force_threshold=0.5, safety_threshold=10.0):
        super(PredForceCost, self).__init__()
        self.tensor_args = tensor_args
        self.weight = weight
        self.force_threshold = force_threshold
        self.safety_threshold = safety_threshold
        self.device = self.tensor_args['device']

    def forward(self, pred_forces):
        # if len(contact_info['force']) == 0 or np.sum(torch.abs(contact_info['force'])) < self.force_threshold:
        #     return torch.zeros((state_batch.shape[0], state_batch.shape[1]), **self.tensor_args)
        # cost = torch.zeros((state_batch.shape[0], state_batch.shape[1]), **self.tensor_args)
        # penalize state if predicted force is higher than force threshold
        
        cost = torch.sum(torch.abs(pred_forces), dim=-1) - self.force_threshold
        cost = torch.clamp(cost, min=0.0)
        cost = self.weight * cost
        top_k = cost.sum(dim=1).topk(20, largest=True)
        print ("Top K pred cost: ", top_k)
        
        return cost.to(self.device)