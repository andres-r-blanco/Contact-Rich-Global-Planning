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

class LatentGoalCost(nn.Module):
    """ Rotation cost 

    .. math::
     
    r  &=  \sum_{i=0}^{num_rows} (R^{i,:} - R_{g}^{i,:})^2 \\
    cost &= \sum w \dot r

    
    """
    def __init__(self, weight, tensor_args={'device':"cpu", 'dtype':torch.float32}):

        super(LatentGoalCost, self).__init__()
        self.tensor_args = tensor_args
        self.I = torch.eye(3,3, **tensor_args)
        self.weight = weight
        self.dtype = self.tensor_args['dtype']
        self.device = self.tensor_args['device']
    

    def forward(self, latent_state_batch, latent_goal):
        threshold = 1
        latent_state_batch = latent_state_batch.to(self.device)
        latent_goal = latent_goal.to(self.device)
        # latent_state_batch = torch.cat([latent_state_batch, torch.zeros_like(latent_state_batch)], dim=-1)
        # latent_goal = torch.cat([latent_goal, torch.zeros_like(latent_goal)], dim=-1)
        # print("latent_state_batch", latent_state_batch.shape)
        # print("latent_goal", latent_goal.shape)
        cost = self.weight * torch.sum((latent_state_batch - latent_goal)**2, dim=-1)
        # set values less than threshold to zero
        cost = torch.where(cost < threshold, torch.zeros_like(cost), cost)
        # print ("latent cost shape", cost.shape)
        print ("latent cost: ", cost[:10])
        return cost