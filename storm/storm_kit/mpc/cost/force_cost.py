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


class ForceThresholdCost(nn.Module):
    def __init__(self, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float64},
                 bounds=[], weight=1.0, gaussian_params={}, k=1000.0, bound_thresh=0.1, force_threshold=0.5, safety_threshold=10.0):
        super(ForceThresholdCost, self).__init__()
        self.tensor_args = tensor_args
        self.weight = weight
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        default_k = k
        self.k_c = torch.eye(3, **tensor_args).fill_(default_k).unsqueeze(0)
        self.force_threshold = force_threshold
        self.safety_threshold = safety_threshold

    def forward(self, start_state, state_batch, contact_info, contact_jac, contact_norm):
        inp_device = state_batch.device
        # if len(contact_info['force']) == 0 or np.sum(torch.abs(contact_info['force'])) < self.force_threshold:
        #     return torch.zeros((state_batch.shape[0], state_batch.shape[1]), **self.tensor_args)
        
        t = time.time()
        # df = normals * self.k_c * J * dq
        dq_batch = torch.as_tensor(state_batch - start_state, **self.tensor_args) # (n_batch, n_steps, 7)
        print ("Forces: ", contact_info['force'])
        # contact jac is of shape (n_batch, n_steps, 6, 7)
        # contact norm is of shape (n_batch, n_steps, 3)
        delta_f = torch.matmul(self.k_c, torch.matmul(contact_jac, dq_batch.unsqueeze(-1))) # (n_batch, n_steps, 6, 1)
        contact_norm = contact_norm.unsqueeze(-2)

        delta_f = torch.matmul(contact_norm, delta_f).squeeze(-1) # (n_batch, n_steps, 6)
        delta_f = delta_f.squeeze(-1) # (n_batch, n_steps, 6)
        # jacobian = torch.as_tensor(np.array(contact_info['jac']), **self.tensor_args) # (num_contacts, 6, 7)
        # dq_batch = dq_batch.unsqueeze(-1) # (n_batch, n_steps, n_dof, 1)
        # jacobian = jacobian.view(jacobian.shape[0], 1, 1, jacobian.shape[1], jacobian.shape[2]) # (num_contacts, 1, 1, goal_dim, n_dof)
        # delta_f = torch.matmul(self.k_c, torch.matmul(jacobian, dq_batch)) # (num_contacts, n_batch, n_steps, goal_dim, 1)
        # normals = torch.as_tensor(contact_info['normal'], **self.tensor_args) # (num_contacts, 3)
        # normals = normals.view(normals.shape[0], 1, 1, 1, normals.shape[1]) # (num_contacts, 1, 1, 1, 3)
        # delta_f = torch.matmul(normals, delta_f).view(delta_f.shape[0], delta_f.shape[1], delta_f.shape[2]) # (num_contacts, n_batch, n_steps)

        forces = torch.as_tensor(contact_info['force'], **self.tensor_args) # (num_contacts, 1)
        forces = forces.view(forces.shape[0], 1, 1) # (num_contacts, 1, 1)
        predicted_forces = forces + delta_f # (num_contacts, n_batch, n_steps, 1)
        predicted_forces[predicted_forces < 0] = 0
        penalty = predicted_forces
        penalty = torch.pow(predicted_forces, 2)
        cost = torch.sum(penalty, dim=0)
        
        discount_factor = torch.as_tensor(np.array([0.9**i for i in range(cost.shape[1])]), **self.tensor_args).unsqueeze(0)
        # discount_factor[:,3:] = 0
        # cost is of shape (n_batch, n_steps)
        cost = cost * discount_factor
        
        # with torch.cuda.amp.autocast(enabled=False):
        #     J_J_t = torch.matmul(jac_batch, jac_batch.transpose(-2,-1))
        #     score = torch.sqrt(torch.det(J_J_t))
        # score[score != score] = 0.0
        top_k = cost.sum(dim=1).topk(20, largest=False)
        print ("Cost: ", top_k)
        # penalize based on force magnitude; ideal is to have 0 force
        print ("Time: ", time.time() - t)
        return cost.to(inp_device)