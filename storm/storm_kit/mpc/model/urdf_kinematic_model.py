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
from typing import List, Tuple, Dict, Optional, Any
import torch
import torch.autograd.profiler as profiler

from ...differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel
from urdfpy import URDF
from .model_base import DynamicsModelBase
from .integration_utils import build_int_matrix, build_fd_matrix, tensor_step_acc, tensor_step_vel, tensor_step_pos, tensor_step_jerk

import os
import sys
from pathlib import Path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from .cfg.train_cfg import TrainConfig
from .networks.mlp import MLP, AEMLP, Dynamics, AE, VAE, WM_dynamics, WM_encoder, WM_predictor_force, WM_encoder_baseline, WM_encoder_rgbt, WM_predictor_force_baseline, WM_TransformerDynamics_baseline
import numpy as np
import random
import datetime
import torch.nn.functional as F

class URDFKinematicModel(DynamicsModelBase):
    def __init__(self, urdf_path, dt, batch_size=1000, horizon=5,
                 tensor_args={'device':'cpu','dtype':torch.float32}, ee_link_name='ee_link', link_names=[], dt_traj_params=None, vel_scale=0.5, control_space='acc'):
        self.urdf_path = urdf_path
        self.device = tensor_args['device']

        self.float_dtype = tensor_args['dtype']
        self.tensor_args = tensor_args
        self.dt = dt
        self.ee_link_name = ee_link_name
        self.batch_size = batch_size
        self.horizon = horizon
        self.num_traj_points = int(round(horizon / dt))
        self.link_names = link_names
        self.zero_forces = torch.zeros((1, 56), **self.tensor_args)

        self.robot_model = DifferentiableRobotModel(urdf_path, None, tensor_args=tensor_args)

        #self.robot_model.half()
        self.n_dofs = self.robot_model._n_dofs
        self.urdfpy_robot = URDF.load(urdf_path) #only for visualization
        
        self.d_state = 3 * self.n_dofs + 1
        self.d_action = self.n_dofs

        #Variables for enforcing joint limits
        self.joint_names = self.urdfpy_robot.actuated_joint_names
        self.joint_lim_dicts = self.robot_model.get_joint_limits()
        self.state_upper_bounds = torch.zeros(self.d_state, device=self.device, dtype=self.float_dtype)
        self.state_lower_bounds = torch.zeros(self.d_state, device=self.device, dtype=self.float_dtype)
        for i in range(self.n_dofs):
            self.state_upper_bounds[i] = self.joint_lim_dicts[i]['upper']
            self.state_lower_bounds[i] = self.joint_lim_dicts[i]['lower']
            self.state_upper_bounds[i+self.n_dofs] = self.joint_lim_dicts[i]['velocity'] * vel_scale
            self.state_lower_bounds[i+self.n_dofs] = -self.joint_lim_dicts[i]['velocity'] * vel_scale
            self.state_upper_bounds[i+2*self.n_dofs] = 10.0
            self.state_lower_bounds[i+2*self.n_dofs] = -10.0
        # self.state_lower_bounds[-1] = -10.0
        # self.state_upper_bounds[-1] = 10.0
        print (self.state_upper_bounds, self.state_lower_bounds)


        #print(self.state_upper_bounds, self.state_lower_bounds)
        # #pre-allocating memory for rollouts
        self.state_seq = torch.zeros(self.batch_size, self.num_traj_points, self.d_state, device=self.device, dtype=self.float_dtype)
        self.ee_pos_seq = torch.zeros(self.batch_size, self.num_traj_points, 3, device=self.device, dtype=self.float_dtype)
        self.ee_rot_seq = torch.zeros(self.batch_size, self.num_traj_points, 3, 3, device=self.device, dtype=self.float_dtype)
        self.Z = torch.tensor([0.], device=self.device, dtype=self.float_dtype) #torch.zeros(batch_size, self.n_dofs, device=self.device, dtype=self.float_dtype)

        self._integrate_matrix = build_int_matrix(self.num_traj_points, device=self.device, dtype=self.float_dtype)
        self.control_space = control_space
        if(control_space == 'acc'):
            self.step_fn = tensor_step_acc
        elif(control_space == 'vel'):
            self.step_fn = tensor_step_vel
        elif(control_space == 'jerk'):
            self.step_fn = tensor_step_jerk
        elif(control_space == 'pos'):
            self.step_fn = tensor_step_pos

        self._fd_matrix = build_fd_matrix(self.num_traj_points, device=self.device,
                                          dtype=self.float_dtype, order=1)
        if(dt_traj_params is None):
            dt_array = [self.dt] * int(1.0 * self.num_traj_points) 
        else:
            dt_array = [dt_traj_params['base_dt']] * int(dt_traj_params['base_ratio'] * self.num_traj_points)
            smooth_blending = torch.linspace(dt_traj_params['base_dt'],dt_traj_params['max_dt'], steps=int((1 - dt_traj_params['base_ratio']) * self.num_traj_points)).tolist()
            dt_array += smooth_blending
            
            
            self.dt = dt_traj_params['base_dt']
        if(len(dt_array) != self.num_traj_points):
            dt_array.insert(0,dt_array[0])
        self.dt_traj_params = dt_traj_params
        self._dt_h = torch.tensor(dt_array, dtype=self.float_dtype, device=self.device)
        self.dt_traj = self._dt_h
        self.traj_dt = self._dt_h
        self._traj_tstep = torch.matmul(self._integrate_matrix, self._dt_h)
        
        self.link_pos_seq = torch.empty((self.batch_size, self.num_traj_points, len(self.link_names),3), **self.tensor_args)
        self.link_rot_seq = torch.empty((self.batch_size, self.num_traj_points, len(self.link_names),3,3), **self.tensor_args)

        self.prev_state_buffer = None 
        self.prev_state_fd = build_fd_matrix(9, device=self.device, dtype=self.float_dtype, order=1, PREV_STATE=True)


        self.action_order = 0
        self._integrate_matrix_nth = build_int_matrix(self.num_traj_points, order=self.action_order, device=self.device, dtype=self.float_dtype, traj_dt=self.traj_dt)
        self._nth_traj_dt = torch.pow(self.traj_dt, self.action_order)

        # self.init_latent_model()

    def init_latent_model(self):
        # Get the root directory
        root_dir = Path(__file__).resolve().parents[2]

        # Create the configuration for training
        cfg = TrainConfig(root_dir=root_dir)
        cfg.train_taxel = True
        cfg.network_config.input_size = 14
        cfg.network_config.output_size = 70
        cfg.network_config.action_size = 70
        cfg.network_config.sequence_length = 1
        self.cfg = cfg
        
        # Initialize the models
        encoder = WM_encoder_rgbt(cfg.network_config) # WM_encoder_baseline(cfg.network_config)
        self.encoder = encoder.to("cuda")
        predictor = WM_predictor_force_baseline(cfg.network_config)
        self.predictor = predictor.to("cuda")
        dynamics = WM_TransformerDynamics_baseline(cfg.network_config)
        self.dynamics = dynamics.to("cuda")
        
        # TODO: change checkpoint path
        checkpoint_path = '/home/mahika/storm_gen3_learned_dynamics/model_checkpoints/rgbt/best_model.pth'
        checkpoint = torch.load(checkpoint_path)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        self.dynamics.load_state_dict(checkpoint['dynamics_state_dict'])

        self.encoder.eval()
        self.predictor.eval()
        self.dynamics.eval()
        
        self.encoder = self.encoder.float()
        self.predictor = self.predictor.float()
        self.dynamics = self.dynamics.float()

    def get_goal_latent_state(self, traj_data):
        # Get the latent state
        rgb = traj_data['RGB']
        depth = traj_data['Depth']
        joint_pos = torch.tensor(traj_data['joint_pos']).cuda().unsqueeze(0)
        joint_vel = torch.tensor(traj_data['joint_vel']).cuda().unsqueeze(0)
        force = torch.tensor(traj_data['force']).cuda().unsqueeze(0)
        # TODO: generate x based on the input data
        
        x = torch.cat((joint_pos, joint_vel, force), dim=1)
        x = x.float()
        
        rgb = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        curr_latent_state = self.encoder(x, rgb)
        return curr_latent_state

    def predict_using_latent_model(self, state_dict, action, method=''):
        # Get the latent state
        # state = state.float()
        with torch.autocast("cuda"):
            if method == 'baseline':
                state = state_dict['state']
                curr_latent_state = self.encoder(state)
                future_latent_states = self.dynamics(curr_latent_state, action)
                force_pred = torch.stack([self.predictor(future_latent_states[:, t]) for t in range(self.cfg.horizon)], dim=1)
            elif method == 'rgbt':
                state = state_dict['state']
                img = state_dict['img']
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                curr_latent_state = self.encoder(state, img)
                future_latent_states = self.dynamics(curr_latent_state, action)
                force_pred = torch.stack([self.predictor(future_latent_states[:, t]) for t in range(self.cfg.horizon)], dim=1)
        return future_latent_states, force_pred


    def get_next_state(self, curr_state: torch.Tensor, act:torch.Tensor, dt):
        """ Does a single step from the current state
        Args:
        curr_state: current state
        act: action
        dt: time to integrate
        Returns:
        next_state
        """

        
        if(self.control_space == 'jerk'):
            curr_state[2 * self.n_dofs:3 * self.n_dofs] = curr_state[self.n_dofs:2*self.n_dofs] + act * dt
            curr_state[self.n_dofs:2*self.n_dofs] = curr_state[self.n_dofs:2*self.n_dofs] + curr_state[self.n_dofs*2:self.n_dofs*3] * dt
            
            curr_state[:self.n_dofs] = curr_state[:self.n_dofs] + curr_state[self.n_dofs:2*self.n_dofs] * dt
        elif(self.control_space == 'acc'):
            curr_state[2 * self.n_dofs:3 * self.n_dofs] = act 
            curr_state[self.n_dofs:2*self.n_dofs] = curr_state[self.n_dofs:2*self.n_dofs] + curr_state[self.n_dofs*2:self.n_dofs*3] * dt
            
            curr_state[:self.n_dofs] = curr_state[:self.n_dofs] + curr_state[self.n_dofs:2*self.n_dofs] * dt
        elif(self.control_space == 'vel'):
            curr_state[2 * self.n_dofs:3 * self.n_dofs] = 0.0
            curr_state[self.n_dofs:2*self.n_dofs] = act * dt
            
            curr_state[:self.n_dofs] = curr_state[:self.n_dofs] + curr_state[self.n_dofs:2*self.n_dofs] * dt
        elif(self.control_space == 'pos'):
            curr_state[2 * self.n_dofs:3 * self.n_dofs] = 0.0
            curr_state[1 * self.n_dofs:2 * self.n_dofs] = 0.0
            curr_state[:self.n_dofs] = act
        return curr_state
    def tensor_step(self, state: torch.Tensor, act: torch.Tensor, state_seq: torch.Tensor, dt=None) -> torch.Tensor:
        """
        Args:
        state: [1,N]
        act: [H,N]
        todo:
        Integration  with variable dt along trajectory
        """
        inp_device = state.device
        state = state.to(self.device, dtype=self.float_dtype)
        act = act.to(self.device, dtype=self.float_dtype)
        nth_act_seq = self.integrate_action(act)
        
        
        #print(state.shape)
        state_seq = self.step_fn(state, nth_act_seq, state_seq, self._dt_h, self.n_dofs, self._integrate_matrix, self._fd_matrix)
        #state_seq = self.enforce_bounds(state_seq)
        # timestep array
        state_seq[:,:, -1] = self._traj_tstep

        
        return state_seq
        
        
    def rollout_open_loop_2(self, start_state: torch.Tensor, act_seq: torch.Tensor,
                          dt=None, contact_info=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # batch_size, horizon, d_act = act_seq.shape
        # TODO: pass all the inputs needed by the latent model here
        # x -> joint pos, joint vel, taxel_force
        # get x from start_state and contact_info
        # rgb = contact_info['rgb']
        # depth = contact_info['depth']
        joint_pos = torch.rad2deg(start_state[:, :self.n_dofs])
        joint_vel = start_state[:, self.n_dofs:2*self.n_dofs]
        print ("joint pos tens: ", joint_pos)
        print ("act_seq: ", act_seq.size())
        if 'forces' in contact_info:
            forces = torch.tensor(contact_info['forces'], **self.tensor_args).unsqueeze(0)
        else:
            # create tensor of length 56
            forces = self.zero_forces
        
        x = torch.cat((joint_pos, joint_vel, forces), dim=1)
        # set type to torch.float32
        x = x.float()
        x = x.to('cuda')
        # reshape and concat to x
        state_dict = {'state': x}
        if 'rgb' in contact_info:
            state_dict['img'] = contact_info['rgb']
        
        try:
            future_latent_states, force_pred = self.predict_using_latent_model(state_dict, act_seq, method='rgbt')
        except:
            future_latent_states, force_pred = None, None
        # store future_latent_states and force_pred in state_dict
        # action -> horizon, joint_vel
        
        curr_dt = self.dt if dt is None else dt
        curr_horizon = self.horizon
        # get input device:
        inp_device = start_state.device
        start_state = start_state.to(self.device, dtype=self.float_dtype)
        act_seq = act_seq.to(self.device, dtype=self.float_dtype)
        
        # add start state to prev state buffer:
        #print(start_state.shape, self.d_state)
        if(self.prev_state_buffer is None):
            self.prev_state_buffer = torch.zeros((10, self.d_state), **self.tensor_args)
            self.prev_state_buffer[:,:] = start_state
        self.prev_state_buffer = self.prev_state_buffer.roll(-1, dims=0)
        self.prev_state_buffer[-1,:] = start_state

        
        #print(self.prev_state_buffer[:,-1])
        # compute dt w.r.t previous data?
        state_seq = self.state_seq
        ee_pos_seq = self.ee_pos_seq
        ee_rot_seq = self.ee_rot_seq
        curr_horizon = self.horizon
        curr_batch_size = self.batch_size
        num_traj_points = self.num_traj_points
        link_pos_seq = self.link_pos_seq
        link_rot_seq = self.link_rot_seq

        
        
        curr_state = self.prev_state_buffer[-1:,:self.n_dofs * 3]
 
        with profiler.record_function("tensor_step"):
            # forward step with step matrix:
            state_seq = self.tensor_step(curr_state, act_seq, state_seq, curr_dt)
        
        #print(start_state[:,self.n_dofs*2 : self.n_dofs*3])

        shape_tup = (curr_batch_size * num_traj_points, self.n_dofs)
        with profiler.record_function("fk + jacobian"):
            ee_pos_seq, ee_rot_seq, lin_jac_seq, ang_jac_seq = self.robot_model.compute_fk_and_jacobian(state_seq[:,:,:self.n_dofs].view(shape_tup),
                                                                                                    state_seq[:,:,self.n_dofs:2 * self.n_dofs].view(shape_tup),
                                                                                                    link_name=self.ee_link_name)
        
        # compute linear contact jacobians
        contacts_lin_jac = None
        contacts_norm = None
        # print ("inside rollout open loop contact_info: ", contact_info)
        if 'force' in contact_info:
            print ("Computing contact jacobian")
            contacts_lin_jac = torch.zeros((len(contact_info['force']), curr_batch_size * num_traj_points, 3, self.n_dofs), device=self.device, dtype=self.float_dtype)
            contacts_norm = torch.zeros((len(contact_info['force']), curr_batch_size * num_traj_points, 3), device=self.device, dtype=self.float_dtype)
            for i in range(len(contact_info['force'])):
                contacts_lin_jac[i], contacts_norm[i] = self.robot_model.compute_contact_jacobian(state_seq[:,:,:self.n_dofs].view(shape_tup), state_seq[:,:,self.n_dofs:2 * self.n_dofs].view(shape_tup), contact_info['link'][i], contact_info['pos'][i], contact_info['normal'][i])
                print ("Current contact jac: ", contact_info['jac'][i])
                print ("Computed contact jac: ", contacts_lin_jac[i][0])
                print ("Current contact norm: ", contact_info['normal'][i])
                print ("Computed contact norm: ", contacts_norm[i][0])
                

        # get link poses:
        for ki,k in enumerate(self.link_names):
            link_pos, link_rot = self.robot_model.get_link_pose(k)
            link_pos_seq[:,:,ki,:] = link_pos.view((curr_batch_size, num_traj_points,3))
            link_rot_seq[:,:,ki,:,:] = link_rot.view((curr_batch_size, num_traj_points,3,3))
            
        
        ee_pos_seq = ee_pos_seq.view((curr_batch_size, num_traj_points, 3))
        ee_rot_seq = ee_rot_seq.view((curr_batch_size, num_traj_points, 3, 3))
        lin_jac_seq = lin_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))
        ang_jac_seq = ang_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))
        if 'force' in contact_info:
            contacts_lin_jac = contacts_lin_jac.view((len(contact_info['force']), curr_batch_size, num_traj_points, 3, self.n_dofs))
            contacts_norm = contacts_norm.view((len(contact_info['force']), curr_batch_size, num_traj_points, 3))

        state_dict = {'start_state': start_state.to(inp_device),
                      'state_seq':state_seq.to(inp_device),
                      'ee_pos_seq': ee_pos_seq.to(inp_device),
                      'ee_rot_seq': ee_rot_seq.to(inp_device),
                      'lin_jac_seq': lin_jac_seq.to(inp_device),
                      'ang_jac_seq': ang_jac_seq.to(inp_device),
                      'link_pos_seq':link_pos_seq.to(inp_device),
                      'link_rot_seq':link_rot_seq.to(inp_device),
                      'prev_state_seq':self.prev_state_buffer.to(inp_device)}
        if future_latent_states is not None:
            state_dict['future_latent_states'] = future_latent_states.to(inp_device)
        if force_pred is not None:
            state_dict['pred_forces'] = force_pred.to(inp_device)
        if 'force' in contact_info:
            state_dict['contacts_lin_jac'] = contacts_lin_jac.to(inp_device)
            state_dict['contacts_norm'] = contacts_norm.to(inp_device)
        return state_dict

    def rollout_open_loop(self, start_state: torch.Tensor, act_seq: torch.Tensor,
                          dt=None, contact_info=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # batch_size, horizon, d_act = act_seq.shape
        # TODO: pass all the inputs needed by the latent model here
        # x -> joint pos, joint vel, taxel_force
        # get x from start_state and contact_info
        # rgb = contact_info['rgb']
        # depth = contact_info['depth']
        
        curr_dt = self.dt if dt is None else dt
        curr_horizon = self.horizon
        # get input device:
        inp_device = start_state.device
        start_state = start_state.to(self.device, dtype=self.float_dtype)
        act_seq = act_seq.to(self.device, dtype=self.float_dtype)
        
        # add start state to prev state buffer:
        #print(start_state.shape, self.d_state)
        if(self.prev_state_buffer is None):
            self.prev_state_buffer = torch.zeros((10, self.d_state), **self.tensor_args)
            self.prev_state_buffer[:,:] = start_state
        self.prev_state_buffer = self.prev_state_buffer.roll(-1, dims=0)
        self.prev_state_buffer[-1,:] = start_state

        
        #print(self.prev_state_buffer[:,-1])
        # compute dt w.r.t previous data?
        state_seq = self.state_seq
        ee_pos_seq = self.ee_pos_seq
        ee_rot_seq = self.ee_rot_seq
        curr_horizon = self.horizon
        curr_batch_size = self.batch_size
        num_traj_points = self.num_traj_points
        link_pos_seq = self.link_pos_seq
        link_rot_seq = self.link_rot_seq

        
        
        curr_state = self.prev_state_buffer[-1:,:self.n_dofs * 3]
 
        with profiler.record_function("tensor_step"):
            # forward step with step matrix:
            state_seq = self.tensor_step(curr_state, act_seq, state_seq, curr_dt)
        
        #print(start_state[:,self.n_dofs*2 : self.n_dofs*3])

        shape_tup = (curr_batch_size * num_traj_points, self.n_dofs)
        with profiler.record_function("fk + jacobian"):
            ee_pos_seq, ee_rot_seq, lin_jac_seq, ang_jac_seq = self.robot_model.compute_fk_and_jacobian(state_seq[:,:,:self.n_dofs].view(shape_tup),
                                                                                                    state_seq[:,:,self.n_dofs:2 * self.n_dofs].view(shape_tup),
                                                                                                    link_name=self.ee_link_name)
        # get link poses:
        for ki,k in enumerate(self.link_names):
            link_pos, link_rot = self.robot_model.get_link_pose(k)
            link_pos_seq[:,:,ki,:] = link_pos.view((curr_batch_size, num_traj_points,3))
            link_rot_seq[:,:,ki,:,:] = link_rot.view((curr_batch_size, num_traj_points,3,3))
            
        
        ee_pos_seq = ee_pos_seq.view((curr_batch_size, num_traj_points, 3))
        ee_rot_seq = ee_rot_seq.view((curr_batch_size, num_traj_points, 3, 3))
        lin_jac_seq = lin_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))
        ang_jac_seq = ang_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))

        state_dict = {'start_state': start_state.to(inp_device),
                      'state_seq':state_seq.to(inp_device),
                      'ee_pos_seq': ee_pos_seq.to(inp_device),
                      'ee_rot_seq': ee_rot_seq.to(inp_device),
                      'lin_jac_seq': lin_jac_seq.to(inp_device),
                      'ang_jac_seq': ang_jac_seq.to(inp_device),
                      'link_pos_seq':link_pos_seq.to(inp_device),
                      'link_rot_seq':link_rot_seq.to(inp_device),
                      'prev_state_seq':self.prev_state_buffer.to(inp_device)}

        return state_dict


    def enforce_bounds(self, state_batch):
        """
            Project state into bounds
        """
        batch_size = state_batch.shape[0]
        bounded_state = torch.max(torch.min(state_batch, self.state_upper_bounds), self.state_lower_bounds)
        bounded_q = bounded_state[...,:,:self.n_dofs]
        bounded_qd = bounded_state[...,:,self.n_dofs:2*self.n_dofs]
        bounded_qdd = bounded_state[...,:,2*self.n_dofs:3*self.n_dofs]
        
        # #set velocity and acc to zero where position is at bound
        bounded_qd = torch.where(bounded_q < self.state_upper_bounds[:self.n_dofs], bounded_qd, self.Z)
        bounded_qd = torch.where(bounded_q > self.state_lower_bounds[:self.n_dofs], bounded_qd, self.Z)
        bounded_qdd = torch.where(bounded_q < self.state_upper_bounds[:self.n_dofs], bounded_qdd, -10.0*bounded_qdd)
        bounded_qdd = torch.where(bounded_q > self.state_lower_bounds[:self.n_dofs], bounded_qdd, -10.0*bounded_qdd)

        # #set acc to zero where vel is at bounds 
        bounded_qdd = torch.where(bounded_qd < self.state_upper_bounds[self.n_dofs:2*self.n_dofs], bounded_qdd, self.Z)
        bounded_qdd = torch.where(bounded_qd > self.state_lower_bounds[self.n_dofs:2*self.n_dofs], bounded_qdd, self.Z)
        state_batch[...,:,:self.n_dofs] = bounded_q
        state_batch[...,:,self.n_dofs:self.n_dofs*2] = bounded_qd
        state_batch[...,:,self.n_dofs*2:self.n_dofs*3] = bounded_qdd
        
        #bounded_state = torch.cat((bounded_q, bounded_qd, bounded_qdd), dim=-1) 
        return state_batch

    def integrate_action(self, act_seq):
        if(self.action_order == 0):
            return act_seq

        nth_act_seq = self._integrate_matrix_nth  @ act_seq
        return nth_act_seq

    def integrate_action_step(self, act, dt):
        for i in range(self.action_order):
            act = act * dt
        
        return act

    #Rendering
    def render(self, state):
        q = state[:, 0:self.n_dofs]
        state_dict = {}
        for (i,joint) in enumerate(self.urdfpy_robot.actuated_joints):
            state_dict[joint.name] = q[:,i].item()
        self.urdfpy_robot.show(cfg=state_dict,use_collision=True) 


    def render_trajectory(self, state_list):
        state_dict = {}
        q = state_list[0][:, 0:self.n_dofs]
        for (i,joint) in enumerate(self.urdfpy_robot.actuated_joints):
            state_dict[joint.name] = [q[:,i].item()]
        for state in state_list[1:]:
            q = state[:, 0:self.n_dofs]
            for (i,joint) in enumerate(self.urdfpy_robot.actuated_joints):
                state_dict[joint.name].append(q[:,i].item())
        self.urdfpy_robot.animate(cfg_trajectory=state_dict,use_collision=True) 

