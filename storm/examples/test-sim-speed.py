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
""" Example spawning a robot in gym 

"""
from tkinter import W
import copy
import imp
import re
from turtle import position
from isaacgym import gymapi
from isaacgym import gymutil

from email import policy

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#

import ipdb
import matplotlib
# matplotlib.use('tkagg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import collections

import time
import yaml
import argparse
import numpy as np
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt

from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.contact_task import ContactTask

import sys
import os
import time
import threading

from pyrfuniverse.envs import RFUniverseBaseEnv
from pyrfuniverse.utils.kinova_controller import RFUniverseKinovaController
import pyrfuniverse.assets as assets_path
import math
import pybullet as p
import os
from scipy.spatial.transform import Rotation as R

import cv2
import pickle

import pinocchio
import PyKDL as kdl
import kdl_parser_py.urdf as kdl_parser
import kdl_utils as ku
from kdl_kinematics import create_kdl_kin, KDLKinematics

from scipy.spatial.transform import Rotation as R


class IKTestEnv(RFUniverseBaseEnv):
    def __init__(self):
        super().__init__(
            assets=['Line']
        )
        self.urdf_path = '/home/rishabh/Documents/pyrfuniverse/URDF/kinova_gen3/GEN3_URDF_V12.urdf'
        self.robot_model = pinocchio.buildModelFromUrdf(self.urdf_path)
        self.data = self.robot_model.createData()
        self.ik_controller = RFUniverseKinovaController(
            robot_urdf=os.path.abspath(self.urdf_path)
        )
        self.instance_channel.set_action(
            'GetDepth',
            id=999991,
            width=512,
            height=512,
            zero_dis=1,
            one_dis=5
        )
        self.instance_channel.set_action(
            'GetRGB',
            id=999991,
            width=512,
            height=512
        )
        self.force_scaling_factor = 1500
        
        # Taxel variables
        self.num_taxels = 0
        while True:
            try:
                _ = self.instance_channel.data[1000+self.num_taxels]['force']
                self.num_taxels += 1
            except:
                break
        self.num_taxels -= 1
        self.taxel_ids = [1000+i for i in range(self.num_taxels)]
        self.taxel_positions = np.array([self.instance_channel.data[i]['local_position'] for i in self.taxel_ids])
        self.taxel_links = np.zeros(self.num_taxels, dtype=int)
        link_5_taxel_ids = np.array([1100, 1177, 1333, 1156, 1230, 1009, 1300, 1325, 1239, 1123, 1061, 1015, 1144, 1210, 
             1140, 1271, 1267, 1236, 1130, 1103, 1331, 1111, 1172, 1068, 1286, 1314, 1290, 1295, 
             1329, 1340, 1266, 1057, 1311, 1078, 1129, 1020, 1122, 1344, 1308, 1334, 1178, 1261, 
             1098, 1269, 1262, 1110, 1256, 1182, 1118, 1274, 1297, 1154, 1249, 1232, 1251, 1326, 
             1093, 1119, 1037, 1048, 1060, 1047, 1011, 1024, 1000, 1013, 1234, 1042, 1164, 1348])
        self.taxel_links[link_5_taxel_ids - 1000] = 5
        self.link_joint_dict = {0:[0], 1:[0, 1], 2:[1, 2], 3:[2, 3], 4:[3,4], 5:[4,5], 6:[5,6]}
        self.kdl_kinematics = KDLKinematics(self.urdf_path, "base_link", "EndEffector_Link")
        self.transform = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
        # ok, tree = kdl_parser.treeFromFile(self.urdf_path)
        # self.kdl_chains = self.create_kdl_chains(tree)
        # self.stiffness_percent = 0.75
        # self.joint_stiffness = (np.array([1, 1, 1, 1, 0.06, 0.08, 0.08])*180/np.pi*self.stiffness_percent).tolist()

    def create_kdl_chains(self, tree):
        ch = tree.getChain("base_link", "EndEffector_Link")
        fk, ik_v, ik_p, jac = self.create_solvers(ch)
        kdl_chains = {}
        kdl_chains['chain'] = ch
        kdl_chains['nJnts'] = ch.getNrOfJoints()
        kdl_chains['fk_p'] = fk
        kdl_chains['ik_v'] = ik_v
        kdl_chains['ik_p'] = ik_p
        kdl_chains['jacobian_solver'] = jac
        return kdl_chains

    def kdl_to_mat(self, m):
        mat =  np.mat(np.zeros((m.rows(), m.columns())))
        for i in range(m.rows()):
            for j in range(m.columns()):
                mat[i,j] = m[i,j]
        return mat

    def create_solvers(self, ch):
        fk = kdl.ChainFkSolverPos_recursive(ch)
        ik_v = kdl.ChainIkSolverVel_pinv(ch)
        ik_p = kdl.ChainIkSolverPos_NR(ch, fk, ik_v)
        jac = kdl.ChainJntToJacSolver(ch)
        return fk, ik_v, ik_p, jac

    def joint_list_to_kdl(self, q):
        if q is None:
            return None
        if type(q) == np.matrix and q.shape[1] == 0:
            q = q.T.tolist()[0]
        q_kdl = kdl.JntArray(len(q))
        for i, q_i in enumerate(q):
            q_kdl[i] = q_i
        return q_kdl

    def fk_kdl(self, q, link_number):
        fk_solver = self.kdl_chains['fk_p']
        endeffec_frame = kdl.Frame()
        kinematics_status = fk_solver.JntToCart(self.joint_list_to_kdl(q), endeffec_frame,
                                                link_number)
        if kinematics_status >= 0:
            print ('End effector transformation matrix:', endeffec_frame)
            return endeffec_frame
        else:
            print ('Could not compute forward kinematics.')
            return None

    def fk_vanilla(self, q, link_number = 7):
        # TODO: Check if this is still needed
        # q = -q
        frame = self.fk_kdl(q, link_number)
        pos = frame.p
        pos = ku.kdl_vec_to_np(pos)
        m = frame.M
        rot = ku.kdl_rot_to_np(m)
        return pos, rot

    # ## compute Jacobian at point pos. 
    # # p is in the ground coord frame.
    # def jacobian(self, q, pos=None):
    #     if pos is None:
    #         pos = self.fk_vanilla(q)[0]

    #     ch = self.kdl_chains['chain']
    #     v_list = []
    #     w_list = []

    #     for i in range(self.kdl_chains['nJnts']):
    #         p, rot = self.fk_vanilla(q, i)
    #         r = pos - p
    #         z_idx = ch.getSegment(i).getJoint().getType() - 1
    #         z = rot[:, z_idx]

    #         # this is a nasty trick. The rotation matrix returned by
    #         # FK_vanilla describes the orientation of a frame of the
    #         # KDL chain in the base frame. It just so happens that the
    #         # way Advait defined the KDL chain, the axis of rotation
    #         # in KDL is the -ve of the axis of rotation on the real
    #         # robot for every joint.
    #         # Advait apologizes for this.
    #         # TODO: Check if this is still true.
    #         # z = -z

    #         v_list.append(np.matrix(np.cross(z.A1, r.A1)).T)
    #         w_list.append(z)

    #     J = np.row_stack((np.column_stack(v_list), np.column_stack(w_list)))
    #     return J

    def move(self, pos: list, rot: list):
        joint_positions = self.ik_controller.calculate_ik_recursive(pos, eef_orn=p.getQuaternionFromEuler(rot))
        self._set_kinova_joints(joint_positions)

    def forward_kinematics(self, q_des):
        """
        Compute positions for all joints using forward kinematics given the joint configuration of the robot
        """
        joint_config = np.zeros(self.robot_model.nq)
        joint_config[0:7] = q_des
        pinocchio.forwardKinematics(self.robot_model, self.data, joint_config)
        return self.data.oMi

    def get_jacobian(self, joint_configuration):
        """
        Compute the jacobian for the robot given the joint configuration
        """
        pinocchio.computeJointJacobians(self.robot_model, self.data, joint_configuration)
        return self.data.oJ

    def _set_kinova_joints(self, joint_positions=None, joint_velocities=None):
        if joint_positions is not None:
            self.instance_channel.set_action(
                'SetJointPosition',
                id=315892,
                joint_positions=list(joint_positions[0:7]),
            )
        if joint_velocities is not None:
            self.instance_channel.set_action(
                'SetJointVelocity',
                id=315892,
                joint_velocitys=list(joint_velocities[0:7]),
            )
        self._step()

    def _get_kinova_joint_force(self, object_id: int) -> dict:
        self.instance_channel.set_action(
            "GetJointInverseDynamicsForce",
            id=object_id
        )
        self._step()
        return self.instance_channel.data[object_id]['drive_forces']

    def _get_kinova_joint_pos(self, object_id: int):
        return self.instance_channel.data[object_id]['joint_positions']

    def _get_kinova_joint_vel(self, object_id: int):
        return self.instance_channel.data[object_id]['joint_velocities']

    def _get_kinova_joint_acc(self, object_id: int):
        return self.instance_channel.data[object_id]['joint_accelerations']

    def _get_joint_state(self, object_id: int):
        # TODO: Change acceleration to torque to be consistent with real robot
        joint_state = np.zeros((2, 7), dtype=np.float32)
        joint_state[0] = self._get_kinova_joint_pos(object_id)
        joint_state[1] = self._get_kinova_joint_vel(object_id)
        # joint_state[2] = self._get_kinova_joint_acc(object_id)
        return joint_state

    def _get_target_pose(self):
        target_pose = {}
        target_pose['position'] = self.instance_channel.data[6666]['position']
        target_pose['orientation'] = self.instance_channel.data[6666]['rotation']
        return target_pose

    def _get_skin_data(self):
        # TODO: Change force scaling factor. This will be conditional on avatar and body part?
        forces = np.zeros((self.num_taxels), dtype=np.float64)
        force_positions = np.zeros((self.num_taxels, 3), dtype=np.float64)
        force_normals = np.zeros((self.num_taxels, 3), dtype=np.float64)
        for i in range(self.num_taxels):
            # print ("Before: ", forces[i])
            forces[i] = self.instance_channel.data[1000 + i]['force'] * self.force_scaling_factor #+ np.random.normal(0, 0.1)
            # print ("After: ", forces[i])
            force_positions[i] = np.matmul(self.transform, self.instance_channel.data[1000 + i]['position'])
            rm = R.from_euler('zxy', self.instance_channel.data[1000 + i]['rotation'], degrees=True).as_matrix()
            # TODO: Check if normals are in the right frame
            fn = np.matmul(rm, np.array([0, 0, -1]))
            force_normals[i] = fn # np.matmul(self.transform, fn) / np.linalg.norm(fn)
        return forces, force_positions, force_normals

    def _get_rgb_data(self):
        self.instance_channel.set_action(
            'GetRGB',
            id=999991,
            width=512,
            height=512
        )
        self._step()
        img_bytes = self.instance_channel.data[999991]['rgb']
        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
        return img_array

    def _get_depth_data(self):
        # TODO: Scale depth data to be consistent with actual distance
        # TODO: Fix depth key error
        self.instance_channel.set_action(
            'GetDepth',
            id=999991,
            width=512,
            height=512,
            zero_dis=1,
            one_dis=5
        )
        self._step()
        img_bytes = self.instance_channel.data[999991]['depth']
        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
        return img_array

    def _contact_jacobian(self, joint_config):
        forces, force_positions, force_normals = self._get_skin_data()
        # TODO: Change the force threshold to be a parameter
        force_threshold = 0.5
        idx = [103]
        # print ("Force idx: ", idx, forces[idx])
        # for i in range(len(idx)):
            # print ("Force: ", forces[idx[i]])
            # print ("Force position: ", force_positions[idx[i]])
            # print ("Idx: ", idx[i])
        loc_active_taxels = force_positions[idx]
        link_id = self.taxel_links[idx]
        joint_ids = [self.link_joint_dict[link][0] for link in link_id]
        Jc_l = np.empty((len(loc_active_taxels), 3, 7))
        i = 0
        for jt_li, loc_li in zip(joint_ids, loc_active_taxels):
            Jc = self.kdl_kinematics.jacobian(joint_config, np.array(loc_li))
            Jc[:, jt_li+1:] = 0.0
            Jc = Jc[0:3, :]
            Jc_l[i] = Jc
            i+=1
        contact_info = {'jac': Jc_l, 'loc': loc_active_taxels, 'force': forces[idx], 'normal': force_normals[idx]}
        return contact_info

def get_contact_stiffness_matrix(n_ci):
    # TODO: Change k_default to be a parameter
    k_default = 759.
    n_ci = np.nan_to_num(n_ci)
    # TODO: Fix this operation
    # K_ci = np.outer(n_ci, n_ci)
    K_ci = np.eye(3)
    K_ci = k_default * K_ci
    return K_ci

def compute_rollout_forces(joint_config, q_des, force_normals, contact_jacobians):
    """
    Compute the forces for the rollout
    """
    # TODO: Should be performed for the entire rollout
    deltas = []
    print ("Diff: ", joint_config - q_des)
    # Debug
    # print ("x_dot: ", np.matmul(contact_jacobians[0], joint_config - q_des))
    # return np.matmul(contact_jacobians[0], joint_config - q_des)
    for i, j_ci in enumerate(contact_jacobians):
        n_ci = np.reshape(force_normals[i], (1, -1))
        k_ci = get_contact_stiffness_matrix(n_ci)
        deltas.append(n_ci @ k_ci @ j_ci @ (q_des - joint_config).reshape(-1, 1))
    print ("Deltas: ", deltas)
    return deltas

class DataCollector:
    def __init__(self, env, num_episodes, policy=None) -> None:
        """
        Class to collect data from the gym environment and store it in a dictionary saved as a pickle file.
        Uses a policy to collect RGB, depth, proprioceptive, skin, and action data.
        """
        self.data = []
        self.ep_data = {'rgb': [], 'depth': [], 'prop': [], 'skin': [], 'action': []}
        self.lock = threading.Lock()
        self.env = env
        self.policy = policy
        self.running = False
        self.num_episodes = num_episodes

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def run(self):
        while self.running:
            self.collect_data()

        self.save_data()

    def collect_data(self):
        action = self.policy.get_action()
        self.env._set_kinova_joints(joint_positions=action)
        self.env._step()
        rgb = self.env._get_rgb_data()
        depth = self.env._get_depth_data()
        prop = self.env._get_joint_state(315892)
        skin = self.env._get_skin_data()
        
        self.lock.acquire()
        self.data['rgb'].append(rgb)
        self.data['depth'].append(depth)
        self.data['prop'].append(prop)
        self.data['skin'].append(skin)
        self.data['action'].append(action)
        self.lock.release()

        if self.env.is_done == True:
            self.data.append(self.ep_data)
            self.num_episodes -= 1
            self.ep_data = {'rgb': [], 'depth': [], 'prop': [], 'skin': [], 'action': []}
            self.env.reset()
            if self.num_episodes == 0:
                self.running = False

    def save_data(self):
        with open('data.pickle', 'wb') as f:
            pickle.dump(self.data, f)

np.set_printoptions(precision=4)

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

class Reacher():
    def __init__(self) -> None:
        # start collections with zeros
        self.joint_positions = []
        self.joint_vel = []
        self.q_des = np.zeros(7)
        self.qd_des = np.zeros(7)
        for _ in range(7):
            self.joint_positions.append(collections.deque(np.zeros(10))) # define and adjust figure
            self.joint_vel.append(collections.deque(np.zeros(10)))

        # fig = plt.figure(figsize=(12,6), facecolor='#DEDEDE')
        self.fig, self.axes = plt.subplots(nrows=2, ncols=7, sharex='all', sharey='row',figsize=(30,30))
        self.ax_position = []
        self.ax_vel = []
        self.lines = []
        self.ctr = 0

        for i in range(len(self.axes)):
            for j in range(len(self.axes[i])):
                self.lines.append(self.axes[i][j].plot([], [], 'b-')[0])
                self.axes[i][j].grid('both')
                self.axes[i][j].xaxis.set_tick_params(labelbottom=True)
                self.axes[i][j].yaxis.set_tick_params(labelbottom=True)
                self.axes[i][j].set_ylim(-1, 1)
                self.axes[i][j].set_xlim(0, 3)
        # plt.subplots_adjust(wspace=0.3)

        self.x_data, self.y_data = [] , [[] for i in range(len(self.lines))]
        # self.ani = FuncAnimation(self.fig, self.plot_util, self.init_func)
        self.my_writer= FFMpegWriter(fps=10)


    def init_func(self):
        # get data
        return iter(self.lines)

        
        print("Starting angular action movement ...")
        action = Base_pb2.Action()
        action.name = "Example angular action movement"
        action.application_data = ""

        actuator_count = base.GetActuatorCount()

        # Place arm straight up
        for joint_id in range(actuator_count.count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = 0

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
        base.Unsubscribe(notification_handle)

        if finished:
            print("Angular movement completed")
        else:
            print("Timeout on action notification wait")
        return finished

    def mpc_robot_interactive(self, args, sim_params):
        vis_ee_target = True
        robot_file = args.robot + '.yml'
        task_file = args.robot + '_reacher.yml'
        world_file = 'collision_primitives_3d.yml'

        env = IKTestEnv()        # self.instance_channel.set_action(
        #     "GetJointPositions",
        #     id=object_id
        # )

        world_yml = join_path(get_gym_configs_path(), world_file)
        with open(world_yml) as file:
            world_params = yaml.load(file, Loader=yaml.FullLoader)

        if(args.cuda):
            device = 'cuda'
        else:
            device = 'cpu'

        device = torch.device('cuda', 0) 

        tensor_args = {'device':device, 'dtype':torch.float32}

        w_T_robot = torch.eye(4)
        quat = torch.tensor([1,0,0,0]).unsqueeze(0)
        rot = quaternion_to_matrix(quat)
        w_T_robot[0,3] = 0
        w_T_robot[1,3] = 0
        w_T_robot[2,3] = 0
        w_T_robot[:3,:3] = rot[0]

        mpc_control = ContactTask(task_file, robot_file, world_file, tensor_args)
        n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
        start_qdd = torch.zeros(n_dof, **tensor_args)

        exp_params = mpc_control.exp_params
        
        ee_list = []
        mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}

        franka_bl_state = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                                    0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        x_des_list = [franka_bl_state]
        tgt_p = env._get_target_pose()
        g_pos_ = tgt_p['position']
        transform = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        g_pos = np.matmul(transform, g_pos_)
        g_q = R.from_euler('zxy', tgt_p['orientation'], degrees=True).as_quat()
        # exit(0)
        
        ee_error = 10.0
        j = 0
        t_step = 0
        i = 0
        x_des = x_des_list[0]
        
        # mpc_control.update_params(goal_state=x_des)
        mpc_control.update_params(goal_ee_pos=g_pos, goal_ee_quat=g_q)
    

        g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())

        g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())

        n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
        prev_acc = np.zeros(n_dof)
        # ee_pose = gymapi.Transform()
        w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3,3].unsqueeze(0),
                                            rot=w_T_robot[0:3,0:3].unsqueeze(0))

        rollout = mpc_control.controller.rollout_fn
        tensor_args = mpc_tensor_dtype
        sim_dt = mpc_control.exp_params['control_dt']
        
        log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[],
                    'qddd_des':[]}

        q_des = np.zeros(7)
        qd_des = np.zeros(7)
        t_step = 0 # env._get_sim_time()

        g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())

        # # ipdb.set_trace()
        self.t_now = time.time()
        self.t_cyclic = self.t_now


        j = np.array([6.2583e+00, 3.2867e-01, 3.1477e-03, 6.2809e+00, 2.0326e-02, 4.4613e+00, 1.5781e+00])
        j = np.degrees((j + 2*np.pi) % (2*np.pi))
        # # qd_des = np.degrees(qd_des)
        # # qd_des = np.zeros(7)
        # # qd_des[-1] = 1
        # # qdd_des = np.degrees(qdd_des)
        # # qdd_des = np.array([0,0,0,0,0,0,10])
        # # # self.q_des = np.degrees(q_des)
        # # self.qd_des = np.degrees(qd_des)
        # # self.qd_des = qd_des
        # # print("Vel: ", qd_des)
        # # qd_des = np.degrees(qd_des)
        # # forces, force_positions = env._get_skin_data()
        # # rgb_array = env._get_rgb_data()
        # # depth_array = env._get_depth_data()
        # # prop_array = env._get_joint_state(315892)
        env._set_kinova_joints(joint_positions=j)
        j = np.zeros(7)
        for i in range(7):
            if j[i] > np.pi:
                j[i] -= 2*np.pi
        first_contact = False
        prev_q = None
        prev_x = [0,0,0]
        x_dot = [0,0,0]
        t = time.time()
        f0 = 0
        k = 0
        while True:
        #     # if i > 10:
        #     #     ipdb.set_trace()
            try:
                env._step()
                env.move(env.instance_channel.data[6666]['position'], [0, -math.pi / 2, 0])
                env._step()
                dt = time.time() - t
                t = time.time()
                k += 1
                if k < 100:
                    continue
                # g_pos_ = env._get_kinova_target_pos()
                # transform = np.array([[0, 0, -1], [0, 0, 1], [0, -1, 0]])
                # q_des = np.array([6.2562e+00, 2.8974e-01, 3.4438e-03, 6.2814e+00, 2.2060e-02, 4.4969e+00, 1.5781e+00])
                
                # env._set_kinova_joints(joint_positions=q_des)
                # g_pos = np.matmul(transform, g_pos_)
                # tgt_p = env._get_target_pose()
                # g_pos_ = tgt_p['position']
                # transform = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
                # g_pos = np.matmul(transform, g_pos_)
                # g_q = R.from_euler('zxy', tgt_p['orientation'], degrees=True).as_quat()
                # mpc_control.update_params(goal_ee_pos=g_pos,
                #                         goal_ee_quat=g_q)
                # t_step += sim_dt

                q_des = (np.radians(np.array(env._get_kinova_joint_pos(315892)) % 360.))
                for i in range(7):
                    if q_des[i] > np.pi:
                        q_des[i] -= 2*np.pi
                # print ("q value: ", q_des)
                # qd_des = (np.radians(np.array(env._get_kinova_joint_vel(315892)) % 360.))
                # # qdd_des = np.array(env._get_kinova_joint_acc(315892)) % (2*np.pi)
                # # TODO: Replace this with the actual joint acceleration
                # qdd_des = np.zeros(7)
                # # qd_des = np.array(env._get_kinova_joint_vel(315892))
                # for i in range(7):
                #     if q_des[i] > np.pi:
                #         q_des[i] = q_des[i] - (2 * np.pi)
                #     # elif q_des[i] < np.pi:
                #     #     q_des[i] = q_des[i] + (2 * np.pi)
                # # # TODO: Change current_robot_state
                # # current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
                
                # current_robot_state = {}

                # current_robot_state['position'] = copy.deepcopy(q_des)
                # current_robot_state['velocity'] = copy.deepcopy(qd_des)
                # current_robot_state['acceleration'] = copy.deepcopy(qdd_des)
                # # time_1 = time.time()
                env._step()
                # print ("Config norm: ", np.rad2deg(np.linalg.norm(q_des - j)))
                contact_info = env._contact_jacobian(q_des)
                print ("Contact force: ", contact_info['force'])
                if np.abs(contact_info['force'][0]) > 1.0 and first_contact is False:
                    first_contact = True
                    j = q_des
                    f0 = contact_info['force'][0]
                    # print ("J set to: ", j)
                deltas = compute_rollout_forces(j, q_des, contact_info['normal'], contact_info['jac'])
                # print("Delta gt: ", contact_info['force'][0] - f0)

                # print ("x_dot_gt: ", (contact_info['loc'][0] - prev_x))
                # print ("diff: ", np.linalg.norm((contact_info['loc'][0] - prev_x) - x_dot))
                prev_q = q_des
                prev_x = contact_info['loc'][0]
                
                # # print ("Time taken for contact jacobian: ", time.time() - time_1)
                # # current_robot_state['contact_jacobians'] = contact_jacobians
                # # current_robot_state['forces'] = forces
                # # current_robot_state['force_positions'] = force_positions
                # command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, contact_info=contact_info, WAIT=False)
                # # # print("Time: ", time.time() - t_now)
                # filtered_state_mpc = current_robot_state #mpc_control.current_state
                # # curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

                # # curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
                # # get position command:
                # ee_error = mpc_control.get_current_error(filtered_state_mpc)
                # # time_1 = time.time()
                # # deltas = compute_rollout_forces(q_des, command['position'], force_normals, contact_jacobians)
                # # print ("Time for compute_rollout_forces: ", time.time() - time_1)
                
                # q_des = copy.deepcopy(command['position'])
                # qd_des = copy.deepcopy(command['velocity'])
                # qdd_des = copy.deepcopy(command['acceleration'])
                
                # # # print ("EE error: ", ee_error)
                # # # pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
                
                # # # get current pose:
                # # # e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
                # # # e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
                # # # # ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
                # # # # ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
                
                # # # # ee_pose = copy.deepcopy(w_T_r) * copy.deepcopy(ee_pose)
                
                # # # # if(vis_ee_target):
                # # # #     gym.set_rigid_transform(env_ptr, ee_body_handle, copy.deepcopy(ee_pose))

                # # # # gym_instance.clear_lines()
                # # top_trajs = mpc_control.top_trajs.cpu().float()#.numpy()
                # # n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
                # # w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)


                # # # top_trajs = w_pts.cpu().numpy()
                # # # color = np.array([0.0, 1.0, 0.0, 1.0])
                # # # for k in range(top_trajs.shape[0]):
                # # #     pts = top_trajs[k,:,:]
                # # #     color[0] = float(k) / float(top_trajs.shape[0])
                # # #     color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                # #     # env._draw_lines(pts, color)
                # #     # break
                # #     # env._draw_lines(pts, color=color)
                # # # print("FK: ", env.forward_kinematics(q_des)[0].translation.reshape(-1).tolist())
                # # # joint_pose = env.forward_kinematics(q_des)
                # # # if len(contact_jacobians) > 0:
                # # #     print ("Total contacts: ", len(contact_jacobians))
                
                # j = np.degrees((j + 2*np.pi) % (2*np.pi))
                # # # qd_des = np.degrees(qd_des)
                # # # qd_des = np.zeros(7)
                # # # qd_des[-1] = 1
                # # # qdd_des = np.degrees(qdd_des)
                # # # qdd_des = np.array([0,0,0,0,0,0,10])
                # # # # self.q_des = np.degrees(q_des)
                # # # self.qd_des = np.degrees(qd_des)
                # # # self.qd_des = qd_des
                # # # print("Vel: ", qd_des)
                # # # qd_des = np.degrees(qd_des)
                # # # forces, force_positions = env._get_skin_data()
                # # # rgb_array = env._get_rgb_data()
                # # # depth_array = env._get_depth_data()
                # # # prop_array = env._get_joint_state(315892)
                # env._set_kinova_joints(joint_positions=j)
                # # env._set_kinova_joints(joint_velocities=qd_des)
                # # print("qdd_des: ", qdd_des)
                # # env._set_kinova_acc(qdd_des)
                # t_now = time.time()
                # # env._set_kinova_joints(joint_positions=q_des)
                # # print("Time: ", time.time() - t_now)
                # current_state = command
                # env._step()
                # i += 1
            except KeyboardInterrupt:
                print('Closing')
                # self.ani.save('storm_ani')
                # self.ani.save(filename='storm_sim.mp4', writer=self.my_writer)
                done = True
                break
        mpc_control.close()
        return 1 
    
if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    reacher = Reacher()
    success = True
   

    # success &= example_move_to_home_position(base)
    parser.add_argument('--robot', type=str, default='gen3', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='pos', help='Robot to spawn')
    args = parser.parse_args()
    args.cuda = True
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless

    reacher.mpc_robot_interactive(args, sim_params)
    exit(0 if success else 1)
    
    # mpc_robot_interactive(args, gym_instance)
