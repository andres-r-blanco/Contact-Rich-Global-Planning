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
import copy
from isaacgym import gymapi
from isaacgym import gymutil

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

from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.ActuatorCyclicClientRpc import ActuatorCyclicClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import Session_pb2, ActuatorConfig_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2
from kortex_api.RouterClient import RouterClientSendOptions

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

from sensor_msgs.msg import JointState
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker

from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectoryPoint


def generate_rectangle_trajectory(length, breadth, x, start_y, resolution, start_z):
    points = []
    points.append((x, start_y, start_z))
    # for y in np.arange(start_y, start_y+length, resolution):
    #     points.append((x, y, start_z))
    # for z in np.arange(start_z, start_z+breadth, resolution):
    #     points.append((x, y+length, z))
    # for y in np.arange(start_y+length, start_y, -resolution):
    #     points.append((x, y, start_z+breadth))
    # for z in np.arange(start_z+breadth, start_z, -resolution):
    #     points.append((x, start_y, z))
    return points


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
                'SetJointPositionDirectly',
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

    def _contact_jacobian_torque(self, joint_config, torque):
        forces, force_positions, force_normals = self._get_skin_data()
        forces = [torque[1] / 0.4]
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
        # TODO: Fix output force vector
        contact_info = {'jac': Jc_l, 'loc': loc_active_taxels, 'force': forces, 'normal': force_normals[idx]}
        return contact_info

np.set_printoptions(precision=4)

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return e
 
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
        self.urdf_path = '/home/rishabh/Documents/pyrfuniverse/URDF/kinova_gen3/GEN3_URDF_V12.urdf'
        self.kdl_kinematics = KDLKinematics(self.urdf_path, "base_link", "EndEffector_Link")

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

        rospy.init_node('reacher', anonymous=True)
        self.joint_state_pub = rospy.Publisher('/joint_states', JointState, queue_size=1)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
        # self.taxel1_sub = rospy.Subscriber('/calibration/taxel1', Float32, self.taxel1_callback)
        self.taxel1 = 0
        self.wiper = 0
        self.transform = None
        self.transform2 = None
        self.cmd_pub = rospy.Publisher('/compliant_controller/command', JointTrajectoryPoint, queue_size=1)
        self.cmd = JointTrajectoryPoint()
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        self.joint_pos = None
        self.joint_vel = None
        self.joint_eff = None
        self.cmd_initialized = False

    def joint_state_callback(self, msg):
        self.joint_pos = np.array(msg.position)[1:]
        self.joint_vel = np.array(msg.velocity)[1:]
        self.joint_eff = np.array(msg.effort)[1:]

    def TFThread(self):
        rate = rospy.Rate(1000.0)
        while not rospy.is_shutdown():
            try:
                # self.transform = self.tfBuffer.lookup_transform('base_link', 'taxel1_link', rospy.Time())
                self.transform2 = self.tfBuffer.lookup_transform('base_link', 'wiper_link', rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()
                continue


    def taxel1_callback(self, msg):
        self.taxel1 = msg.data

    def plot_util(self, i):
        self.x_data.append(self.ctr)
        self.ctr+=1
        self.axes[0][0].set_xlim(0, self.ctr)

        y = np.concatenate((self.q_des, self.qd_des))

        for i in range(len(y)):
            self.y_data[i].append(y[i])
        
        for i in range(len(self.axes)):
            ylim = self.axes[i][0].get_ylim()
            update_ylim = False
            for j in range(i*3,i*3+3):
                if y[j] < ylim[0]:
                    ylim = (y[j], ylim[1])
                    update_ylim = True
                if y[j] > ylim[1]:
                    ylim = (ylim[0], y[j])
                    update_ylim = True
            if update_ylim:
                self.axes[i][0].set_ylim(*ylim)

        for i in range(len(self.lines)):
            self.lines[i].set_data(self.x_data, self.y_data[i])
        return iter(self.lines)
    
    def init_func(self):
        # get data
        return iter(self.lines)

    @staticmethod
    def SendCallWithRetry(call, retry,  *args):
        i = 0
        arg_out = []
        while i < retry:
            try:
                arg_out = call(*args)
                break
            except:
                i = i + 1
                continue
        if i == retry:
            print("Failed to communicate")
        return arg_out

    def InitCyclic(self, sampling_time_cyclic):
        print("Init Cyclic")
        sys.stdout.flush()

        print ("Start thread")
        sys.stdout.flush()
        rate = rospy.Rate(1000)
        t_now = 0
        # Publisher is running at ~1000 Hz
        while not rospy.is_shutdown():
            if not self.cmd_initialized:
                continue
            t_now = time.time()
            # Don't forget to uncomment self.q_des init in mpc loop
            # self.q_des = self.joint_pos + 0.001 * self.qd_des
            # self.q_des = self.q_des + 0.001 * self.qd_des
            # TODO: Switch to true desired velocity
            self.cmd.positions = (self.q_des + 2 * np.pi) % (2 * np.pi)
            # self.cmd.velocities = self.qd_des
            self.cmd.time_from_start = rospy.Duration(0.001)
            self.cmd_pub.publish(self.cmd)
            rate.sleep()
        # self.cyclic_thread = threading.Thread(target=self.RunCyclic, args=(sampling_time_cyclic, False))
        # self.cyclic_thread.daemon = False
        # self.cyclic_thread.start()
        return True

    def RunCyclic(self, t_sample, print_stats):
        self.cyclic_running = True
        print("Run Cyclic")
        sys.stdout.flush()
        cyclic_count = 0  # Counts refresh
        stats_count = 0  # Counts stats prints
        failed_cyclic_count = 0  # Count communication timeouts

        t_now = time.time()
        t_cyclic = t_now  # cyclic time
        t_stats = t_now  # print  time
        t_init = t_now  # init   time

        # print("Running torque control example for {} seconds".format(self.cyclic_t_end))

        # TODO: Kill the thread properly
        kill_thread = False
        while not kill_thread:
            if not self.cmd_initialized:
                continue
            t_now = time.time()
            if (t_now - t_cyclic) >= 0.001:
                t_cyclic = t_now
                self.cmd.positions = [0.001, 0.26,  3.14, 3.98, 0.0001,  0.94, 1.57]#self.q_des
                self.cmd.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.cmd.time_from_start = rospy.Duration(0.005)
                # self.cmd_pub.publish(self.cmd)
        return True

    def _contact_jacobian(self, joint_config):
        self.taxels = rospy.wait_for_message("/calibration", Float32MultiArray, timeout=None)
        # self.wiper = rospy.wait_for_message("/forque/forqueSensor", WrenchStamped, timeout=None)
        # set taxel data to 0 below 1.0
        data = np.array(self.taxels.data)
        data[data > 0] = 0.0
        data = np.abs(data)
        # data = np.zeros(7)
        data[data < 1.0] = 0.0
        data[data > 30.0] = 30.0
        print ("Taxel data: ", data)
        forces = torch.Tensor([np.abs(data[4])])
        transform2 = self.transform2
        if transform2 is None:
            return {'jac': [], 'force': forces, 'normal': []}
        loc_active_taxels = [[transform2.transform.translation.x, transform2.transform.translation.y, transform2.transform.translation.z]]
        link_id = 4
        joint_ids = [5]
        Jc_l = np.empty((len(loc_active_taxels), 3, 7))
        i = 0
        ros_rot2 = transform2.transform.rotation
        rot2 = R.from_quat((ros_rot2.x, ros_rot2.y, ros_rot2.z, ros_rot2.w))
        # pos2 = np.matmul(rot2.as_matrix(), [0, -1, 0])
        # find normal vector in the direction of x axis
        pos2 = rot2.as_matrix()[:, 0]
        # pos2 = np.matmul(rot2.as_matrix(), [0, 0, -1])
        # pos2 = np.matmul(rot2.as_matrix(), [1,0,0])
        normal2 = pos2 / np.linalg.norm(pos2)
        force_normals = np.array([normal2])
        for jt_li, loc_li in zip(joint_ids, loc_active_taxels):
            Jc = self.kdl_kinematics.jacobian(joint_config, np.array(loc_li))
            Jc[:, jt_li+1:] = 0.0
            Jc = Jc[0:3, :]
            Jc_l[i] = Jc
            i+=1
        # TODO: Fix output force vector
        contact_info = {'jac': Jc_l, 'loc': loc_active_taxels, 'force': forces, 'normal': force_normals}
        return contact_info 

    def mpc_robot_interactive(self, args, sim_params):
        vis_ee_target = True
        robot_file = args.robot + '.yml'
        task_file = args.robot + '_reacher_real.yml'
        world_file = 'collision_primitives_3d.yml'

        
        sanity_check = False
        while not sanity_check:
            if self.joint_pos is not None and self.joint_vel is not None:
                sanity_check = True
    
        actuator_count = len(self.joint_pos)
        self.actuator_count = len(self.joint_pos)
        robot_state = np.array(self.joint_pos)
        print ("Joint state: ", robot_state)
        for i in range(actuator_count):
            if robot_state[i] > np.pi:
                robot_state[i] = robot_state[i] - (2 * np.pi)
        sim_params['init_state'] = robot_state

        gym_instance = Gym(**sim_params)
        # env = IKTestEnv()
        # env._set_kinova_joints(robot_state)

        # while True:
        #     self.base_feedback = self.base_cyclic.RefreshFeedback()
        #     actuators = self.base_feedback.actuators
        #     robot_state = np.zeros(actuator_count)
        #     for i in range(actuator_count):
        #         robot_state[i] = actuators[i].position

        #     print ("Robot state: ", robot_state)
        #     env._set_kinova_joints(robot_state)
        #     for i in range(2):
        #         env._step()
        #     print ("Sim state: ", env._get_kinova_joint_pos(315892))

        # robot_state = (np.radians(robot_state) + 0.008) % (2 * np.pi)
        # for i in range(actuator_count):
        #     if robot_state[i] > np.pi:
        #         robot_state[i] = robot_state[i] - (2 * np.pi)
        # sim_params['init_state'] = robot_state

        # # exit(0)
        # gym_instance = Gym(**sim_params)

        # joint_pose = collections.deque(np.zeros(10))
        # joint_speed = collections.deque(np.zeros(10))

        # print ("ROBOT pose ", robot_state)

        
        gym = gym_instance.gym
        sim = gym_instance.sim
        world_yml = join_path(get_gym_configs_path(), world_file)
        with open(world_yml) as file:
            world_params = yaml.load(file, Loader=yaml.FullLoader)

        robot_yml = join_path(get_gym_configs_path(),args.robot + '.yml')
        with open(robot_yml) as file:
            robot_params = yaml.load(file, Loader=yaml.FullLoader)
        robot_params['sim_params']['init_state'] = robot_state
        sim_params = robot_params['sim_params']
        sim_params['asset_root'] = get_assets_path()
        if(args.cuda):
            device = 'cuda'
        else:
            device = 'cpu'

        device = torch.device('cuda', 0) 
        sim_params['collision_model'] = None
        tensor_args = {'device':device, 'dtype':torch.float32}
        robot_sim = RobotSim(gym_instance=gym, sim_instance=sim, **sim_params, device=device)
        
        # create gym env
        robot_pose = sim_params['robot_pose']
        env_ptr = gym_instance.env_list[0]
        robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

        # spawn camera:
        robot_camera_pose = np.array([1.6,-1.5, 1.8,0.707,0.0,0.0,0.707])
        # q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
        # robot_camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])        
        robot_sim.spawn_camera(env_ptr, 60, 640, 480, robot_camera_pose)

        # get pose
        w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)
        
        w_T_robot = torch.eye(4)
        quat = torch.tensor([w_T_r.r.w,w_T_r.r.x,w_T_r.r.y,w_T_r.r.z]).unsqueeze(0)
        rot = quaternion_to_matrix(quat)
        w_T_robot[0,3] = w_T_r.p.x
        w_T_robot[1,3] = w_T_r.p.y
        w_T_robot[2,3] = w_T_r.p.z
        w_T_robot[:3,:3] = rot[0]
        world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)
        table_dims = np.ravel([1.5,2.5,0.7])
        cube_pose = np.ravel([0.35, -0.0,-0.35,0.0, 0.0, 0.0,1.0])

        
        
        cube_pose = np.ravel([0.35,0.3,0.4, 0.0, 0.0, 0.0,1.0])
        
        table_dims = np.ravel([0.3,0.1,0.8])
        
        # mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)
        mpc_control = ContactTask(task_file, robot_file, world_file, tensor_args)
        n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
        start_qdd = torch.zeros(n_dof, **tensor_args)

        

        world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)
        # update goal:

        exp_params = mpc_control.exp_params
        
        # current_pose = copy.deepcopy(env._get_kinova_joint_pos(315892))
        # current_vel = copy.deepcopy(env._get_kinova_joint_vel(315892))
        
        ee_list = []
        mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}

        # franka_bl_state = np.array([-0.2774, -1.4971,  1.4359, -0.0709,  0.3143,  0.0521, -0.131,
                                    # 0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                                    # 0.001, 0.47,  3.14, 3.92, 0.0001,  1.254, 1.559
        # franka_bl_state = np.array([0.001, 0.26,  3.14, 3.98, 0.0001,  0.94, 1.57,
        #                             0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        zrs = np.zeros(7, dtype=np.float32)
        franka_bl_state = np.concatenate([robot_state, zrs])

        x_des_list = [franka_bl_state]
        
        ee_error = 10.0
        j = 0
        t_step = 0
        i = 0
        x_des = x_des_list[0]
        
        mpc_control.update_params(goal_state=x_des)

        # spawn object:
        x,y,z = 0, 0, 0
        tray_color = gymapi.Vec3(0.8, 0.1, 0.1)
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002


        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(x, y, z)
        object_pose.r = gymapi.Quat(0,0,0, 1)

        obj_asset_file = "urdf/mug/movable_mug.urdf" 
        obj_asset_root = get_assets_path()
        
        if(vis_ee_target):
            target_object = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_target_object')
            obj_base_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 0)
            obj_body_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 6)
            gym.set_rigid_body_color(env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
            gym.set_rigid_body_color(env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)


            obj_asset_file = "urdf/mug/mug.urdf"
            obj_asset_root = get_assets_path()


            ee_handle = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_current_as_mug')
            ee_body_handle = gym.get_actor_rigid_body_handle(env_ptr, ee_handle, 0)
            tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
            gym.set_rigid_body_color(env_ptr, ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)


        g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())

        g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
        object_pose.p = gymapi.Vec3(g_pos[0], g_pos[1], g_pos[2])

        object_pose.r = gymapi.Quat(g_q[1], g_q[2], g_q[3], g_q[0])
        object_pose = w_T_r * object_pose
        if(vis_ee_target):
            gym.set_rigid_transform(env_ptr, obj_base_handle, object_pose)
        n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
        prev_acc = np.zeros(n_dof)
        ee_pose = gymapi.Transform()
        w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3,3].unsqueeze(0),
                                            rot=w_T_robot[0:3,0:3].unsqueeze(0))

        rollout = mpc_control.controller.rollout_fn
        tensor_args = mpc_tensor_dtype
        sim_dt = mpc_control.exp_params['control_dt']
        
        log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[],
                    'qddd_des':[]}

        q_des = np.zeros(7)
        qd_des = np.zeros(7)
        t_step = gym_instance.get_sim_time()

        g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())

        # # ipdb.set_trace()
        self.t_now = time.time()
        self.t_cyclic = self.t_now
        qdd_des = np.zeros(7)
        t = 0
        last = time.time()
        ctr = 0
        # data = env.robot_model.createData()
        reference_trajectory = generate_rectangle_trajectory(0.3, -0.15, -0.36838533, -0.69243374, 0.01, 0.4)
        ref_id = 0
        while(i > -100):
            # if i > 10:
            #     ipdb.set_trace()
            try:
                if (time.time() - last > 0.01):
                    last = time.time()
                    gym_instance.step()
                    # print ("gym step time: ", time.time() - last)
                    
                    if(vis_ee_target):
                        pose = copy.deepcopy(world_instance.get_pose(obj_body_handle))
                        pose = copy.deepcopy(w_T_r.inverse() * pose)

                        if(np.linalg.norm(g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (np.linalg.norm(g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z]))>0.0)):
                            # g_pos[0] = reference_trajectory[ref_id][0]
                            # g_pos[1] = reference_trajectory[ref_id][1]
                            # g_pos[2] = reference_trajectory[ref_id][2]
                            g_pos[0] = pose.p.x
                            g_pos[1] = pose.p.y
                            g_pos[2] = pose.p.z
                            g_q[1] = pose.r.x
                            g_q[2] = pose.r.y
                            g_q[3] = pose.r.z
                            g_q[0] = pose.r.w
                            mpc_control.update_params(goal_ee_pos=g_pos,
                                                    goal_ee_quat=g_q)
                    # print ("vis time: ", time.time() - last)
                    # env._step()
                    # if(vis_ee_target):
                    #     pose = copy.deepcopy(world_instance.get_pose(obj_body_handle))
                    #     pose = copy.deepcopy(w_T_r.inverse() * pose)

                    #     if(np.linalg.norm(g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (np.linalg.norm(g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z]))>0.0)):
                    #         g_pos[0] = pose.p.x
                    #         g_pos[1] = pose.p.y
                    #         g_pos[2] = pose.p.z
                    #         g_q[1] = pose.r.x
                    #         g_q[2] = pose.r.y
                    #         g_q[3] = pose.r.z
                    #         g_q[0] = pose.r.w
                    # t = time.time()
                    # tgt_p = env._get_target_pose()
                    # g_pos_ = tgt_p['position']
                    # transform = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
                    # g_pos = np.matmul(transform, g_pos_)
                    # g_q = R.from_euler('zxy', tgt_p['orientation'], degrees=True).as_quat()
                    # mpc_control.update_params(goal_ee_pos=g_pos,
                    #                         goal_ee_quat=g_q)
                    t_step += sim_dt
                    q_des = self.joint_pos
                    qd_des = self.joint_vel
                    # TODO: Switch to true values later
                    # qd_des = np.zeros(7)

                    rate = rospy.Rate(1000.0)
                    while not rospy.is_shutdown():
                        try:
                            self.transform2 = self.tfBuffer.lookup_transform('base_link', 'taxel_4', rospy.Time())
                            break
                        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                            rate.sleep()
                            continue
                    # trans = self.tfBuffer.lookup_transform('spherical_wrist_1_link', 'base_link', rospy.Time())
                    # print ("Received quat: ", trans.quaternion)
                    
                    # ros_rot2 = trans2.transform.rotation
                    # rot2 = R.from_quat((ros_rot2.x, ros_rot2.y, ros_rot2.z, ros_rot2.w))
                    # pos2 = np.matmul(rot2.as_matrix(), [0, -1, 0])
                    # normal = pos2 / np.linalg.norm(pos2)
                    # marker_msg = Marker()
                    # marker_msg.header.frame_id = "base_link"
                    # marker_msg.header.stamp = rospy.Time.now()
                    # marker_msg.ns = "my_namespace"
                    # marker_msg.id = 0
                    # marker_msg.type = Marker.SPHERE
                    # marker_msg.action = Marker.ADD
                    # marker_msg.pose.position.x = normal[0]
                    # marker_msg.pose.position.y = normal[1]
                    # marker_msg.pose.position.z = normal[2]
                    # marker_msg.pose.orientation = trans.transform.rotation
                    # marker_msg.color.a = 1.0
                    # marker_msg.scale.x = 0.1
                    # marker_msg.scale.y = 0.1
                    # marker_msg.scale.z = 0.1
                    # self.marker_pub.publish(marker_msg)
                    # print ("Received quat: ", trans.transform.rotation.x)
                    
                    # tau = env.ik_controller.grav_comp(q_des, qd_des)
                    # print ("qdd_des: ", qdd_des)
                    # qd = np.zeros(8)
                    # qd[:7] = qd_des
                    # ptau = pinocchio.rnea(env.robot_model, data, q, qd, pinocchio.utils.zero(env.robot_model.nv))
                    # print ("ptau: ", ptau)
                    # qdd_des = qdd_des - ptau[:7]
                    # print ("External torque: ", qdd_des[1])
                    for i in range(self.actuator_count):
                        if q_des[i] > np.pi:
                            q_des[i] = q_des[i] - (2 * np.pi)
                        if q_des[i] < -np.pi:
                            q_des[i] = q_des[i] + (2 * np.pi)
                
                    # TODO: Change current_robot_state
                    # current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
                    current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
                    current_robot_state['position'] = q_des
                    current_robot_state['velocity'] = qd_des
                    # current_robot_state['acceleration'] = np.zeros(n_dof)

                    contact_info = self._contact_jacobian(q_des)
                    # contact_info = {'jac': [], 'force': [], 'normal': []}
                    # contact_info = env._contact_jacobian_torque(q_des, qdd_des)
                    # contact_info = {'jac': [], 'loc': [], 'force': [], 'normal': []}
                    # last = time.time()
                    command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, contact_info=contact_info, WAIT=False)
                    
                    # print ("Command time: ", time.time() - last)
                    filtered_state_mpc = current_robot_state #mpc_control.current_state
                    curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

                    curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
                    # get position command:
                    q_des = copy.deepcopy(command['position'])
                    qd_des = copy.deepcopy(command['velocity'])
                    qdd_des = copy.deepcopy(command['acceleration'])
                    
                    ee_error = mpc_control.get_current_error(filtered_state_mpc)
                    print ("Position error: ", ee_error[1], " Orientation error: ", ee_error[2])
                    if ee_error[1] < 0.14 and ee_error[2] < 0.05:
                        print ("======== Updating goal pose ========")
                        ref_id = (ref_id + 1) % len(reference_trajectory)
                        # g_pos[0] = reference_trajectory[ref_id][0]
                        # g_pos[1] = reference_trajectory[ref_id][1]
                        # g_pos[2] = reference_trajectory[ref_id][2]
                        g_pos[0] = pose.p.x
                        g_pos[1] = pose.p.y
                        g_pos[2] = pose.p.z
                        g_q[1] = pose.r.x
                        g_q[2] = pose.r.y
                        g_q[3] = pose.r.z
                        g_q[0] = pose.r.w
                        print ("Goal pose: ", g_pos)
                        print ("Goal quat: ", g_q)
                    
                    pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
                    
                    # get current pose:
                    e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
                    e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
                    ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
                    ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
                    
                    ee_pose = copy.deepcopy(w_T_r) * copy.deepcopy(ee_pose)
                    
                    if(vis_ee_target):
                        gym.set_rigid_transform(env_ptr, ee_body_handle, copy.deepcopy(ee_pose))

                    gym_instance.clear_lines()
                    top_trajs = mpc_control.top_trajs.cpu().float()#.numpy()
                    n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
                    w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)


                    top_trajs = w_pts.cpu().numpy()
                    color = np.array([0.0, 1.0, 0.0])
                    for k in range(top_trajs.shape[0]):
                        pts = top_trajs[k,:,:]
                        color[0] = float(k) / float(top_trajs.shape[0])
                        color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                        gym_instance.draw_lines(pts, color=color)

                    # self.q_des = q_des
                    # print ("initial q_des: ", q_des)
                    # exit(0)
                    # if not self.cmd_initialized:
                    self.q_des = (q_des + 2*np.pi) % (2*np.pi)
                    # # self.q_des = np.degrees(q_des)
                    self.qd_des = qd_des
                    # # self.qd_des = qd_des
                    # # print("Vel: ", self.qd_des)
                    # print ("Cmd des pos: ", self.qd_des)
                    # self.cmd.positions = self.q_des #self.q_des
                    # self.cmd.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    # self.cmd.time_from_start = rospy.Duration(0.005)
                    # self.cmd_pub.publish(self.cmd)

                    q_des = self.joint_pos
                    qd_des = self.joint_vel
                    for i in range(self.actuator_count):
                        if q_des[i] > np.pi:
                            q_des[i] = q_des[i] - (2 * np.pi)
                        if q_des[i] < -np.pi:
                            q_des[i] = q_des[i] + (2 * np.pi)
                    robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
                    # robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)
                    # robot_sim.command_robot_velocity(qd_des, env_ptr, robot_ptr)
                    ctr += 1
                    if ctr > 2:
                        ctr -= 1
                        self.cmd_initialized = True
                    current_state = command
                    i += 1
                    # print ("Time taken: ", time.time() - last)
                    t = time.time()
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
    parser.add_argument("--ip", type=str, help="IP address of destination", default="192.168.2.4")
    parser.add_argument("-u", "--username", type=str, help="username to login", default="admin")
    parser.add_argument("-p", "--password", type=str, help="password to login", default="admin")
    args = parser.parse_args()
    
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    reacher = Reacher()
    # Parse arguments
    args = utilities.parseConnectionArguments()
    success = True
    # with utilities.DeviceConnection.createTcpConnection(args) as router:
    #     with utilities.DeviceConnection.createUdpConnection(args) as router_real_time:
    #         # Create required services
    #         device_manager = DeviceManagerClient(router)
    #         reacher.actuator_config = ActuatorConfigClient(router)

    #         reacher.base = BaseClient(router)
    #         reacher.base_cyclic = BaseCyclicClient(router_real_time)
    #         reacher.base_command = BaseCyclic_pb2.Command()
    #         reacher.base_feedback = BaseCyclic_pb2.Feedback()
    #         reacher.base_custom_data = BaseCyclic_pb2.CustomData()
            
    #         device_handles = device_manager.ReadAllDevices()
    #         for handle in device_handles.device_handle:
    #             if handle.device_type == Common_pb2.BIG_ACTUATOR or handle.device_type == Common_pb2.SMALL_ACTUATOR:
    #                 reacher.base_command.actuators.add()
    #                 reacher.base_feedback.actuators.add()

    #         reacher.sendOption = RouterClientSendOptions()
            # reacher.sendOption.andForget = False
            # reacher.sendOption.delay_ms = 0
            # reacher.sendOption.timeout_ms = 3

            # Example core
            # success = True

            # success &= example_move_to_home_position(base)
    parser.add_argument('--robot', type=str, default='gen3', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='pos', help='Robot to spawn')
    args = parser.parse_args()
    args.cuda = True
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless
    # success &= example_cartesian_action_movement(base, base_cyclic)
    
    reacher.t_sample = 0.001
    reacher.kill_thread = False
    init_thread = threading.Thread(target=reacher.InitCyclic, args=(0.001,))
    init_thread.daemon = True
    init_thread.start()
    # tf_thread = threading.Thread(target=reacher.TFThread)
    # tf_thread.daemon = True
    # tf_thread.start()
    # while not reacher.kill_thread:
    #     try:
    #         time.sleep(0.5)
    #     except KeyboardInterrupt:
    #         break
    reacher.mpc_robot_interactive(args, sim_params)
            # success &= example_angular_action_movement(base)


    exit(0 if success else 1)
    
    # mpc_robot_interactive(args, gym_instance)
