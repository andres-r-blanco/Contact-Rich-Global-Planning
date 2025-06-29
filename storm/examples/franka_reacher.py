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
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

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
from storm_kit.mpc.task.reacher_task import ReacherTask

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

# np.set_printoptions(precision=4)

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
    return check
 
class Reacher():
    def example_move_to_home_position(self, base):
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        
        # Move arm to ready position
        print("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle == None:
            print("Can't reach safe position. Exiting")
            return False

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        base.ExecuteActionFromReference(action_handle)
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        if finished:
            print("Safe position reached")
        else:
            print("Timeout on action notification wait")
        return finished

    def example_angular_action_movement(self, base):
        
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

        # if not self.example_move_to_home_position(self.base):
        #     return False

        actuator_count = self.base.GetActuatorCount().count
        self.actuator_count = actuator_count
        self.kill_thread = False
        self.q_des = np.zeros(actuator_count)
        self.q_des += 0.001
        self.qd_des = np.zeros(actuator_count)
        self.qd_des +=  0.001
        base_feedback = self.SendCallWithRetry(self.base_cyclic.RefreshFeedback, 3)
        if base_feedback:
            self.base_feedback = base_feedback
            # Init command frame
            for x in range(actuator_count):
                self.base_command.actuators[x].flags = 1  # servoing
                self.base_command.actuators[x].position = base_feedback.actuators[x].position
                self.q_des[x] = base_feedback.actuators[x].position

            # # First actuator is going to be controlled in torque
            # # To ensure continuity, torque command is set to measure
            # self.base_command.actuators[0].torque_joint = self.base_feedback.actuators[0].torque

            # Set arm in LOW_LEVEL_SERVOING
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)

            # Send first frame
            self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)

            # Set first actuator in torque mode now that the command is equal to measure
            control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
            control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
            device_id = 1  # first actuator as id = 1

            for x in range(self.actuator_count):
                self.SendCallWithRetry(self.actuator_config.SetControlMode, 3, control_mode_message, x+1)

            # Init cyclic thread
            print ("Start thread")
            sys.stdout.flush()
            self.cyclic_thread = threading.Thread(target=self.RunCyclic, args=(sampling_time_cyclic, False))
            self.cyclic_thread.daemon = True
            self.cyclic_thread.start()
            
            # while True:
            #     self.RunCyclic(sampling_time_cyclic, True)
            return True

        else:
            print("InitCyclic: failed to communicate")
            return False

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
            t_now = time.time()

            # Cyclic Refresh
            # print ("time: ", t_now - t_cyclic)
            if (t_now - t_cyclic) >= t_sample:

                # Position command to first actuator is set to measured one to avoid following error to trigger
                # Bonus: When doing this instead of disabling the following error, if communication is lost and first
                #        actuator continue to move under torque command, resulting position error with command will
                #        trigger a following error and switch back the actuator in position command to hold its position
                # self.base_command.actuators[0].position = self.base_feedback.actuators[0].position

                # # First actuator torque command is set to last actuator torque measure times an amplification
                # self.base_command.actuators[0].torque_joint = init_first_torque + \
                #     self.torque_amplification * (self.base_feedback.actuators[self.actuator_count - 1].torque - init_last_torque)

                # # First actuator position is sent as a command to last actuator
                # self.base_command.actuators[self.actuator_count - 1].position = self.base_feedback.actuators[0].position - init_delta_position

                # # Incrementing identifier ensure actuators can reject out of time frames
                self.base_command.frame_id += 1
                if self.base_command.frame_id > 65535:
                    self.base_command.frame_id = 0
                for i in range(self.actuator_count):
                    self.base_command.actuators[i].command_id = self.base_command.frame_id

                # print ("Sent: ")
                for i in range(self.actuator_count):
                    # self.base_command.actuators[i].position = self.base_feedback.actuators[i].position
                    # if i == self.actuator_count - 1:
                         # self.q_des[i]
                        # self.base_command.actuators[i].velocity = 0.1 * self.qd_des[i]
                    self.base_command.actuators[i].position = self.q_des[i]
                    # self.base_command.actuators[i].position = self.base_feedback.actuators[i].position + (float(t_now - t_cyclic) * self.qd_des[i])
                    # self.base_command.actuators[i].position = (self.base_command.actuators[i].position + 360.0) % (360.0)
                #     print (self.base_command.actuators[i].position, ", ")
                # print("\n")
                # Frame is sent
                try:
                    self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)
                except:
                    failed_cyclic_count = failed_cyclic_count + 1
                t_cyclic = t_now
                # print ("Failed: ", failed_cyclic_count)
                cyclic_count = cyclic_count + 1

        self.cyclic_running = False
        return True

    def mpc_robot_interactive(self, args, sim_params):
        vis_ee_target = True
        robot_file = args.robot + '.yml'
        task_file = args.robot + '_reacher.yml'
        world_file = 'collision_primitives_3d.yml'

        actuator_count = self.base.GetActuatorCount().count
        actuators = self.base_feedback.actuators
        robot_state = np.zeros(actuator_count)
        for i in range(actuator_count):
            robot_state[i] = (np.radians(actuators[i].position) + 0.01) % (2*np.pi)#- np.pi
        sim_params['init_state'] = robot_state

        gym_instance = Gym(**sim_params)

        print ("ROBOT pose ", robot_state)

        
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

        sim_params['collision_model'] = None
        # create robot simulation:
        robot_sim = RobotSim(gym_instance=gym, sim_instance=sim, **sim_params, device=device)

        
        # create gym environment:
        robot_pose = sim_params['robot_pose']
        env_ptr = gym_instance.env_list[0]
        robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

        device = torch.device('cuda', 0) 

        tensor_args = {'device':device, 'dtype':torch.float32}
        

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
        

        # get camera data:
        
        mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)

        
        n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs

        
        start_qdd = torch.zeros(n_dof, **tensor_args)

        # update goal:

        exp_params = mpc_control.exp_params
        
        current_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
        ee_list = []
        

        mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}

        franka_bl_state = np.array([6.2831, 0.3603, 6.2832, 0.1004, 6.2831, 0.1112, 6.2829,
                                    0.0,0.0,0.0,0.0,0.0,0.0,0.0])
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

        q_des = None
        qd_des = None
        t_step = gym_instance.get_sim_time()

        g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())

        # ipdb.set_trace()
        self.t_now = time.time()
        self.t_cyclic = self.t_now
        while(i > -100):
            # if i > 10:
            #     ipdb.set_trace()
            try:
                gym_instance.step()
                if(vis_ee_target):
                    pose = copy.deepcopy(world_instance.get_pose(obj_body_handle))
                    pose = copy.deepcopy(w_T_r.inverse() * pose)

                    if(np.linalg.norm(g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (np.linalg.norm(g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z]))>0.0)):
                        g_pos[0] = pose.p.x
                        g_pos[1] = pose.p.y
                        g_pos[2] = pose.p.z
                        g_q[1] = pose.r.x
                        g_q[2] = pose.r.y
                        g_q[3] = pose.r.z
                        g_q[0] = pose.r.w

                        mpc_control.update_params(goal_ee_pos=g_pos,
                                                goal_ee_quat=g_q)
                t_step += sim_dt
                current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
                

                
                command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)

                filtered_state_mpc = current_robot_state #mpc_control.current_state
                curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

                curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
                # get position command:
                q_des = copy.deepcopy(command['position'])
                qd_des = copy.deepcopy(command['velocity']) * 0.5
                qdd_des = copy.deepcopy(command['acceleration'])
                
                ee_error = mpc_control.get_current_error(filtered_state_mpc)
                
                pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
                
                # get current pose:
                e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
                e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
                ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
                ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
                
                ee_pose = copy.deepcopy(w_T_r) * copy.deepcopy(ee_pose)
                
                if(vis_ee_target):
                    gym.set_rigid_transform(env_ptr, ee_body_handle, copy.deepcopy(ee_pose))

                print(["{:.3f}".format(x) for x in ee_error], "{:.3f}".format(mpc_control.opt_dt),
                    "{:.3f}".format(mpc_control.mpc_dt))
            
                
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
                
                # robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)
                
                self.q_des = q_des
                for x in range(self.actuator_count):
                    q_des[x] = (self.base_feedback.actuators[x].position)
                q_des = (np.radians(q_des)) % (2 * np.pi)
                print ("Pose: ", q_des)
                print ("Vel: ", qd_des)
                q_des += (qd_des * 0.0001)
                print ("New pose: ", q_des)
                # q_des = q_des % (2 * np.pi)
                # q_des = np.clip(np.float32(q_des), mpc_control.controller.rollout_fn.dynamics_model.state_lower_bounds[:7].cpu().numpy(), mpc_control.controller.rollout_fn.dynamics_model.state_upper_bounds[:7].cpu().numpy())

                # self.q_des += 0.1 * qd_des
                # print ("qd_des: ", self.qd_des)
                # for x in range(self.actuator_count):
                #     if self.q_des[x] < 0:
                #         self.q_des[x] += (2 * np.pi)
                # self.q_des = self.q_des % (2 * np.pi)
                self.q_des = np.degrees(self.q_des)
                # self.q_des = (np.degrees(self.q_des)) % 360.0
                print ("Joints ", self.q_des)
                # print ("qdes: ", self.q_des)
                # self.q_des = np.degrees(self.q_des) % 360.0
                

                # self.t_now = time.time()
                # if (self.t_now - self.t_cyclic) >= self.t_sample:
                #     self.t_cyclic = self.t_now
                #     for i in range(self.actuator_count):
                #         # if i == self.actuator_count - 5:
                #             # self.base_command.actuators[i].position = 0.02 # self.base_feedback.actuators[i].position
                #         base_command.actuators[i].position = np.degrees(self.q_des[i] - 0.001)
                #         base_command.actuators[i].position = base_command.actuators[i].position % 360.0
                #         # base_command.actuators[i].velocity = 0.0001 * np.degrees(self.q_des[i]) % (360.0)
                    
                #     # Frame is sent
                #     try:
                #         base_feedback = base_cyclic.Refresh(base_command, 0, self.sendOption)
                #     except:
                #         print("Could not send command")
                
                
                # print ("Q_des: ", q_des)
                q_des = (np.radians(q_des)) % (2 * np.pi)
                robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)
                exit(0)
                # action = Base_pb2.Action()
                # action.name = "Example angular action movement"
                # action.application_data = ""

                # actuator_count = base.GetActuatorCount()

                # # Place arm straight up
                # for joint_id in range(actuator_count.count):
                #     joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
                #     joint_angle.joint_identifier = joint_id
                #     joint_angle.value = q_des[joint_id] - np.pi

                # base.ExecuteAction(action)
                #robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
                current_state = command
                
                i += 1

                

                
            except KeyboardInterrupt:
                print('Closing')
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
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        with utilities.DeviceConnection.createUdpConnection(args) as router_real_time:
            # Create required services
            device_manager = DeviceManagerClient(router)
            reacher.actuator_config = ActuatorConfigClient(router)

            reacher.base = BaseClient(router)
            reacher.base_cyclic = BaseCyclicClient(router_real_time)
            reacher.base_command = BaseCyclic_pb2.Command()
            reacher.base_feedback = BaseCyclic_pb2.Feedback()
            reacher.base_custom_data = BaseCyclic_pb2.CustomData()
            
            device_handles = device_manager.ReadAllDevices()
            for handle in device_handles.device_handle:
                if handle.device_type == Common_pb2.BIG_ACTUATOR or handle.device_type == Common_pb2.SMALL_ACTUATOR:
                    reacher.base_command.actuators.add()
                    reacher.base_feedback.actuators.add()

            reacher.sendOption = RouterClientSendOptions()
            reacher.sendOption.andForget = False
            reacher.sendOption.delay_ms = 0
            # reacher.sendOption.timeout_ms = 3

            # Example core
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
            # success &= example_cartesian_action_movement(base, base_cyclic)
            
            reacher.t_sample = 0.001
            reacher.kill_thread = False
            init_thread = threading.Thread(target=reacher.InitCyclic, args=(0.001,))
            init_thread.daemon = True
            init_thread.start()
            # while not reacher.kill_thread:
            #     try:
            #         time.sleep(0.5)
            #     except KeyboardInterrupt:
            #         break
            reacher.mpc_robot_interactive(args, sim_params)
            # success &= example_angular_action_movement(base)


    exit(0 if success else 1)
    
    # mpc_robot_interactive(args, gym_instance)
