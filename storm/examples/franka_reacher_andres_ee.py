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

import os
import sys

# Get root directory (assumes you're running from /Andres directory)
ROOT_DIR = r"/home/rishabh/Andres/"

# Add necessary folders to the Python path
sys.path.insert(0, os.path.join(ROOT_DIR, 'storm'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'storm', 'scripts'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'Manip_planning', 'mp-osc','multipriority'))  # adjust if needed
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'Manip_planning', 'mp-osc','pybullet_planning_master'))
sys.path.insert(1, os.path.join(ROOT_DIR, 'Manip_planning', 'mp-osc','multipriority', 'urdfs'))

import copy
import csv
from isaacgym import gymapi
from isaacgym import gymutil

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import open3d as o3d
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import time
import yaml
import argparse
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import math
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt

import pybullet as p
import pybullet_data
from pybullet_tools.utils import add_data_path, create_box, create_cylinder, quat_from_euler, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF
from storm_pybullet_utils import (
    wrap_to_pi, angle_diff, interpolate_configs, set_joint_positions,
    move_robot, q_to_endeff, compute_signed_distance_for_link,
    correct_J_form, calculate_taxel_manip_and_dist,vis_waypoints, 
    get_js_waypoint_list, draw_waypoints, set_robot_state, save_log_to_csv, get_endeff
)
from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
# from multipriority.envs.toy_env_sim import ToyOneEnv

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask
from storm_kit.mpc.task.contact_task import ContactTask

import numpy as np
# import rospy
import time
# from std_msgs.msg import Float64MultiArray
# from trajectory_msgs.msg import JointTrajectoryPoint

LOGGER_RUNNING = False
TACTILE_KINOVA_URDF = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12_w_taxels.urdf"
CSV_FOLDER_LOCATION = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/data/manip_data/new_reach_over_body"
CSV_FILE_NAME = "nrob_weight1_contactsamplechance0.0_objreduction0.0_Min Iterations4000"
JS_WAYPOINT_CSV_PATH = os.path.join(CSV_FOLDER_LOCATION, CSV_FILE_NAME)
JOINT_INDICES = [0,1,2,3,4,5,6]
PAUSE = True
VIZ_WAYPOINTS = False
DISTANCE_THRESHOLD = 0.2
# from multipriority.envs.toy_env_ros import RealToyEnv
# from multipriority.utils import Tracker, load_yaml, simulate_feedback, simulate_simple_feedback, load_trajectory_jointspace, simulate_last_feedback
# from multipriority.controllers import TaskController, ContactController, MultiPriorityController
# from multipriority.bandit import PriorityLinUCB
# from collections import deque
        
def setup_pybullet_env(sim_type = 3):
    if p.isConnected():
        p.disconnect()
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable GUI elements
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # Ensure rendering is enabled
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # Optional
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # Optional
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)  # Ensure RGB rendering is enabled

    add_data_path(pybullet_data.getDataPath())

    plane_id = p.loadURDF("plane.urdf") 
    robot_id = p.loadURDF(TACTILE_KINOVA_URDF, [0, 0, 0],useFixedBase=1)
    
    if sim_type == 3:
        set_camera_pose(camera_point=[0.9, 0.2, 1], target_point = [0.35, -0.2, 0.13])
        box1_position = [0.35, -0.3, 0.13]
        box1_dims = [0.26,1,0.14]
        col_box_id1 = create_box(box1_position,box1_dims)
        cyl_position1 = (0.35,-0.3,0.3)
        cyl_quat1 = p.getQuaternionFromEuler([3.14/2, 0, 0])
        rad1 = 0.18
        h1 = 1 
        col_cyl_id1 = create_cylinder(rad1, h1, cyl_position1, cyl_quat1)
        p.changeVisualShape(col_cyl_id1, -1, rgbaColor=[0.2,0.2,0.2,1])
        obstacles = [col_cyl_id1,col_box_id1]
    elif sim_type == 6:
        set_camera_pose(camera_point=[-0.3, 0.3, 1.1], target_point = [0.4, 0, 0.7])
        cyl_position1 = (0.45, -0.09, 0.5)
        cyl_quat1 = p.getQuaternionFromEuler([0, 0, 0])
        rad1 = 0.1
        h1 = 1
        col_cyl_id1 = create_cylinder(rad1, h1, cyl_position1, cyl_quat1)
        cyl_position2 = (0.22, 0.2, 0.5)
        cyl_quat2 = p.getQuaternionFromEuler([0, 0, 0])
        rad2 = 0.1
        h2 = 1
        col_cyl_id2 = create_cylinder(rad2, h2, cyl_position2, cyl_quat2)
        cyl_position3 = (0.15, -0.19, 0.5)
        cyl_quat3 = p.getQuaternionFromEuler([0, 0, 0])
        rad3 = 0.1
        h3 = 1
        col_cyl_id3 = create_cylinder(rad3, h3, cyl_position3, cyl_quat3)
        obstacles = [col_cyl_id1,col_cyl_id2,col_cyl_id3]
    else:
        raise ValueError(f"Invalid sim_type {sim_type}. Anything other than 3 or 6 not set up yet.")

    return obstacles, robot_id, plane_id


np.set_printoptions(precision=2)

def mpc_robot_interactive(args, gym_instance):
    # Logger setup for metrics
    save_data = True
    output_path = args.output_save_file
    if output_path == '':
        save_data = False
    
    #
    vis_ee_target = True
    robot_file = args.robot + '.yml'
    task_file = args.robot + '_reacher_new_andres_ee.yml'
    world_file = 'collision_primitives_3d_andres_reach.yml'
    
    if save_data:
        if not output_path.endswith('.csv'):
            wait_for_user(f"Output file {output_path} does not end with .csv, please check the path and try again.")
        # Check if the output CSV file already exists
        if os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Replacing it.")
        else:
            print(f"Output file {output_path} does not exist. Creating it.")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Create or replace the file with a blank CSV
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
    trial_num = args.trial_num
    for trial in range(trial_num):
        torch.manual_seed(10*trial)  # Add trial or timestamp
        np.random.seed(10*trial)
        
        if args.g_pos is None or args.init_config is None or args.g_pos_state is None:
            raise ValueError("Please provide --g_pos, --g_pos_state and --init_config arguments.")
        
            
        world_params = 0
        world_yml = join_path(get_gym_configs_path(), world_file)
        with open(world_yml, encoding='utf-8') as file:
            world_params = yaml.load(file, Loader=yaml.FullLoader)
            
        robot_params = 0
        robot_yml = join_path(get_gym_configs_path(),args.robot + '.yml')
        with open(robot_yml, encoding='utf-8') as file:
            robot_params = yaml.load(file, Loader=yaml.FullLoader)
        sim_params = robot_params['sim_params']
        sim_params['asset_root'] = get_assets_path()
        if(args.cuda):
            device = 'cuda'
        else:
            device = 'cpu'
            
        # js_waypoint_list = get_js_waypoint_list(JS_WAYPOINT_CSV_PATH, 0)
        # if js_waypoint_list is None:
        #     raise ValueError(f"js_waypoint_list is None. Check your CSV file at {JS_WAYPOINT_CSV_PATH} and the row number {trial*11-row_adjust}.")
        # # js_waypoint_list = js_waypoint_list[6:]
        # #add waypoints so that all are at most some distance away
        # # js_waypoint_list = [[-3.14, -3.13, -3.07,  3.13, -2.99,  3.06,  2.42],[-3.14, -3.13, -3.07,  3.13, -2.99,  3.06,  2.42]]
        # # js_waypoint_list = [[0.2,1.5,0.1,0.1,0.1,0.1,0.1],[0.2,0.5,0.1,0.1,0.1,0.1,0.1]]
        # # if VIZ_WAYPOINTS: 
        # #     vis_waypoints(js_waypoint_list, robot_id, interpolate = True)
        # #     wait_for_user("Press Enter to continue...")
        # # else:
        # #     draw_waypoints(js_waypoint_list, robot_id)
        
        # print(js_waypoint_list)
        
        # if not js_waypoint_list or len(js_waypoint_list) == 0:
        #     raise ValueError("js_waypoint_list is empty. Check your CSV file and path.")
        # init_config = js_waypoint_list[0]   # Get the first row of joint positions

        init_tensor = torch.tensor(args.init_config, device=device, dtype=torch.float32)
        noise = 0.001 * torch.randn_like(init_tensor)
        init_config = (init_tensor + noise).tolist()
        sim_params['init_state'] = init_config

        sim_params['collision_model'] = None
        
        # create gym environment:
        robot_pose = sim_params['robot_pose']


        device = torch.device('cuda', 0) 
        tensor_args = {'device':device, 'dtype':torch.float32}

        # get camera data:
        
        obstacles, robot_id, plane_id = setup_pybullet_env(args.sim_type)
        
        mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)

        row_adjust = 0
        # for trial in range(trial_num):
        print(f"trial {trial} of {trial_num}")

        mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}
        
        # g_pos = q_to_endeff(robot_id, js_waypoint_list[-1])[0]
        g_pos = args.g_pos
        set_robot_state(robot_id, init_config)
        x_des = np.concatenate((init_config, np.zeros(7, dtype=np.float32)))
        # x_des_list = [np.concatenate((q, np.zeros(7, dtype=np.float32))) for q in js_waypoint_list]
        # x_des_list = [x_des_list[0], x_des_list[-1]]
        
        goal_index = 1
        max_goals =2
        next_waypoint_threshold = 0.1  # radian L2 threshold
        # max_goals = len(x_des_list)
        
        t_step = 0
        i = 0
        
        # mpc_control.update_params(goal_ee_pos=g_pos,
                                            # goal_ee_quat=None)
        goal_state = np.concatenate((args.g_pos_state, np.zeros(7, dtype=np.float32)))
        mpc_control.update_params(goal_state=goal_state)
        # mpc_control.update_params(goal_state=x_des)

        rollout = mpc_control.controller.rollout_fn
        tensor_args = mpc_tensor_dtype
        sim_dt = mpc_control.exp_params['control_dt']
        
        log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[],
                    'qddd_des':[]}

        # torch.set_num_threads(4)
        
        q_des = None
        qd_des = None
        t_step = 0 #gym_instance.get_sim_time()
        buffer_margin = 10 * sim_dt

        # g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        # g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
        # old_joint_pos = np.array(unity_robot_sim.robot.current_joint_positions)
        print(f"\nInitial joint position: {init_config}")
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)
        # print(f"Trying to reach goal {goal_index} out of {len(x_des_list)-1}: {x_des_list[goal_index]}\n")
        current_robot_state = {'position':[], 'velocity':[], 'acceleration':np.zeros(7, dtype=np.float32)}
        counter = 0 
        
        #data save stuff
        start_time = time.time()
        log_data = {
            'time': [],
            'joint_position': [],
            'joint_norm_distance_so_far': [],
            'manipulability': [],
            'distance_to_taxel': [],
            'closest_taxel_id': [],
        }
        cumulative_distance = 0.0
        previous_q = init_config
        
        ee_error = 1000
        ee_error_threshold = 0.05  # meters
        
        stuck_counter = 0
        max_stuck_counter = 20  # Number of steps to consider the robot stuck
        got_stuck = False
        while(goal_index < max_goals):
            try:
                # gym_instance.step()
                # p.stepSimulation()
                # time.sleep(sim_dt)
                
                t_step += sim_dt
                counter += 1
                # command_tstep = mpc_control.control_process.command_tstep
                # if len(command_tstep) > 0 and (t_step > command_tstep[-1] - buffer_margin):
                #     print("Re-optimizing MPC to refresh buffer.")
                #     mpc_control.update_params(goal_state=x_des_list[goal_index])
                #     while t_step > mpc_control.control_process.command_tstep[-1] - buffer_margin:
                #         print("Waiting for MPC to reoptimize...")
                #         time.sleep(0.01)
                
                current_robot_state['position'] = wrap_to_pi(np.array([p.getJointState(robot_id, i)[0] for i in JOINT_INDICES]))
                current_robot_state['velocity'] = np.array([p.getJointState(robot_id, i)[1] for i in JOINT_INDICES])

                # print(f"t_step: {t_step}")
                # print(f"curr_state: {current_robot_state}")
                # print(f"control_dt: {sim_dt}")
                # print(f"contact_info: {None}")
                # print(f"WAIT: {False}")```1122`
                
                # start= time.time()
                command = mpc_control.get_command(
                        t_step,
                        current_robot_state,
                        control_dt=sim_dt,
                        contact_info=None,  # no contact info used here
                        WAIT=True
                    )
                # print(f"MPC step took {time.time() - start:.3f} seconds")
                
                # get position command:
                q_des = wrap_to_pi(copy.deepcopy(command['position']))
                qd_des = copy.deepcopy(command['velocity']) #* 0.5
                qdd_des = copy.deepcopy(command['acceleration'])
                
                # q_des = q_des % 2*np.pi
                # q_des[q_des >= np.pi] -= 2*np.pi
                # q_des[q_des < -np.pi] += 2*np.pi
                
                # filtered_state_mpc = current_robot_state #mpc_control.current_state
                # curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
                # print("q_des:", q_des)
                # start = time.time()
                move_robot(robot_id, q_des)
                
                # data saving stuff
                curr_time = time.time() - start_time
                curr_q = current_robot_state['position']
                step_distance = np.linalg.norm(angle_diff(curr_q, previous_q))
                cumulative_distance += step_distance
                manipulability, lowest_signed_distance, closest_taxel_id = calculate_taxel_manip_and_dist(robot_id, q_des[:7], obstacles)
                log_data['time'].append(curr_time)
                log_data['joint_position'].append(curr_q.tolist())
                log_data['joint_norm_distance_so_far'].append(cumulative_distance)
                log_data['manipulability'].append(manipulability)
                log_data['distance_to_taxel'].append(lowest_signed_distance)
                log_data['closest_taxel_id'].append(closest_taxel_id)
                previous_q = curr_q

                # print(f"Pybullet robot move step took {time.time() - start:.3f} seconds")
                # Check if close enough to current goal
                
                # if goal_index >= len(x_des_list):
                #     print(f"Goal index {goal_index} is out of bounds for x_des_list with length {len(x_des_list)}")
                #     break

                # Check if MPC has been running too long
                max_mpc_runtime = 420  # Maximum runtime in seconds (7 minutes)
                if curr_time > max_mpc_runtime:
                    print(f"MPC has been running for too long ({curr_time:.2f} seconds). Breaking and moving to next trial.")
                    got_stuck = True
                    break
                
                # log_traj['q'].append(current_robot_state['position'])

                if counter%(10) == 0:
                    print("\n")
                    print(f"Current joint position: {current_robot_state['position']}")
                    print(f"q_des: {q_des}")
                    print(f"goal pos ee: {g_pos}")
                    print(f"error: {ee_error}")
                    print(f"next_waypoint_threshold: {next_waypoint_threshold}")
                    print("\n")
                # joint_err = np.linalg.norm(current_robot_state['position'] - x_des_list[goal_index][:7])
                # joint_err = np.linalg.norm(angle_diff(current_robot_state['position'], x_des_list[goal_index][:7]))
                current_ee = get_endeff(robot_id)[0]
                ee_error = np.linalg.norm(current_ee - g_pos)
                if ee_error < ee_error_threshold:
                    print(f"\nReached waypoint {goal_index} (err: {ee_error:.3f})")
                    goal_index += 1
                    # if goal_index < max_goals:
                    #     mpc_control.update_params(goal_state=x_des_list[goal_index])
                        # print(f"Now trying to reach goal {goal_index} out of {len(x_des_list)-1}: {x_des_list[goal_index]}\n")
                        # if PAUSE: wait_for_user()
                i += 1
                
            except KeyboardInterrupt:
                print('Closing')
                done = True
                break
        
        # Remove all PyBullet debug lines
        p.removeAllUserDebugItems()
        if not got_stuck:
            print(f"Reached goal ee position {g_pos} in {trial_num} trials.")
            # Save data to CSV:
            if save_data: save_log_to_csv(log_data,output_path,trial)
            # if PAUSE: wait_for_user()
        mpc_control.close()
        del mpc_control
    return 1
            
    
if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='gen3', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    parser.add_argument('--g_pos', type=str, default = None, help='goal position xyz')
    parser.add_argument('--g_pos_state', type=str, default = None, help='goal position config example')
    parser.add_argument('--init_config', type=str, default = None, help='initial joint config list of size ndof')
    # parser.add_argument('--js_waypoint_csv_file', type=str, default='', help='Path to the joint space waypoint CSV file')
    parser.add_argument('--output_save_file', type=str, default='', help='Path to the output data CSV file')
    parser.add_argument('--trial_num', type=int, default=1, help='number of trials to run')
    args = parser.parse_args()
    
    weight_list = [1,0.8]
    object_reduction_list = [0.0,0.02]
    min_iterations = 2000
    trial_num = 1
    
    args.sim_type = 6
    args.trial_num = trial_num
    
    if args.sim_type == 3:
        args.g_pos = [0.60, -0.42,  0.41]
        args.init_config = [-0.3,0.3,0.4,1.2,0.5,1.2,0]
        args.g_pos_state = [0.347,0.899,0.231,0.817,0.681,0.91,-1.744]

    elif args.sim_type == 6:
        args.g_pos = [0.45, 0.25, 0.5]
        args.init_config = [0,-0.7,0,1.5,0,1.5,0]
        args.g_pos_state = [0.115,0.339,-0.568,1.724,1.679,-0.658,0.775]
    else:
        raise ValueError(f"Invalid sim_type {args.sim_type}. Only 3 or 6 are supported.")
    
    # args.g_pos_state = [-2.20135674, -1.20621985,  1.33341749,  0.60052824,  1.55639591,  1.47572172,-1.93047825]
    # args.js_waypoint_csv_file = f"/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/data/manip_data/new_reach_over_body/nrob_weight0.0_contactsamplechance0.0_objreduction0.0_Min Iterations4000.csv"
    # args.output_save_file = f"/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/data/manip_data/mpc_mink_reach_over_body/mink_mpc_no_planning_ee.csv"

    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless
    # gym_instance = Gym(**sim_params)

    mpc_robot_interactive(args, None)
    
    # for w in weight_list:
    #     for obj_reduction in object_reduction_list:
    #         # Manually set the file path for the joint space waypoint CSV file
    #         args.js_waypoint_csv_file = f"/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/data/manip_data/new_reach_over_body/nrob_weight{w}_contactsamplechance0.0_objreduction{obj_reduction}_Min Iterations{min_iterations}.csv"
    #         args.output_save_file = f"/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/data/manip_data/mpc_new_reach_over_body/nrob_mpc_weight{w}_contactsamplechance0.0_objreduction{obj_reduction}_Min Iterations{min_iterations}.csv"
    
    #         sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    #         sim_params['headless'] = args.headless
    #         # gym_instance = Gym(**sim_params)
    
    #         mpc_robot_interactive(args, None)
    
