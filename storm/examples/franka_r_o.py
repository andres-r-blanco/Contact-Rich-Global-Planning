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

import open3d as o3d

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

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from multipriority.envs.toy_env_sim import ToyOneEnv

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask
from storm_kit.mpc.task.contact_task import ContactTask

import numpy as np
import rospy
import time

from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

LOGGER_RUNNING = False


# from multipriority.envs.toy_env_ros import RealToyEnv
# from multipriority.utils import Tracker, load_yaml, simulate_feedback, simulate_simple_feedback, load_trajectory_jointspace, simulate_last_feedback
# from multipriority.controllers import TaskController, ContactController, MultiPriorityController
# from multipriority.bandit import PriorityLinUCB
# from collections import deque

def unity2storm(position=None, orientation=None):
    # transformation from Storm to Unity is : then mirror img about xy axis and then rotate -90 deg about x
    # (vice versa also works somehow) (must mean mirror img and axis rotation are commutative for this case)
    # TODO Pranav : verify this
    unity2storm_rot = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
    
    x_neg90_rot_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    x_neg90_rot_quat = Quaternion(axis=[1, 1, 1], angle=0) #same as above matrix but in quaternion form, convention : wxyzquaternion form, convention : wxyz

    if position==None and not orientation==None:
        
        # Orientation input has to be in Euler angles form, output will be quaternion
        
        # Rotations are intrinsic since they are around body axes , so capital letters. 
        # Source : https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html

        # Rotation convention taken from here:
        # https://docs.unity3d.com/ScriptReference/Transform-eulerAngles.html 

        orientation[2] = -1 * orientation[2]
        
        # print(R.from_euler('ZXY', orientation, degrees=True).as_matrix())
        q = Quaternion(matrix=R.from_euler('ZXY', orientation, degrees=True).as_matrix())

        # pyquaternion works xyzw form. So no need for reindexing
        g_q = x_neg90_rot_quat * q * x_neg90_rot_quat.inverse

        return np.ravel([g_q[0], g_q[1], g_q[2], g_q[3]])

    elif not position==None and orientation==None:
    
        # For position vector, the mirror image and rotation can be condensed into one matrix.
        g_pos = np.matmul(unity2storm_rot,position)
        return g_pos

    else:
        print("Both/Neither position & orientation entered, check again!")



    



np.set_printoptions(precision=2)

def mpc_robot_interactive(args, gym_instance):
    
    # Logger setup for metrics
    
    
    #
    vis_ee_target = True
    robot_file = args.robot + '.yml'
    task_file = args.robot + '_reacher_new.yml'
    world_file = 'collision_primitives_3d.yml'

    world_yml = join_path(get_gym_configs_path(), world_file)
    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)

    robot_yml = join_path(get_gym_configs_path(),args.robot + '.yml')
    with open(robot_yml) as file:
        robot_params = yaml.load(file, Loader=yaml.FullLoader)
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    if(args.cuda):
        device = 'cuda'
    else:
        device = 'cpu'
        
    init_config = [0.21, 293.64, 182, 230.15, 359.25, 332.14, 90.87]
    # convert to -180 to 180
    for i in range(len(init_config)):
        if init_config[i] > 180:
            init_config[i] = init_config[i] - 360
    
    sim_params['init_state'] = np.radians(init_config)

    # mesh = o3d.io.read_triangle_mesh('/home/mahika/storm_gen3/cylinder_env.obj')
    # mesh_vertices = np.asarray(mesh.vertices).astype(np.float32)
    # mesh_triangles = np.asarray(mesh.triangles).astype(np.uint32)

    # tm_params = gymapi.TriangleMeshParams()
    # tm_params.nb_vertices = mesh_vertices.shape[0]
    # tm_params.nb_triangles = mesh_triangles.shape[0]
    # gym.add_triangle_mesh(sim, mesh_vertices.flatten(order='C'),
    #                             mesh_triangles.flatten(order='C'),
    #                             tm_params)
    
    
    # asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    # asset_options.armature = 0.01

    # asset = gym.load_asset(sim, "", asset_file, asset_options)


    sim_params['collision_model'] = None
    kwargs = {'taxel_cfg': 'taxel_data.yaml'}
    unity_robot_sim = ToyOneEnv(**kwargs)
    
    # create robot simulation:
    # robot_sim = RobotSim(gym_instance=gym, sim_instance=sim, **sim_params, device=device)

    
    # create gym environment:
    robot_pose = sim_params['robot_pose']
    # env_ptr = gym_instance.env_list[0]
    # robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

    device = torch.device('cuda', 0)

    
    tensor_args = {'device':device, 'dtype':torch.float32}
    

    # spawn camera:
    # robot_camera_pose = np.array([1.6,-1.5, 1.8,0.707,0.0,0.0,0.707])
    # q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
    # robot_camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])

    
    # robot_sim.spawn_camera(env_ptr, 60, 640, 480, robot_camera_pose)

    # # get pose
    # w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)
    
    # w_T_robot = torch.eye(4)
    # quat = torch.tensor([w_T_r.r.w,w_T_r.r.x,w_T_r.r.y,w_T_r.r.z]).unsqueeze(0)
    # rot = quaternion_to_matrix(quat)
    # w_T_robot[0,3] = w_T_r.p.x
    # w_T_robot[1,3] = w_T_r.p.y
    # w_T_robot[2,3] = w_T_r.p.z
    # w_T_robot[:3,:3] = rot[0]

    # world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)
    

    
    table_dims = np.ravel([1.5,2.5,0.7])
    cube_pose = np.ravel([0.35, -0.0,-0.35,0.0, 0.0, 0.0,1.0])
    cube_pose = np.ravel([0.9,0.3,0.4, 0.0, 0.0, 0.0,1.0])
    
    table_dims = np.ravel([0.35,0.1,0.8])

    
    
    cube_pose = np.ravel([0.35,0.3,0.4, 0.0, 0.0, 0.0,1.0])
    
    # table_dims = np.ravel([0.3,0.1,0.8])
    

    # get camera data:
    mpc_control = ContactTask(task_file, robot_file, world_file, tensor_args)

    # n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs

    
    # start_qdd = torch.zeros(n_dof, **tensor_args)

    # update goal:

    # exp_params = mpc_control.exp_params
    
    # current_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
    # ee_list = []
    

    mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}

    for _ in range(100):
        unity_robot_sim.robot.setJointPositionsDirectly(init_config)
        obs = unity_robot_sim.get_obs()
        # TODO: get RGB data and visualize
        # unity_robot_sim.camera.initializeRGB()

        # # Get RGB camera data
        # rgb_img = unity_robot_sim.camera.getRGB()
        # rgb_img = np.array(rgb_img)
        pb_pos = list(obs['robot']['positions'][8])
        unity_pos = unity_robot_sim.robot.ik_controller.get_unity_pos_from_bullet(pb_pos)
        unity_robot_sim.viz_cube.setTransform(position=unity_pos)
        unity_robot_sim._step()
    
    zrs = np.zeros(7, dtype=np.float32)
    gen3_bl_state = np.concatenate((np.radians(init_config), zrs))
    # franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4,0.0,
    #                             0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    x_des_list = [gen3_bl_state]
    
    ee_error = 10.0
    j = 0
    t_step = 0
    i = 0
    x_des = x_des_list[0]
    
    mpc_control.update_params(goal_state=x_des)

    # spawn object:
    x,y,z = 0.0, 0.0, 0.0
    tray_color = gymapi.Vec3(0.8, 0.1, 0.1)
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002


    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(x, y, z)
    object_pose.r = gymapi.Quat(0,0,0, 1)
    
    ex,ey,ez = 0.0, 1.0, 1.0    
    env_pose = gymapi.Transform()
    env_pose.p = gymapi.Vec3(ex,ey,ez)
    env_pose.r = gymapi.Quat(0,0,0, 1)
    
    obj_asset_file = "urdf/mug/movable_mug.urdf" 
    obj_asset_root = get_assets_path()
    
    # if(vis_ee_target):
    #     target_object = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_target_object')
    #     obj_base_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 0)
    #     obj_body_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 6)
    #     gym.set_rigid_body_color(env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
    #     gym.set_rigid_body_color(env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)


    #     obj_asset_file = "urdf/mug/mug.urdf"
    #     obj_asset_root = get_assets_path()


    #     ee_handle = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_current_as_mug')
    #     ee_body_handle = gym.get_actor_rigid_body_handle(env_ptr, ee_handle, 0)
    #     tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
    #     gym.set_rigid_body_color(env_ptr, ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
    
    # obj_asset_file = "urdf/env.urdf"
    obj_asset_root = get_assets_path()
    # cylinder_env = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='cylinder_env')
    # cylinder_env_handle = gym.get_actor_rigid_body_handle(env_ptr, cylinder_env, 0)
    # cylinder_body_handle = gym.get_actor_rigid_body_handle(env_ptr, cylinder_env, 6)
    tray_color = gymapi.Vec3(0.8, 0.8, 0.0)
    # gym.set_rigid_body_color(env_ptr, cylinder_env, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
    # gym.set_rigid_body_color(env_ptr, cylinder_env, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
    # gym.set_rigid_transform(env_ptr, cylinder_env_handle, env_pose)
    
    ee_pose = gymapi.Transform()
    # w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3,3].unsqueeze(0),
    #                                     rot=w_T_robot[0:3,0:3].unsqueeze(0))

    rollout = mpc_control.controller.rollout_fn
    tensor_args = mpc_tensor_dtype
    sim_dt = mpc_control.exp_params['control_dt']
    
    log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[],
                'qddd_des':[]}

    q_des = None
    qd_des = None
    t_step = 0 #gym_instance.get_sim_time()

    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    old_joint_pos = np.array(unity_robot_sim.robot.current_joint_positions)
    current_robot_state = {'position':[], 'velocity':[], 'acceleration':np.zeros(7, dtype=np.float32)}
    while(i > -100):
        try:
            # gym_instance.step()
            unity_robot_sim._step()
            obs = unity_robot_sim.get_obs()
            force_threshold = 1.0
            contact_info = {}
            # populate contact info for taxels with forces > force_threshold
            skin_data = obs['skin']['forces']
            # for i in range(len(skin_data)):
            #     if skin_data[i] > force_threshold:
            #         if contact_info.keys() == []:
            #             contact_info = {'force':[], 'normal':[], 'jac':[], 'link':[], 'pos':[]}
            #         taxel = unity_robot_sim.robot.taxels[i]
            #         contact_info['force'].append(skin_data[i])
            #         contactF, normalVec, localPos, linkID = taxel.get_contact_info()
            #         contact_info['link'].append(linkID)
            #         contact_info['pos'].append(localPos)
            #         currentJointPos = unity_robot_sim.robot.current_joint_positions
            #         currentJointVel = unity_robot_sim.robot.current_joint_velocities
            #         # currPBTrans, currPBOri = unity_robot_sim.robot.link_pos, unity_robot_sim.robot.link_ori
            #         # world_to_link_tr = currPBOri[linkID]
            #         # # world_to_link_tr = np.linalg.inv(world_to_link_tr)
            #         # normalVec_ = np.dot(world_to_link_tr, normalVec)
            #         contact_info['normal'].append(normalVec)
                    
            #         Jt, Jr = unity_robot_sim.robot.ik_controller.ros_calc_contact_jacobian(linkID, localPos, currentJointPos, currentJointVel, [0, 0, 0, 0, 0, 0, 0])
                    
            #         contact_info['jac'].append(Jt)
                    # normalVec_ = normalVec_ / np.linalg.norm(normalVec_)
            if 'force' in contact_info.keys():
                print ("Forces: ", contact_info['force'])
            
            if(vis_ee_target):
                cube_pose = unity_robot_sim.viz_cube.getPosition()
                goal_pose = unity2storm(position=cube_pose)
                g_pos = goal_pose
                g_q = unity2storm(orientation=unity_robot_sim.viz_cube.getRotation())
                mpc_control.update_params(goal_ee_pos=g_pos,
                                            goal_ee_quat=g_q)
            t_step += sim_dt
            
            # TODO: Get skin data, including force, normal, and contact jacobian
            
            # current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
            # print ("Current robot state dict: ", current_robot_state.keys())
            # current_robot_state['position'] = np.array(current_robot_state['position'])
            # current_robot_state['velocity'] = np.array(current_robot_state['velocity'])
            current_robot_state['position'] = np.array(unity_robot_sim.robot.current_joint_positions)# np.array(current_robot_state['position'])
            current_robot_state['velocity'] = np.array(unity_robot_sim.robot.current_joint_velocities)# np.array(current_robot_state['velocity'])
            # print ("Current state: ", current_robot_state['position'])
            
            # print joint id and joint angle with a jump higher than 0.1
            # for i in range(len(current_robot_state['position'])):
            #     if abs(current_robot_state['position'][i] - old_joint_pos[i]) > 0.1:
            #         print ("Joint: ", i, " angle: ", current_robot_state['position'][i])
            
            # old_joint_pos = current_robot_state['position']
            
            command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, contact_info=contact_info, WAIT=False)

            filtered_state_mpc = current_robot_state #mpc_control.current_state
            curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
            # get position command:
            q_des = copy.deepcopy(command['position'])
            qd_des = copy.deepcopy(command['velocity']) #* 0.5
            qdd_des = copy.deepcopy(command['acceleration'])
            
            # ee_error = mpc_control.get_current_error(filtered_state_mpc)
             
            # pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
            
            # get current pose:
            # e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
            # e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
            # ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
            # ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
            
            # ee_pose = copy.deepcopy(w_T_r) * copy.deepcopy(ee_pose)
            

            # print(["{:.3f}".format(x) for x in ee_error], "{:.3f}".format(mpc_control.opt_dt),
            #       "{:.3f}".format(mpc_control.mpc_dt))
        
            
            # gym_instance.clear_lines()
            # top_trajs = mpc_control.top_trajs.cpu().float()#.numpy()
            # n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
            # w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)


            # top_trajs = w_pts.cpu().numpy()
            # color = np.array([0.0, 1.0, 0.0])
            # for k in range(top_trajs.shape[0]):
            #     pts = top_trajs[k,:,:]
            #     color[0] = float(k) / float(top_trajs.shape[0])
            #     color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
            #     gym_instance.draw_lines(pts, color=color)
    
            # M_q = unity_robot_sim.robot.ik_controller.ros_calc_Mq(q_des.tolist())
            # # print ("Qdd des: ", qdd_des)
            # # tau_cmd = M_q @ qdd_des
            # # print ("Tau cmd: ", tau_cmd)
            # # tau_cmd = 100.0 * np.array(tau_cmd).reshape(-1, 1)
            # # increase qdd_des magnitude by 100x
            # qdd_des = 1.0 * qdd_des
            # # print ("Qdd new des: ", qdd_des)
            
            # gq = unity_robot_sim.robot.ik_controller.calc_gq(unity_robot_sim.robot.current_joint_positions, unity_robot_sim.robot.current_joint_velocities, qdd_des.tolist())
            # gq_storm = mpc_control.controller.rollout_fn.dynamics_model.robot_model.compute_inverse_dynamics(torch.Tensor(unity_robot_sim.robot.current_joint_positions).unsqueeze(0), torch.Tensor(unity_robot_sim.robot.current_joint_velocities).unsqueeze(0), torch.Tensor(qdd_des).unsqueeze(0), include_gravity=True)
            # gq = np.array(gq).reshape(-1, 1)
            # # print ("Gq: ", gq)
            # # print ("Gq storm: ", gq_storm)
            # gq_storm = gq_storm.cpu().numpy().reshape(-1, 1)
            # tau = tau_cmd + gq
            # robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)
            # robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
            # robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
            # robot_sim.command_robot_position(unity_robot_sim.robot.current_joint_positions, env_ptr, robot_ptr)
            # q_des = np.degrees(q_des)
            # q_des = q_des % 360.0
            # q_des[q_des >= 180.0] = q_des[q_des > 180.0] - 360
            # q_des[q_des < -180.0] = q_des[q_des < -180.0] + 360
            # print ("unity cmd: ", q_des)

            
            # gq = unity_robot_sim.robot.ik_controller.calc_gq(unity_robot_sim.robot.current_joint_positions, unity_robot_sim.robot.current_joint_velocities, [0, 0, 0, 0, 0, 0, 0])
            # print ("Tau cmd: ", tau_cmd.shape)
            # print ("Gq: ", gq.shape)
            # tau = tau_cmd + gq
            # unity_robot_sim.step(gq_storm)
            
            unity_robot_sim.robot.setJointVelocity(np.degrees(qd_des))
            
            # obs = unity_robot_sim.get_obs()
            
            # robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
            # current_state = command
            
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
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless
    # gym_instance = Gym(**sim_params)
    
    
    mpc_robot_interactive(args, None)
    
