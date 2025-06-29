import os
import multipriority
import pygame
import numpy as np
import yaml
import pathlib

from multipriority.utils import *
from scipy.spatial.transform import Rotation as R

import rospy
from multipriority.controllers import MultiPriorityController, TaskController, ContactController
from multipriority.utils import *
from multipriority.agents.osc_controller import PBAgent
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation
from pyrr import quaternion

from controller_manager_msgs.srv import SwitchController
from moveit_msgs.msg import CartesianTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import json

FORCE_THRESHOLD = -0.1
urdf_path='/home/mahika/mp_new/multipriority/urdfs/gen3_7dof_vision_with_skin.urdf'
base_pose = np.array([0, 0, 0])
base_orientation = [0, 0.0, 0.0, 1]
# pb_robot = KinovaOSCController(
#     robot_urdf=urdf_path,
#     base_pos=base_pose,
#     base_orn=base_orientation,
#     render=True
# )

CONTROL_MODE='velocity' # position, torque

taxel_data = load_yaml('taxel_real_robot.yaml')
# Environment setup
##############################################
init_pos = [0] * 7
obs = {}

robot = PBAgent(
    robot_urdf=urdf_path,
    base_pos=base_pose,
    base_orn=base_orientation,
    render=True
)
robot.set_joint_positions(init_pos)
link_tr_pb, link_ori_pb, local_pos, local_ori = robot.ros_get_link_states_pybullet(init_pos)
world_to_link_trs = []
for i in range(len(link_tr_pb)):
    pb_world_to_link_tr = np.eye(4)
    pb_world_to_link_tr[:3, :3] = np.asarray(link_ori_pb[i]).reshape(3,3)
    pb_world_to_link_tr[:3, 3] = link_tr_pb[i]
    world_to_link_trs.append(pb_world_to_link_tr)

pb_taxel_data = {}
for i in range(1000):
    for taxel_id in taxel_data:
        taxel = taxel_data[taxel_id]
        taxel_id = 1000 + (1027 - int(taxel_id))
        link_id = taxel['Link']
        print ("Link ID: ", link_id, " Old Link ID: ", taxel['Link'])
        pos = taxel['Position'] # pos in joint
        # shift position to be in the link frame
        pos = (np.array(pos) - np.array(local_pos[link_id])).tolist()
        # homogenous transformation matrix for going from joint frame to link frame
        tr = np.eye(4)
        tr[:3, :3] = np.asarray(local_ori[link_id]).reshape(3,3)
        tr[:3, 3] = local_pos[link_id]
        # pos = np.dot(np.linalg.inv, np.array(pos + [1]))[:3]
        joint_frame_ori = R.from_quat(taxel['Orientation']).as_matrix()
        taxel_rot = np.dot(np.linalg.inv(joint_frame_ori), local_ori[link_id][:3, :3])
        
        
        # neg_x = taxel_rot[1, 0:3] * -1
        # pt_to_x = pos + 0.1 * neg_x
        # robot.bullet_client.addUserDebugLine(lineFromXYZ = pos,
        #                                     lineToXYZ = [pt_to_x[0], pt_to_x[1], pt_to_x[2]],
        #                                     lineColorRGB = [1, 0, 0] ,
        #                                     lineWidth = 5,
        #                                     lifeTime = 0,
        #                                     parentObjectUniqueId = robot.robot,
        #                                     parentLinkIndex=link_id)
        # visualize x y z axes using RGB colors
        # x
        pt_to_x = pos + 0.1 * taxel_rot[0, 0:3]
        if taxel_id == 1012:
            print ("Taxel Rot: ", taxel_rot)
            print ("Pos: ", pos)
            print ("Pt to x: ", pt_to_x)
            robot.bullet_client.addUserDebugLine(lineFromXYZ = pos,
                                                lineToXYZ = [pt_to_x[0], pt_to_x[1], pt_to_x[2]],
                                                lineColorRGB = [1, 0, 0] ,
                                                lineWidth = 5,
                                                lifeTime = 0,
                                                parentObjectUniqueId = robot.robot,
                                                parentLinkIndex=link_id)
        # pt_to_y = pos + 0.1 * taxel_rot[1, 0:3]
        # robot.bullet_client.addUserDebugLine(lineFromXYZ = pos,
        #                                     lineToXYZ = [pt_to_y[0], pt_to_y[1], pt_to_y[2]],
        #                                     lineColorRGB = [0, 1, 0] ,
        #                                     lineWidth = 5,
        #                                     lifeTime = 0,
        #                                     parentObjectUniqueId = robot.robot,
        #                                     parentLinkIndex=link_id)
        # pt_to_z = pos + 0.1 * taxel_rot[2, 0:3]
        # robot.bullet_client.addUserDebugLine(lineFromXYZ = pos,
        #                                     lineToXYZ = [pt_to_z[0], pt_to_z[1], pt_to_z[2]],
        #                                     lineColorRGB = [0, 0, 1] ,
        #                                     lineWidth = 5,
        #                                     lifeTime = 0,
        #                                     parentObjectUniqueId = robot.robot,
        #                                     parentLinkIndex=link_id)
        
    # time.sleep(1)
    # for taxel_id, force, taxel_pos in zip(active_taxel_ids, forces, positions):
    #     taxel_dict[taxel_id].update(force, taxel_pos)
    #     point_local = taxel_dict[taxel_id].calculate_local_position(obs['robot']['world_to_link_trans_pb'][taxel_dict[taxel_id].link_id])
    #     pt = env.robot.ik_controller.get_bullet_pos_from_unity(taxel_dict[taxel_id].pos)
    #     axis_scale = 0.1
    #     rot_mat =  obs['robot']['world_to_link_trans_pb'][taxel_dict[taxel_id].link_id][:3, :3]

    #     ang = taxel_objects[taxel_id].getQuaternion()
    #     taxel_rot = R.from_quat(ang).as_matrix()
    #     transformed_matrix = taxel_rot.copy()
    #     transformed_matrix[[0, 1, 2]] = taxel_rot[[2, 0, 1]]
    #     transformed_matrix[1, :] *= -1
    #     taxel_rot = transformed_matrix
    #     taxel_rot = np.dot(np.linalg.inv(taxel_rot), rot_mat)
    #     # force vector is opposite to x-axis
    #     neg_x = taxel_rot[2, 0:3] * -1
    #     pt_to_x = point_local + axis_scale * neg_x
    #     env.robot.ik_controller.bullet_client.addUserDebugLine(lineFromXYZ = point_local,
    #                                                             lineToXYZ = [pt_to_x[0], pt_to_x[1], pt_to_x[2]],
    #                                                             lineColorRGB = [1, 0, 0] ,
    #                                                             lineWidth = 5,
    #                                                             lifeTime = 0,
    #                                                             parentObjectUniqueId = env.robot.ik_controller.robot,
    #                                                             parentLinkIndex=taxel_dict[taxel_id].link_id)
        if i > 3:
            pb_taxel_data[int(taxel_id)] = {}
            pb_taxel_data[int(taxel_id)]['Link'] = link_id
            print ("Stored Link ID: ", link_id)
            pb_taxel_data[int(taxel_id)]['Position'] = pos
            pb_taxel_data[int(taxel_id)]['Normal'] = (taxel_rot[0, 0:3] / np.linalg.norm(taxel_rot[0, 0:3])).tolist()
    
    # if i > 3:
    #     # to make sure there are no misreads because of position update delays
    #     # save dictionary of dictionaries to yaml
    #     with open('real_taxel_data.yaml', 'w') as file:
    #         yaml.dump(pb_taxel_data, file)
    #     break
    # if i > 3:
    #     # to make sure there are no misreads because of position update delays
    #     # save dictionary of dictionaries to yaml
    #     with open('taxel_data.yaml', 'w') as file:
    #         yaml.dump(taxel_data, file)
    #     break

    prev_obs = obs