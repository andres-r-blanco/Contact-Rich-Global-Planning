import os
import multipriority
from multipriority.envs.toy_env_sim import ToyOneEnv
import pygame
import numpy as np
import yaml
import pathlib

from multipriority.controllers import MultiPriorityController, TaskController, ContactController
from multipriority.utils import *
from scipy.spatial.transform import Rotation as R

import json

FORCE_THRESHOLD = -0.1

env = ToyOneEnv()
pygame.init()
display = pygame.display.set_mode((300, 300))

# Environment setup
##############################################
init_pos = [90.21, 293.64, 182, 230.15, 359.25, 332.14, 90.87]
# init_pos = [0, 0, 0, 0, 0, 0, 0]
# init_pos = [0, 30, 12, 22, 11, 0, 0]
for i in range(len(init_pos)):
    if init_pos[i] > 180:
        init_pos[i] -= 360
env.robot.setJointPositionsDirectly(init_pos)
env._step()
obs = {}
obs['robot'] = env.robot.getRobotState()
obs['skin'] = env.skin.getInfo()

config_name = 'taxel_data.yaml'
taxel_dict = load_taxel_dict(config_name, obs)
skeleton_config_name = 'test_cube_params.yaml'
skeleton_contact_params = load_yaml(skeleton_config_name)
skeleton_priority = {"b0":1, "b1":2, "b2":3, "b3":4}
##############################################

task_controller = TaskController(env.robot)
task_controller.set_goal_pose(np.array([0.62, 0.5, 0.1]), np.array([90,0,0]))
task_controller.set_controller_gains(0, 0.0)
master_controller = MultiPriorityController(skeleton_contact_params, skeleton_priority, env.robot, task_controller)

taxel_objects = {}
for taxel_id in taxel_dict:
    taxel_objects[taxel_id] = env.create_object(id=(int(taxel_id)+1000), name=f"Taxel{taxel_id}", is_in_scene=True)

# Save taxel data to yaml
taxel_data = {}
# entries = ['ID', 'Link', 'Position', 'Normal']
for i in range(100000):
    obs['robot'] = env.robot.getRobotState()
    gq = env.robot.ik_controller.calc_gq(obs['robot']['joint_positions'], obs['robot']['joint_velocities'], [0, 0, 0, 0, 0, 0, 0])
    
    action = env.boost_torque(gq)
    env.set_joint_sim(action)
    env._step()

    obs['robot'] = env.robot.getRobotState()
    obs['robot']['world_to_link_trans_pb'] = env.robot.getPBWorldToLinkTransforms(obs['robot']['joint_positions'])
    obs['skin'] = env.skin.getInfo()
    
    forces, active_taxel_ids, positions = filter_active_taxels(obs['skin']['forces'], obs['skin']['ids'], obs['skin']['positions'], FORCE_THRESHOLD)

    for taxel_id, force, taxel_pos in zip(active_taxel_ids, forces, positions):
        taxel_dict[taxel_id].update(force, taxel_pos)
        point_local = taxel_dict[taxel_id].calculate_local_position(obs['robot']['world_to_link_trans_pb'][taxel_dict[taxel_id].link_id])
        pt = env.robot.ik_controller.get_bullet_pos_from_unity(taxel_dict[taxel_id].pos)
        axis_scale = 0.1
        rot_mat =  obs['robot']['world_to_link_trans_pb'][taxel_dict[taxel_id].link_id][:3, :3]

        ang = taxel_objects[taxel_id].getQuaternion()
        taxel_rot = R.from_quat(ang).as_matrix()
        transformed_matrix = taxel_rot.copy()
        transformed_matrix[[0, 1, 2]] = taxel_rot[[2, 0, 1]]
        transformed_matrix[1, :] *= -1
        taxel_rot = transformed_matrix
        taxel_rot = np.dot(np.linalg.inv(taxel_rot), rot_mat)
        # force vector is opposite to x-axis
        neg_x = taxel_rot[2, 0:3] * -1
        pt_to_x = point_local + axis_scale * neg_x
        env.robot.ik_controller.bullet_client.addUserDebugLine(lineFromXYZ = point_local,
                                                                lineToXYZ = [pt_to_x[0], pt_to_x[1], pt_to_x[2]],
                                                                lineColorRGB = [1, 0, 0] ,
                                                                lineWidth = 5,
                                                                lifeTime = 0,
                                                                parentObjectUniqueId = env.robot.ik_controller.robot,
                                                                parentLinkIndex=taxel_dict[taxel_id].link_id)
        if i > 3:
            taxel_data[int(taxel_id)] = {}
            taxel_data[int(taxel_id)]['Link'] = int(taxel_dict[taxel_id].link_id)
            taxel_data[int(taxel_id)]['Position'] = point_local.tolist()
            taxel_data[int(taxel_id)]['Normal'] = (neg_x / np.linalg.norm(neg_x)).tolist()
    
    if i > 3:
        # to make sure there are no misreads because of position update delays
        # save dictionary of dictionaries to yaml
        with open('taxel_data.yaml', 'w') as file:
            yaml.dump(taxel_data, file)
        break

    prev_obs = obs