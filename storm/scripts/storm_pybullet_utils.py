# utils.py - shared helper functions for both waypoint-based and goal-based MPC scripts

import numpy as np
import math
import pybullet as p
import csv

import os
import sys
import time

ROOT_DIR = r"/home/rishabh/Andres/"

# Add necessary folders to the Python path
sys.path.insert(0, os.path.join(ROOT_DIR, 'storm'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'storm', 'scripts'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'Manip_planning', 'mp-osc','multipriority'))  # adjust if needed
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'Manip_planning', 'mp-osc','pybullet_planning_master'))
sys.path.insert(1, os.path.join(ROOT_DIR, 'Manip_planning', 'mp-osc','multipriority', 'urdfs'))
from pybullet_tools.utils import wait_for_user, set_camera_pose
from contact_manip_rrt import calculate_manip_cost


JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6]
DISTANCE_THRESHOLD = 0.2

# ----------------------
# Core utilities
# ----------------------

def wrap_to_pi(q):
    return (q + np.pi) % (2 * np.pi) - np.pi

def angle_diff(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi

def interpolate_configs(config1, config2, num_steps=10):
    return [
        [(1 - t) * c1 + t * c2 for c1, c2 in zip(config1, config2)]
        for t in (i / float(num_steps + 1) for i in range(1, num_steps + 1))
    ]

def set_joint_positions(robot_id, q, joint_indices=JOINT_INDICES):
    for idx, joint_index in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_index, q[idx])
    p.stepSimulation()

def move_robot(robot_id, q, joint_indices=JOINT_INDICES):
    for idx, joint_index in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_index, q[idx])
    p.stepSimulation()

def set_robot_state(robot_id, q,joint_indices=JOINT_INDICES):
    for joint_index in joint_indices:
        p.resetJointState(robot_id, joint_index, q[joint_index])
    p.stepSimulation()

def q_to_endeff(robot_id, q):
    set_joint_positions(robot_id, q)
    pos, ori = p.getLinkState(robot_id, 7)[4:6]
    return np.array(pos), np.array(ori)

def get_endeff(robot_id):
    pos, ori = p.getLinkState(robot_id, 7)[4:6]
    return np.array(pos), np.array(ori)

# ----------------------
# Collision and manipulability
# ----------------------

def correct_J_form(robot_id, J):
    J_out = np.zeros((3, 7))
    i = 0
    for c in range(7):
        if p.getJointInfo(robot_id, c)[3] > -1:
            for r in range(3):
                J_out[r, i] += J[r][i]
            i += 1
    return J_out

def compute_signed_distance_for_link(robot_id, link_index, obstacle_id, distance_threshold=DISTANCE_THRESHOLD):
    distance_threshold = 0.1
    closest_points = p.getClosestPoints(bodyA=robot_id, bodyB=obstacle_id, linkIndexA=link_index, distance=distance_threshold)
    if closest_points:
        closest_point = min(closest_points, key=lambda point: point[8])
        return closest_point[8], closest_point[5], closest_point[6]
    return None, None, None

def calculate_taxel_manip_and_dist(robot_id,q,obstacles):
    lowest_signed_distances = []
    closest_taxel_ids = []
    manipulabilities = []
    ends = []
    starts = []
    signed_dist = None
    start1 = 0
    end1 = 0
    closest_taxel_id = None
    for obstacle_id in obstacles:
        for link_index in range(9,p.getNumJoints(robot_id)):
            signed_dist,start,end = compute_signed_distance_for_link(robot_id, link_index, obstacle_id, distance_threshold=DISTANCE_THRESHOLD)
            if signed_dist is not None and signed_dist < DISTANCE_THRESHOLD:
                lowest_signed_distances.append(signed_dist)
                closest_taxel_ids.append(link_index)
                ends.append(end)
                starts.append(start)
    if not lowest_signed_distances: 
        return [1], [10000], [-1]
    
    for i, closest_taxel_id  in enumerate(closest_taxel_ids):

    
        J, _ = p.calculateJacobian(robot_id, closest_taxel_id, [0, 0, 0], q.tolist(), [0]*7, [0]*7)
        J = correct_J_form(robot_id, J)
        JJT = np.dot(J, np.transpose(J))
        det_JJT = np.linalg.det(JJT)
        det_JJT = max(det_JJT, 0)
        manipulability = np.sqrt(det_JJT)
        manipulabilities.append(manipulability)
        
    return manipulabilities, lowest_signed_distances, closest_taxel_ids

# def calculate_taxel_manip_and_dist(robot_id, q, obstacles):
#     closest_dist = 10000
#     closest_link = -1
#     for obs_id in obstacles:
#         for link_idx in range(9, p.getNumJoints(robot_id)):
#             dist, _, _ = compute_signed_distance_for_link(robot_id, link_idx, obs_id)
#             if dist is not None and dist < closest_dist:
#                 closest_dist = dist
#                 closest_link = link_idx

#     if closest_dist > DISTANCE_THRESHOLD:
#         return 1.0, 10000.0, -1

#     J, _ = p.calculateJacobian(robot_id, closest_link, [0, 0, 0], q.tolist(), [0]*7, [0]*7)
#     J = correct_J_form(robot_id, J)
#     JJT = np.dot(J, J.T)
#     det = np.linalg.det(JJT)
#     return float(np.sqrt(max(det, 0))), closest_dist, closest_link

# ----------------------
# Waypoint utilities
# ----------------------

def get_js_waypoint_list(file_path, row_num, rows_per_trial=12):
    """Takes in a csv file path and a row number, 
    returns a list of joint space waypoints where each list contains joint values for all joints at a specific state."""
    print(row_num)
    js_waypoint_list = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if row_num + 1 < len(rows) and rows[row_num + 1][0].startswith("Trial"):
            print(f"this rrt trial had failed moving on to next one")
            return None
        else:
            for i in range(rows_per_trial-7, rows_per_trial):
                if row_num + i >= len(rows):
                    print(f"Row {row_num + i} does not exist in the file.")
                    return None
                js_row = rows[row_num + i]
                js_row = list(map(float, js_row[1:]))
                # js_row.pop()  # Remove the last value if not needed
                js_waypoint_list.append(js_row)
                
    

    # Transpose the list to get n lists of 7 values
    transposed_list = list(map(list, zip(*js_waypoint_list)))
    transposed_list = [list(map(wrap_to_pi, state)) for state in transposed_list]
    # print(f"First value in transposed list: {transposed_list[0]}")
    # input()
    return transposed_list

def vis_waypoints(js_waypoint_list, robot_id, interpolate = True, wait = False, res = 0.05, line_color=[1, 0, 0]):
    """Takes in a list of joint space waypoints, 
    visualizes each of them via pybullet reset (quickly interpolating if indicated)."""
    for i, q in enumerate(js_waypoint_list):

        if i > 0:
            prev_q = js_waypoint_list[i - 1]

            if interpolate:
                #interpolate and to visualize moving from waypoint to waypoint
                num_steps = math.ceil(5*np.max(np.abs(np.array(q)[:]-np.array(prev_q)[:]))/res) 
                q_list = interpolate_configs(prev_q, q, num_steps = num_steps)
                q_list.append(q)
                for q_interp in q_list:
                    time.sleep(0.5/num_steps)
                    set_robot_state(robot_id, q_interp)
                    p.stepSimulation()

            prev_pos = q_to_endeff(robot_id, prev_q)[0]
            curr_pos = q_to_endeff(robot_id, q)[0]
            # Draw a line between the two configurations
            if isinstance(prev_pos, np.ndarray): prev_pos = prev_pos.tolist()
            if isinstance(curr_pos, np.ndarray): curr_pos = curr_pos.tolist()
            line_id = p.addUserDebugLine(prev_pos, curr_pos, lineColorRGB=line_color, lineWidth=1.5)
            time.sleep(0.2)
            if wait: wait_for_user()
            
        print(f"Waypoint {i}: {q}")
        set_robot_state(robot_id, q)
        # p.setJointMotorControl2(self.robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=joint_angle)
        p.stepSimulation()
        
def draw_waypoints(js_waypoint_list, robot_id, line_color=[1, 0, 0]):
    """Draws red lines between waypoints as fast as possible."""
    for i in range(1, len(js_waypoint_list)):
        prev_pos = q_to_endeff(robot_id, js_waypoint_list[i - 1])[0]
        curr_pos = q_to_endeff(robot_id, js_waypoint_list[i])[0]
        # Draw a line between the two configurations
        if isinstance(prev_pos, np.ndarray): prev_pos = prev_pos.tolist()
        if isinstance(curr_pos, np.ndarray): curr_pos = curr_pos.tolist()
        p.addUserDebugLine(prev_pos, curr_pos, lineColorRGB=line_color, lineWidth=1.5)

# ----------------------
# Save data utilities
# ----------------------
def save_log_to_csv(log_data,output_path,trial):
    with open(output_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Trial', str(trial)])
        writer.writerow(['Time (s)'] + [str(t) for t in log_data['time']])
        writer.writerow(['Joint Norm Distance'] + [str(dist) for dist in log_data['joint_norm_distance_so_far']])
        # writer.writerow(['Manipulability'] + [str(manip) for manip in log_data['manipulability']])
        manip_cost_str = ['Local Manip Cost']
        manip_str = ['Closest Taxel Manip']
        closest_taxel_str = ['Closest Taxel ID']
        dist_closest_taxel_str = ['Distance to Taxel']
        for i, manip in enumerate(log_data['manipulabilities']):
            closest_idx = np.argmin(log_data['closest_taxel_ids'][i])
            manip_cost,_ = calculate_manip_cost(log_data['distance_to_taxels'][i], log_data['manipulabilities'][i])
            manip_cost_str.append(str(manip_cost))
            manip_str.append(str(log_data['manipulabilities'][i][closest_idx]))
            if log_data['closest_taxel_ids'][i][closest_idx] is not None:
                closest_taxel_str.append(str(log_data['closest_taxel_ids'][i][closest_idx]))
            else:
                closest_taxel_str.append('N/A')
            if log_data['distance_to_taxels'][i][closest_idx] is not None:
                dist_closest_taxel_str.append(str(log_data['distance_to_taxels'][i][closest_idx]))
            else:
                dist_closest_taxel_str.append('N/A')
            
        writer.writerow(manip_cost_str)
        writer.writerow(manip_str)
        writer.writerow(closest_taxel_str)
        writer.writerow(dist_closest_taxel_str)
        # writer.writerow(['Local Manip Cost'] + [str(manip) for manip in log_data['manipulability']])
        # writer.writerow(['Distance to Taxel'] + [str(dist if dist is not None else 'N/A') for dist in log_data['distance_to_taxel']])
        # writer.writerow(['Closest Taxel ID'] + [str(taxel_id if taxel_id is not None else 'N/A') for taxel_id in log_data['closest_taxel_id']])
        for joint_idx in range(7):
            writer.writerow([f'Joint {joint_idx + 1}'] + [str(q[joint_idx]) for q in log_data['joint_position']])