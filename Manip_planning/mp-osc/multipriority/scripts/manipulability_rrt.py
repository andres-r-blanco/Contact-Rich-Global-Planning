import sys
import csv
from PIL import Image
sys.path.insert(1, r"C:\Users\arbxe\OneDrive\Desktop\code stuff\EmPRISE repos\Manip planning\mp-osc\pybullet_planning_master")
import pybullet as p
import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pybullet_data
import numpy as np
import time
import yaml
import threading
import math
# from motion.motion_planners import rrt
# import random
from pybullet_tools.utils import add_data_path, create_box, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF
from motion.motion_planners.utils import irange, apply_alpha, RED, INF, elapsed_time

# from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
# from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver

# Constants
# TAXEL_POS_1 = [-0.0002392337409993086,-0.042365606717658225, -0.0032268422164705435]
TAXEL_POS_1 = [0,0,0]
DEFAULT_LOWER_LIMIT = -3.14
DEFAULT_UPPER_LIMIT = 3.14
# SIM_2D = True
# if SIM_2D:
#     NOCOL = False
#     ZERO_ANGLES = [0,2,4,5,6]
#     GOAL_AREA_SAMPLE_PROB = 0.05
# else:
#     NOCOL = False
#     ZERO_ANGLES = []
#     GOAL_AREA_SAMPLE_PROB = 0.55
# RRT_MAX_ITERATIONS = 30000
# RRT_MIN_ITERATIONS = 500
# EXTEND_STEP_SIZE = 0.3
# GOAL_TOLERANCE = 0.1
# GOAL_AREA_DELTA = 2
# GOAL_SAMPLE_PROB = 0.05
# R_FOR_PARENT = EXTEND_STEP_SIZE*3
# EDGE_COLLISION_RESOLUTION = EXTEND_STEP_SIZE
NOCOL = False
VIZ = True
PAUSE_TIME = 0.1
TACTILE_KINOVA_URDF = "C:/Users/arbxe/OneDrive/Desktop/code stuff/EmPRISE repos/Manip planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12.urdf"
CSV_FOLDER_LOCATION = "C:/Users/arbxe/OneDrive/Desktop/code stuff/EmPRISE repos/Manip planning/mp-osc/multipriority/data/manip_data"
SCREENSHOT_FOLDER = "C:/Users/arbxe/OneDrive/Desktop/code stuff/EmPRISE repos/Manip planning/mp-osc/multipriority/data/screenshots"
# WEIGHT_FOR_DISTANCE = 0.85
# MANIP_DIF_THRESHOLD = 0
# MIN_PERCENT_MANIP_INCREASE = 0 #out of 100%
# LAZY = False
# UPDATE_PATH = False
# OLD_COST = True
# INCREASE_WEIGHT = False
# RANDOM_SEED = 10

def data_gathering():
    
    
    trial_num = 2
    truefalse = [True,False]
    test_main(trial_num, 2, 0.85, 0, True, True, False, False)
    test_main(trial_num, 2, 0.85, 0, False, True, False, False)
    test_main(trial_num, 2, 0.85, 0, True, False, False, False)
    test_main(trial_num, 2, 0.85, 0, False, False, False, False)
    # thresholds = [0,10,20]
    # weight_vals = [0.9,0.85,0.8,0.7]
    # for sim_type in [0]:
    #     sIM_type = sim_type
    #     for bool2 in truefalse:
    #         oLD_COST = bool2
    #         for bool3 in truefalse:
    #             lAZY = bool3
    #             for w in weight_vals:
    #                 for t in thresholds:
    #                     if (t==0 or lAZY is False):
    #                         test_main(trial_num, sIM_type, w, t, lAZY, oLD_COST, False, True)

                        
                        # test_main(trial_num, sIM_type, w, t, lAZY, oLD_COST, False, True)


def test_main(trial_num, sIM_type, wEIGHT_FOR_DISTANCE, mIN_PERCENT_MANIP_INCREASE, lAZY, oLD_COST, uPDATE_PATH, iNCREASE_WEIGHT):
    obstacles_3D = False
    sIM_2D = False
    if sIM_type == 0:
        sIM_2D = True
    elif sIM_type == 2:
        obstacles_3D = True

    if sIM_2D:
        zERO_ANGLES = [0,2,4,5,6]
        gOAL_AREA_SAMPLE_PROB = 0.05
        # gOAL_AREA_DELTA = 0.6
        gOAL_AREA_DELTA = 0.6
        eXTEND_STEP_SIZE = 0.2
        r_FOR_PARENT = eXTEND_STEP_SIZE*3
    else:
        zERO_ANGLES = []
        gOAL_AREA_SAMPLE_PROB = 0.05
        gOAL_AREA_DELTA = 0.6
        eXTEND_STEP_SIZE = 0.3
        r_FOR_PARENT = eXTEND_STEP_SIZE*3
    rRT_MAX_ITERATIONS = 5000
    # rRT_MIN_ITERATIONS = 1200
    rRT_MIN_ITERATIONS = 500
    gOAL_TOLERANCE = 0.1
    gOAL_SAMPLE_PROB = 0.05
    eDGE_COLLISION_RESOLUTION = eXTEND_STEP_SIZE

    i = 5
    rANDOM_SEED = i*15
    rng = np.random.default_rng(rANDOM_SEED)
    main(i,rANDOM_SEED, rng,sIM_2D, obstacles_3D, zERO_ANGLES,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
             r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,mIN_PERCENT_MANIP_INCREASE,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT)

    for i in range(1,trial_num+1):
        rANDOM_SEED = i*15
        rng = np.random.default_rng(rANDOM_SEED)
        main(i,rANDOM_SEED, rng,sIM_2D, obstacles_3D, zERO_ANGLES,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
             r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,mIN_PERCENT_MANIP_INCREASE,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT)

def main(trial_num, rANDOM_SEED, rng,sIM_2D, obstacles_3D, zERO_ANGLES,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
             r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,mIN_PERCENT_MANIP_INCREASE,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT):
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable GUI elements
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # Ensure rendering is enabled
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # Optional
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # Optional
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)  # Ensure RGB rendering is enabled

    add_data_path(pybullet_data.getDataPath())
    # draw_pose(Pose(), length=1.)

    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(TACTILE_KINOVA_URDF, [0, 0, 0],useFixedBase=1)
    obstacles = []
    if sIM_2D:
        set_camera_pose(camera_point=[0, -1, 0.7], target_point = [0, 0, 0.7])
        box_position = [-0.1, 0, 3]
        box_dims = [0.02,0.02,0.02]
        goal = [0,-1.1,0,-0.1,0,-0.2,0]
        start = [0,1.1,0,0.1,0,0.2,0]
        # goal = [0,-0.9,0,0.6,0,0.2,0]
        # goal = [0, 0.215, 0, 1.333, 0, 0, 0]
        # start = [0,0.9,0,0.1,0,0.2,0]
        col_box_id = create_box(box_position,box_dims)
        obstacles = [col_box_id]
    else:
        #3D mode
        set_camera_pose(camera_point=[-1.2, 0.9, 0.7], target_point = [0, 0, 0.5])
        box1_position = [-0.7, 0.1, 0.7]
        box1_dims = [0.05,0.05,0.1]
        box2_position = [0.7, 0.1, 0.7]
        box2_dims = [0.05,0.05,0.1]
        box3_position = [0, 0, 1]
        box3_dims = [1,1,0.05]
        goal = [1.7,-1,0.5,0.3,-0.2,0.2,-0.2]
        start = [-1.7,-1,0.2,-0.1,0.3,0.2,0.1]
        # box_position = [-0.7, 0, 0]
        # box_dims = [0.05,0.05,0.6]
        # goal = [0.5,-0.8,0,-0.8,0,0,0]
        # start = [-0.5,-0.8,0,-0.8,0,0,0]
        if obstacles_3D:
            col_box_id1 = create_box(box1_position,box1_dims)
            col_box_id2 = create_box(box2_position,box2_dims)
            col_box_id3 = create_box(box3_position,box3_dims)
            obstacles = [col_box_id1, col_box_id2, col_box_id3]

    
    

    # col_box_id = create_box(box_position,box_dims)
    # obstacles = [col_box_id]
    # taxel_ids = get_taxels(robot_id)


    # dump_body(robot_id)
    # print('Start?')
    # wait_for_user()

    joint_indices = [i for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

    q_init = start
    q_goal = goal
    # print("goal")
    # print(q_goal)
    pos_goal = q_to_endeff(robot_id, q_goal)
    # print(pos_goal)
    dot(pos_goal, [0,1,0,1])    
    set_joint_state(robot_id, q_goal, joint_indices)
    if VIZ:
        time.sleep(1)
    # print("start")
    # print(q_init)
    # print(q_to_endeff(robot_id, q_init))
    set_joint_state(robot_id, q_init, joint_indices)
    if VIZ:
        time.sleep(1)
    num_joints = p.getNumJoints(robot_id)

    # Loop through each joint and print its information
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        
        # Joint details
        joint_name = joint_info[1].decode('utf-8')  # Joint name
        joint_type = joint_info[2]  # Joint type (0=REVOLUTE, 1=PRISMATIC, 4=FIXED)
        
        # Check if it's a fixed joint
        is_fixed = joint_type == p.JOINT_FIXED
        # print(f"Joint {joint_index}: Name = {joint_name}, Fixed = {is_fixed}")
    
    rrt_obj = RRT_BASE(rng, robot_id, joint_indices, 
                 obstacles, eXTEND_STEP_SIZE, q_init, q_goal, gOAL_SAMPLE_PROB, gOAL_AREA_SAMPLE_PROB, 
                 gOAL_AREA_DELTA, rRT_MAX_ITERATIONS, rRT_MIN_ITERATIONS, eDGE_COLLISION_RESOLUTION, PAUSE_TIME, 
                 wEIGHT_FOR_DISTANCE, gOAL_TOLERANCE, r_FOR_PARENT, zERO_ANGLES,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT, mIN_PERCENT_MANIP_INCREASE, max_time=INF)
    

    node_path = rrt_obj.rrt_star_search()
    path = configs(node_path)
    print("done with trial num: " + str(trial_num))
    if path:
        # path = smooth(robot_id, path, joint_indices, obstacles)
        print("Found path:")
        # print_path(path)
        set_joint_state(robot_id, q_init)
        if VIZ:
            rrt_obj.visualize_path(path, line_color=[1, 0, 0])
        else:
            rrt_obj.dot_path(node_path)
    else:
        print("No path found")

    # print("Manip percent difference threshold: " + str(mIN_PERCENT_MANIP_INCREASE))
    # # print("Manip cost difference threshold: " + str(MANIP_DIF_THRESHOLD))
    # print("Min iterations: " + str(rRT_MIN_ITERATIONS))
    # print("Step Size: " + str(eXTEND_STEP_SIZE))
    # print("Parent finding radius R: " + str(round(r_FOR_PARENT,3)))
    # print("Goal Area Sample Probability: " + str(gOAL_AREA_SAMPLE_PROB))
    # print("Distance Weight: " + str(wEIGHT_FOR_DISTANCE))
    # print('Distance cost: ' + str(node_path[-1].d_cost))
    # print("Manip cost: " + str(node_path[-1].manip_cost))
    # print('Total cost: ' + str(node_path[-1].total_cost))
    # plot_manip(node_path)
    # plot_dist(node_path)
    # print('Done?')
    take_screenshot(trial_num, node_path, rANDOM_SEED, sIM_2D,obstacles_3D,zERO_ANGLES,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
             r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,mIN_PERCENT_MANIP_INCREASE,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT)
    save_data (trial_num, node_path, rANDOM_SEED, sIM_2D,obstacles_3D,zERO_ANGLES,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
             r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,mIN_PERCENT_MANIP_INCREASE,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT)
    wait_for_user()
    disconnect()

def take_screenshot(trial_num, node_path, rANDOM_SEED, sIM_2D,obstacles_3D,zERO_ANGLES,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
             r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,mIN_PERCENT_MANIP_INCREASE,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT):
    # Assuming PyBullet is already initialized and a simulation is running
    width, height = 2*640, 2*480  # Set the desired screenshot resolution
    # Use default view and projection matrices for the current camera view
    view_matrix = p.getDebugVisualizerCamera()[2]
    projection_matrix = p.getDebugVisualizerCamera()[3]
    # Capture the image
    image_data = p.getCameraImage(width, height, view_matrix, projection_matrix)

    # Extract RGB data from image_data
    rgb_array = np.array(image_data[2], dtype=np.uint8).reshape((height, width, 4))[:, :, :3]

    # Convert to an image and save it
    rgb_array = np.array(image_data[2], dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    image = Image.fromarray(rgb_array)
    png_name = "pybullet_screenshot_trial"+ str(trial_num) + "(seed" + str(rANDOM_SEED)+ ").png"
    if sIM_2D:
        dim = "2D"
    else:
        if obstacles_3D:
            dim = "3D_w_obstacles"
        else:
            dim = "3D"
    test_type = "".join([
        "Lazy", str(lAZY), "_Node cost function", str(oLD_COST),
        "_Percent Manip Threshold", str(mIN_PERCENT_MANIP_INCREASE), "_Update Path", str(uPDATE_PATH),
        "_Increasing weight", str(iNCREASE_WEIGHT), "_Min Iterations", str(rRT_MIN_ITERATIONS)
        ])
    weight = "".join(["Distance Weight", str(wEIGHT_FOR_DISTANCE)])
    print(test_type + weight)
    file_path = os.path.join(SCREENSHOT_FOLDER, dim, test_type, weight, png_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    image.save(file_path)


def save_data (trial_num, node_path, rANDOM_SEED, sIM_2D,obstacles_3D,zERO_ANGLES,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
             r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,mIN_PERCENT_MANIP_INCREASE,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT):
    
    path = configs(node_path)
    joint_path_matrix = np.empty((len(path[0]), 0))
    for config in path:
        # Round each joint angle in the configuration to 3 decimal points
        rounded_config = np.array([round(angle, 3) for angle in config])
        # Print the rounded configuration
        rounded_config = rounded_config.reshape(-1, 1)
        joint_path_matrix = np.hstack((joint_path_matrix, rounded_config))

    manip = []
    for node in node_path:
        manip.append(node.manipulability)
    manip = np.array(manip)
    manip = [round(x, 3) for x in manip]

    dist_to_last = []
    for node in node_path:
        dist_to_last.append(node.dist_to_last)
    dist_to_last = np.array(dist_to_last)
    dist_to_last = [round(x, 3) for x in dist_to_last]

    if sIM_2D:
        dim = "2D"
    else:
        if obstacles_3D:
            dim = "3D_w_obstacles"
        else:
            dim = "3D"
    test_type = "".join([
        "Lazy", str(lAZY), "_Node cost function", str(oLD_COST),
        "_Percent Manip Threshold", str(mIN_PERCENT_MANIP_INCREASE), "_Update Path", str(uPDATE_PATH),
        "_Increasing weight", str(iNCREASE_WEIGHT), "_Min Iterations", str(rRT_MIN_ITERATIONS)
        ])
    csv_filepath = os.path.join(CSV_FOLDER_LOCATION, dim, test_type)
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)

    with open(csv_filepath, 'a', newline='') as fd:
        writer = csv.writer(fd)
    
        # Write the `manip` list with title in the first column
        writer.writerow(['Trial'] + [str(trial_num)] + ["2D Sim"] + [str(sIM_2D)] 
                        + ["Distance Weight"] + [str(wEIGHT_FOR_DISTANCE)] + ["Random Seed"] + [str(rANDOM_SEED)]
                        + ["Min Iterations"] + [str(rRT_MIN_ITERATIONS)] + ["Lazy"] + [str(lAZY)]
                        + ["Distance cost"] + [str(node_path[-1].d_cost)] + ["Manip Cost"] + [str(node_path[-1].manip_cost)] 
                        + ["Total cost"] + [str(node_path[-1].total_cost)] + ["Node cost function"] + [str(oLD_COST)] 
                        + ["Percent Manip Threshold"] + [str(mIN_PERCENT_MANIP_INCREASE)] + ["Update Path"] + [str(uPDATE_PATH)]
                        + ["Increasing weight"] + [iNCREASE_WEIGHT]
                        + ["Parent finding radius R"] + [str(round(r_FOR_PARENT, 3))] + ["Step Size"] + [str(eXTEND_STEP_SIZE)]
                        + ["Goal Area Sample Probability"] + [str(gOAL_AREA_SAMPLE_PROB)] + ["Goal Area Delta"] + [str(gOAL_AREA_DELTA)])

        writer.writerow(['Manipulability'] + manip)
        
        # Write the `dist` list with title in the first column
        writer.writerow(['Distance to Last Node'] + dist_to_last)
        for i in range(joint_path_matrix.shape[0]):
            row = joint_path_matrix[i,:]
            row = [str(x) for x in row]
            joint_text = 'Joint ' + str(i) + ': '
            writer.writerow([joint_text] + row)
        

class RRT_BASE(object):
    def __init__(self, rng, robot_id, joint_indices, 
                 obstacles, extend_step, q_init, q_goal, goal_probability, goal_area_probability,  
                 goal_area_delta, max_samples, min_samples, res, pause_time, weight_dist, goal_tolerance,
                 R, zERO_ANGLES,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT, mIN_PERCENT_MANIP_INCREASE, max_time=INF, prc=0.01):
        """
        Template RRT planner
        :param extend_step: length of edges added to tree
        :param q_init: tuple, initial location
        :param q_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        self.robot_id = robot_id
        self.R = R #radius for parent finding
        self.joint_indices = joint_indices
        self.goal_tolerance = goal_tolerance
        self.goal_probability = goal_probability
        self.goal_area_probability = goal_area_probability
        self.goal_area_delta = goal_area_delta
        self.obstacles = obstacles
        self.samples_taken = 0
        self.res = res
        self.prc = prc
        self.q_init = q_init
        self.q_goal = q_goal
        self.nodes = [TreeNode(self.q_init,manipulability = self.calculate_manipulability(self.q_init, TAXEL_POS_1))]
        self.extend_step =extend_step
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.pause_time = pause_time
        self.max_weight_dist = weight_dist
        self.weight_dist = weight_dist
        self.max_time = max_time
        self.num_joints = p.getNumJoints(self.robot_id)
        self.threshold = mIN_PERCENT_MANIP_INCREASE
        self.zero_angles = zERO_ANGLES
        self.lazy = lAZY
        self.will_update_path = uPDATE_PATH
        self.use_old_cost = oLD_COST
        self.increase_weight = iNCREASE_WEIGHT
        self.rng = rng

        [self.upper_limits,self.lower_limits] = self.get_joint_limits()
    
    def rrt_star_search(self):
        start_time = time.time()
        if self.collision_fn(self.q_init) or self.collision_fn(self.q_goal):
            return False
        while self.samples_taken < self.max_samples:
            # print(self.samples_taken)
            if self.increase_weight:
                r = min(self.samples_taken/(self.max_samples*(2/3)),1)
                # r = abs(0.5-r)*2
                self.weight_dist = r*(self.max_weight_dist) + (1-r)
            if elapsed_time(start_time) >= self.max_time:
                print("max time reached")
                break
            # will only sample goal (trying to get a solution) when enough samples were taken
            if self.samples_taken >= self.min_samples and (self.samples_taken == self.min_samples or self.rng.random() < self.goal_probability):
                    target_config = self.q_goal
                    # print("sampling goal")   
                    # dot(q_to_endeff(self.robot_id, target_config), [1,0,0,1]) # visualizing sampled point in red             
            else:
                goal_sample_time = self.rng.random() < self.goal_area_probability
                target_config = self.goal_area_sample() if goal_sample_time else self.sample_fn()
            # dot(q_to_endeff(self.robot_id, target_config), [1,0,0,1]) # visualizing sampled point in red
            self.samples_taken = self.samples_taken + 1
            closest = self.get_closest_node(target_config) # getting nearest node
            [q,parent] = self.extend_fn(closest, target_config) # extending from nearest node, returning a parent only if there already exists a node at the sampled location
            if q is not None:
                [_,node_path] = self.create_node(q,parent,True)
                if self.samples_taken > self.min_samples and node_path is not None:
                    if self.will_update_path:
                        node_path = self.update_path(node_path) #computes new parents and costs starting from goal
                    return node_path
            
        return None
    
# self.check_solution_euclidian()
#  if self.samples_taken > self.min_samples and path is not None:
#         return path
        
    def create_node(self, q, parent = None, same_node = False):
            # blue_dot(self.robot_id)
            manip = self.calculate_manipulability(q, TAXEL_POS_1)
            # gradient_dot(self.robot_id, manip)
            if parent is None or same_node is False:
                node = TreeNode(q, manipulability=manip)
                if parent is None:
                    new_parent = self.find_parent(node,self.threshold)
                else:
                    new_parent = parent
                node.parent = new_parent
                node.num_in_path = new_parent.num_in_path + 1
                [node.total_cost, node.d_cost, node.dist_to_last, node.manip_cost] = self.cost_fn(new_parent,node)
                self.nodes.append(node)
            else:
                node = parent
                new_parent = self.find_parent(node,self.threshold)
                node.parent = new_parent
                [node.total_cost, node.d_cost, node.dist_to_last, node.manip_cost] = self.cost_fn(new_parent,node)
            #check if node within goal tolerance was sampled
            if self.goal_test(node.config):
                    return node, node.retrace()

            return node, None
    
    def find_parent(self, new_node, threshold, no_list = []):
        no_list_configs = {tuple(node.config) for node in no_list}
        nodes_in_radius = [
            node for node in self.nodes 
            if node.config != new_node.config
            and tuple(node.config) not in no_list_configs
            and self.distance_fn(new_node.config, node.config) < self.R
        ]
        if nodes_in_radius:
            lowest_dist_parent = self.argmin(lambda n: self.cost_fn(n, new_node)[1], nodes_in_radius) # lowest dist cost node
            full_cost_parent = self.argmin(lambda n: self.cost_fn(n, new_node)[0], nodes_in_radius) # lowest total cost node
            lowest_dist_manip_cost = self.cost_fn(lowest_dist_parent, new_node)[3]
            full_cost_parent_manip_cost = self.cost_fn(full_cost_parent, new_node)[3]
            manip_percent = lowest_dist_manip_cost/full_cost_parent_manip_cost - 1
            
            # print("manip percent: " + str(100*manip_percent))
            # print("manip cost difference between lowest dist and lowest total cost: " + str(manip_dif))
            # choosing lowest total cost parent only if manip difference to lowest distance cost parent is significant
            if manip_percent < threshold/100:
                parent = lowest_dist_parent
            else:
                parent = full_cost_parent
            #interpolating between nodes if dist is greater than step size, creating new ones and checking for collisions
            if self.lazy is False and np.max(np.abs(np.array(parent.config)[:]-np.array(new_node.config)[:])) > self.extend_step:
                num_steps = math.floor(np.max(np.abs(np.array(parent.config)[:]-np.array(new_node.config)[:]))/self.extend_step) 
                q_list = self.interpolate_configs(parent.config, new_node.config, num_steps = num_steps+1)
                q_list.pop()
                parent_interp_node = parent
                for i in range(len(q_list)):
                    q_interp = q_list[i]
                    if not self.collision_fn(q_interp):
                        [parent_interp_node,_] = self.create_node(q_interp, parent_interp_node)
                    else:
                        return self.get_closest_node(new_node.config)
                return parent_interp_node
            else:
                return parent
        return self.get_closest_node(new_node.config)


    def get_closest_node(self,q):
        target_node = TreeNode(q, manipulability=self.calculate_manipulability(q, TAXEL_POS_1))
        return self.argmin(lambda n: self.distance_fn(n.config, target_node.config), self.nodes)

    def get_closest_node_euclidian(self,q):
        target_node = TreeNode(q, manipulability=self.calculate_manipulability(q, TAXEL_POS_1))
        return self.argmin(lambda n: self.euclid_distance_fn(n.config, target_node.config), self.nodes)


    def sample_fn(self):
        # Generate random joint angles within the joint limits
        random_angles = [self.rng.uniform(low, high) for low, high in zip(self.lower_limits[:], self.upper_limits[:])]
        zero_angles = self.zero_angles
        for i in zero_angles:
            random_angles[i] = 0
        return random_angles

    def goal_area_sample(self):
        return self.sample_around_fn(self.q_goal, self.goal_area_delta)

    def sample_around_fn(self, target, delta = 0.4):
        target = np.array(target)
        # Generate random joint angles within the joint limits
        lower_limits = target - delta
        upper_limits = target + delta
        random_angles = [self.rng.uniform(low, high) for low, high in zip(lower_limits[:], upper_limits[:])]
        zero_angles = self.zero_angles
        for i in zero_angles:
            random_angles[i] = 0
        return random_angles

    def distance_fn(self, q1, q2):
        # distance between two configurations (joint angles)
        return np.linalg.norm(np.array(q1) - np.array(q2))

    def euclid_distance_fn(self, q1, q2):
        # Euclidean distance between two configurations (joint angles)
        pos1 = q_to_endeff(self.robot_id, q1)
        pos2 = q_to_endeff(self.robot_id, q2)
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def calculate_manipulability(self, q, taxel_pos):
        # if isinstance(q, np.ndarray):
        #     q = q.tolist() 
        J, _ = p.calculateJacobian(self.robot_id, 8, taxel_pos, q, [0]*7, [0]*7)
        J = correct_J_form(self.robot_id, J)
        JJT = np.dot(J, np.transpose(J))
        det_JJT = np.linalg.det(JJT)
        if det_JJT < 0:
            manipulability = 1 
        else:
            manipulability = np.sqrt(det_JJT)
        return manipulability

    # def og_cost_fn(self, start_node, target_node):
    #     distance = self.distance_fn(start_node.config, target_node.config)
    #     max_distance = self.distance_fn(self.upper_limits, self.lower_limits)
    #     dist_to_last = distance/max_distance
    #     distance_cost = dist_to_last + start_node.d_cost

    #     target_manip_cost = 1/(target_node.get_manip())
    #     manip_cost = (target_manip_cost + start_node.manip_cost)/(1+start_node.num_in_path)

    #     total_cost = self.weight_dist*distance_cost + manip_cost*(1-self.weight_dist)

    #     return total_cost, distance_cost, dist_to_last, manip_cost

    def cost_fn(self, start_node, target_node):
        distance = self.distance_fn(start_node.config, target_node.config)
        max_distance = self.distance_fn(self.upper_limits, self.lower_limits)
        normalized_dist_to_last = distance/max_distance
        parent_dist_cost = start_node.d_cost
        distance_cost = normalized_dist_to_last + parent_dist_cost

        if self.use_old_cost is False:
            avg_manip_cost = 2 / (target_node.get_manip() + start_node.get_manip())
            added_manip_cost = normalized_dist_to_last * avg_manip_cost
            parent_manip_times_dist = parent_dist_cost * start_node.manip_cost
            manip_cost = (added_manip_cost + parent_manip_times_dist) / distance_cost
        else:
            manip_cost = (1/target_node.get_manip() + start_node.manip_cost*start_node.num_in_path)/(1+start_node.num_in_path)

        total_cost = self.weight_dist*distance_cost + manip_cost*(1-self.weight_dist)
        
        return total_cost, distance_cost, distance, manip_cost

    # def compare_parents_manip

    def argmin(self, function, nodes):
        min_node = min(nodes, key=function)
        return  min_node
        # values = list(nodes)
        # scores = [function(x) for x in values]
        # return values[scores.index(min(scores))]

    def interpolate_configs(self, config1, config2, num_steps=10):
        """Interpolate between two configurations. exclusive of start, inclusive of end"""
        return [
            [(1 - t) * c1 + t * c2 for c1, c2 in zip(config1, config2)]
            for t in (i / float(num_steps) for i in range(1, num_steps + 1))
        ]

    # def check_solution_euclidian(self):
    #         # probabilistically check if solution found
    #         if (self.prc and random.random() < self.prc) or self.samples_taken >= self.max_samples:
    #             print("Checking if can connect to goal at", str(self.samples_taken), "samples")
    #             closest = self.get_closest_node_euclidian(self.q_goal)
    #             set_joint_state(self.robot_id, closest.config)
    #             q_goal_array = np.array(self.q_goal)
    #             num_steps = math.ceil(np.max(np.abs(closest.config[:]-q_goal_array[:]))/self.res) 
    #             q_list = self.interpolate_configs(closest.config, self.q_goal, num_steps = num_steps)
    #             for q in q_list:
    #                 if self.collision_fn(q):
    #                     return None
    #                 set_joint_state(self.robot_id, q)
    #                 blue_dot(self.robot_id)
    #                 [new_node, path] = self.create_node(q)
    #                 if path is not None:
    #                     return path
    #         return None

    # def move_through_path(self, path, time_step=0.05):
    #     # Assume the robot has a known number of joints
    #     set_joint_state(self.robot_id, path[0])
    #     for config in path:
    #         # Set joint motor control for all joints
    #         p.setJointMotorControlArray(
    #             bodyUniqueId=self.robot_id,
    #             jointIndices=list(range(self.num_joints)),  # Indices of all joints
    #             controlMode=p.POSITION_CONTROL,        # Control mode
    #             targetPositions=config                  # Desired joint positions
    #         )
            
    #         # Step the simulation for the given time step
    #         p.stepSimulation()
    #         time.sleep(time_step)  # Optional: adjust the sleep duration for smoother movement

    def  extend_fn(self, q1_node, q2):
        q1 = q1_node.config
        # Calculate the direction vector and its norm
        direction = [q2[i] - q1[i] for i in range(len(q1))]
        distance = np.linalg.norm(direction)
        if distance == 0: #ensures new nodes aren't created when sampling goal
            return q2, q1_node
        elif distance <= self.extend_step:   # If the distance is smaller than the extend step, return q2 as the new point
            q = q2
        else:
            # Normalize the direction vector and scale by the extend step
            normalized_direction = [d / distance for d in direction]
            q = [q1[i] + self.extend_step * normalized_direction[i] for i in range(len(q1))]
        
        if not self.collision_fn(q):
            return q, None
        return None, None

    def collision_fn(self, q):
        if NOCOL:
            return False
        # Set the robot's joints to the given configuration
        for joint_index in self.joint_indices:
            p.resetJointState(self.robot_id, joint_index, q[joint_index])
        
        # Check for collisions between the robot and obstacles
        for obstacle in self.obstacles:
            contact_points = p.getClosestPoints(self.robot_id, obstacle, distance=0)
            if len(contact_points) > 0:
                return True  # Collision detected
        return False

    def goal_test(self, q):
        # Check if the current configuration is within the tolerance of the goal
        return self.distance_fn(q, self.q_goal) < self.goal_tolerance

    def update_path(self, node_path):
        goal = node_path[-1]
        node = goal
        no_list = []
        while node.dist_to_last > 0:
            no_list.append(node)
            new_parent = self.find_parent(node,0,no_list = no_list)
            print(node.config)
            print(new_parent.config)
            if new_parent is not None:
                node.parent = new_parent
                [node.total_cost, node.d_cost, node.dist_to_last, node.manip_cost] = self.cost_fn(new_parent,node)
            node = node.parent
            wait_for_user()
        return goal.retrace()

    def get_joint_limits(self):
        upper_limits = [0] * len(self.joint_indices)
        lower_limits = [0] * len(self.joint_indices)

        for i, joint_index in enumerate(self.joint_indices):
            joint_info = p.getJointInfo(self.robot_id, joint_index)
        # Extract the  lower limits, and upper limits
            low = joint_info[8]
            high = joint_info[9]
            if high <= low:
                lower_limits[i] = DEFAULT_LOWER_LIMIT
                upper_limits[i] = DEFAULT_UPPER_LIMIT
            else:
                lower_limits[i] = joint_info[8]
                upper_limits[i] = joint_info[9]
        return [upper_limits,lower_limits]
    
    def dot_path(self, node_path):
        for node in node_path:
            config = node.config
            set_joint_state(self.robot_id, config)
            manip = node.manipulability
            gradient_dot(self.robot_id, manip)

    def visualize_path(self, path, line_color=[1, 0, 0]):
        """
        Visualizes the RRT path by moving the robot through the configurations in the path
        and drawing lines between them.

        :param robot_id: ID of the robot in PyBullet
        :param joint_indices: List of joint indices to control
        :param path: List of configurations (each configuration is a list of joint angles)
        :param line_color: RGB color for the trajectory line (default: red)
        :param pause_time: Time to pause between each step for visualization (in seconds)
        """
        # Remove any existing lines
        line_ids = []


        # Loop through each configuration in the path
        for i, q in enumerate(path):
            time.sleep(0.2)
    
            # If it's not the first configuration, draw a line from the previous one to the current one, and interpolate for visuals
            if i > 0:
                prev_q = path[i - 1]

                #interpolate and to visualize moving from waypoint to waypoint
                num_steps = math.ceil(5*np.max(np.abs(np.array(q)[:]-np.array(prev_q)[:]))/self.res) 
                q_list = self.interpolate_configs(prev_q, q, num_steps = num_steps)
                for q_interp in q_list:
                    time.sleep(0.2/num_steps)
                    set_joint_state(self.robot_id, q_interp)
                    # for joint_idx, joint_angle in zip(self.joint_indices, q_interp):
                        # p.setJointMotorControl2(self.robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=joint_angle)
                    p.stepSimulation()

                # Get the 3D world positions of the end-effector for the previous and current configuration
                prev_pos = q_to_endeff(self.robot_id, prev_q)
                curr_pos = q_to_endeff(self.robot_id, q)
                # Draw a line between the two configurations
                line_id = p.addUserDebugLine(prev_pos, curr_pos, lineColorRGB=line_color, lineWidth=2.0)
                line_ids.append(line_id)

                
            set_joint_state(self.robot_id, q)
                # p.setJointMotorControl2(self.robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=joint_angle)
            # Step the simulation to allow the robot to move
            p.stepSimulation()

        # move_through_path(robot_id, path, joint_indices, time_step=pause_time)
        return line_ids


def print_path(path):
    for config in path:
        # Round each joint angle in the configuration to 3 decimal points
        rounded_config = [round(angle, 3) for angle in config]
        # Print the rounded configuration
        print(rounded_config)
    return

def set_joint_state(robot_id, q, joint_indices=[0,1,2,3,4,5,6]):
        p.stepSimulation()
        for joint_index in joint_indices:
            p.resetJointState(robot_id, joint_index, q[joint_index])

def configs(nodes):
    if nodes is None:
        return None
    return list(map(lambda n: n.config, nodes))

def q_to_endeff(robot_id, q):
        set_joint_state(robot_id, q)
        link_state = p.getLinkState(robot_id, 7)
        return link_state[4]

def correct_J_form(robot_id, J):
    num_joints = 7
    J_out = np.zeros((3, num_joints))
    
    # Fill the corrected Jacobian and track active indices
    i = 0
    for c in range(num_joints):
        if p.getJointInfo(robot_id, c)[3] > -1:  # Check if the joint is active
            for r in range(3):
                J_out[r,i] += J[r][i]
            i += 1
    # J_out[0,:] = J[0][:]
    # J_out[1,:] = J[1][:]
    # J_out[2,:] = J[2][:]
    return J_out

def gradient_dot(robot_id, manip):
    # Get the position of the end-effector (assuming it's the last joint/link)
    link_state = p.getLinkState(robot_id, 7)
    end_effector_position = link_state[4]  # Position is index [4] in the result
    manip = min(1,manip/0.4)
    # Create a small visual sphere at the end-effector's position
    dot_radius = 0.01  # A small dot
    color = [manip, 0, 1-manip, 1]  # RGBA color for blue

    # Create a visual shape for the dot (small sphere)
    dot_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=dot_radius, rgbaColor=color)

    # Create the body for the visual shape (this places the sphere at the end-effector position)
    p.createMultiBody(
        baseMass=0,  # Mass of 0 means it's static and won't be affected by physics
        baseVisualShapeIndex=dot_visual_shape,
        basePosition=end_effector_position
    )

def blue_dot(robot_id):
    # Get the position of the end-effector (assuming it's the last joint/link)
    link_state = p.getLinkState(robot_id, 7)
    end_effector_position = link_state[4]  # Position is index [4] in the result

    # Create a small visual sphere at the end-effector's position
    dot_radius = 0.01  # A small dot
    blue_color = [0, 0, 1, 1]  # RGBA color for blue

    # Create a visual shape for the dot (small sphere)
    dot_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=dot_radius, rgbaColor=blue_color)

    # Create the body for the visual shape (this places the sphere at the end-effector position)
    p.createMultiBody(
        baseMass=0,  # Mass of 0 means it's static and won't be affected by physics
        baseVisualShapeIndex=dot_visual_shape,
        basePosition=end_effector_position
    )

def dot(pos,color = [1,0,0,1]):
    # Create a small visual sphere at the end-effector's position
    dot_radius = 0.01  # A small dot

    # Create a visual shape for the dot (small sphere)
    dot_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=dot_radius, rgbaColor=color)
    # Create the body for the visual shape (this places the sphere at the end-effector position)
    p.createMultiBody(
        baseMass=0,  # Mass of 0 means it's static and won't be affected by physics
        baseVisualShapeIndex=dot_visual_shape,
        basePosition=pos
    )

def normal_to_quaternion(normal):
    # Convert normal vector to quaternion (assuming normal points along z-axis)
    x, y, z = normal
    yaw = np.arctan2(y, x)  # Yaw from x and y components
    pitch = np.arctan2(z, np.sqrt(x**2 + y**2))  # Pitch from z component
    roll = 0  # Assume no roll
    return p.getQuaternionFromEuler([roll, pitch, yaw])

def get_taxels(robot_id, yaml_file=r'C:\Users\arbxe\OneDrive\Desktop\code stuff\EmPRISE repos\Manip planning\mp-osc\multipriority\configs\real_taxel_data_v2.yaml'):
    # Load YAML data
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    taxel_ids = []
    
    # Create objects based on YAML data
    for key, values in data.items():
        link = values['Link']
        local_position = np.array(values['Position'])
        normal = np.array(values['Normal'])

        # Get the current world position and orientation of the link
        link_state = p.getLinkState(bodyUniqueId=robot_id, linkIndex=link)
        link_world_pos = np.array(link_state[0])  # World position
        link_world_orientation = link_state[1]     # World orientation (quaternion)
        
        # Calculate the world position of the taxel based on the link's transform
        link_rotation_matrix = np.array(p.getMatrixFromQuaternion(link_world_orientation)).reshape(3, 3)
        world_position = link_world_pos + link_rotation_matrix.dot(local_position)

        # Convert the normal vector to quaternion orientation
        orientation = normal_to_quaternion(normal)
        
        # Define a small rectangular box shape for the taxel
        half_extents = [0.01, 0.02, 0.03]  # Adjust dimensions as needed
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=half_extents, 
            rgbaColor=[0, 0, 1, 1]  # Blue for visibility
        )
        
        # Create the taxel as a visual body
        taxel_id = p.createMultiBody(
            baseVisualShapeIndex=visual_shape, 
            basePosition=world_position.tolist(), 
            baseOrientation=orientation,
            baseInertialFramePosition=[0, 0, 0],
        )
        
        # Attach the taxel to the robot link with a fixed constraint
        p.createConstraint(
            parentBodyUniqueId=robot_id, 
            parentLinkIndex=link, 
            childBodyUniqueId=taxel_id, 
            childLinkIndex=-1, 
            jointType=p.JOINT_FIXED, 
            jointAxis=[0, 0, 0], 
            parentFramePosition=local_position.tolist(),  # Local position relative to link
            childFramePosition=[0, 0, 0], 
        )
        
        # Add taxel ID to the list
        taxel_ids.append(taxel_id)
    
    return taxel_ids


def get_distance_between_bodies(body_id1, body_id2):
    # Get positions of both bodies
    pos1, _ = p.getBasePositionAndOrientation(body_id1)
    pos2, _ = p.getBasePositionAndOrientation(body_id2)
    
    # Calculate the Euclidean distance between the two positions
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
    return distance

def plot_dist(path_nodes):
    dist = []
    for node in path_nodes:
        if node.dist_to_last is not None:
            dist.append(node.dist_to_last)

    dist = np.array(dist)
    dist = [round(x, 3) for x in dist]

    print("Dist:", dist)

def plot_manip(path_nodes):
    manip = []
    for node in path_nodes:
        manip.append(node.manipulability)

    manip = np.array(manip)
    manip = [round(x, 3) for x in manip]
    x  = np.linspace(0, len(manip)-1, len(manip))
    x = np.array(x)

    print("Manip:", manip)
    # print("X:", x)

    # plt.figure(figsize=(8, 6))

    # Plot Manip against X
    # plt.plot(x, manip, marker='o', label='Manip Values')

    # # Add titles and labels
    # plt.title('Manip over Trajectory')
    # plt.xlabel('Nodes')
    # plt.ylabel('Manip Values')
    # plt.ylim(bottom=0)  # Set the lower limit of y-axis to 0 for better visualization
    # plt.legend()
    # plt.grid(True)

    # Show the plot
    # plt.show(block=False)

class TreeNode(object):

    def __init__(self, config, parent=None,manipulability=0, dist_to_last = 0, d_cost = 0, manip_cost = 0, num_in_path = 1):
        self.config = config
        self.parent = parent
        self.manipulability = manipulability
        self.dist_to_last = dist_to_last
        self.manip_cost = manip_cost
        self.d_cost = d_cost
        self.total_cost = manip_cost+d_cost
        self.num_in_path = num_in_path

    #def retrace(self):
    #    if self.parent is None:
    #        return [self]
    #    return self.parent.retrace() + [self]

    def retrace(self):
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def get_manip(self):
        return self.manipulability

    def clear(self):
        self.node_handle = None
        self.edge_handle = None

    def draw(self, env, color=apply_alpha(RED, alpha=0.5)):
        # https://github.mit.edu/caelan/lis-openrave
        from manipulation.primitives.display import draw_node, draw_edge
        self.node_handle = draw_node(env, self.config, color=color)
        if self.parent is not None:
            self.edge_handle = draw_edge(
                env, self.config, self.parent.config, color=color)

    def __str__(self):
        return 'TreeNode(' + str(self.config) + ')'
    __repr__ = __str__

if __name__ == '__main__':
    data_gathering()


# def smooth(self, N=100):
#         for _ in range(N):
#             # Pick two random points on the path
#             index1 = random.randint(0, len(path) - 1)
#             index2 = random.randint(0, len(path) - 1)
#             if index1 > index2:
#                 index1, index2 = index2, index1
#             config1 = path[index1]
#             config2 = path[index2]
#             # Interpolate between the two configurations
#             interpolated_configs = interpolate_configs(config1, config2)
#             # Check for collisions along the interpolated path
#             collision_free = True
#             for i in range(len(interpolated_configs) - 1):
#                 if collision_fn(robot_id, joint_indices, interpolated_configs[i], obstacles):
#                     collision_free = False
#                     break
#             # If no collision was detected, snip out the segment
#             if collision_free:
#                 path = path[:index1 + 1] + path[index2:]  # Keep the start to index1 and from index2 to the end

#         new_path = path[1]
#         for i in range(len(path)-1):
#             new_path = new_path + interpolate_configs(path[i], path[i+1], num_steps=10) + path[i+1]
#         return new_path