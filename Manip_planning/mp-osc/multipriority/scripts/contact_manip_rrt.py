import sys
import csv
from PIL import Image
sys.path.insert(1, r"/home/rishabh/Andres/Manip_planning/mp-osc/pybullet_planning_master")
sys.path.insert(1, r"/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/urdfs")
import pybullet as p
from mink_IK_collision_free import MinkIKSolver, build_mujoco_obstacle_list, inject_obstacles
import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pybullet_data
import numpy as np
import time
import yaml
import threading
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import math
from copy import deepcopy
from xml.etree import ElementTree as ET
import re
import pinocchio as pin
from pinocchio import SE3, XYZQUATToSE3
import mujoco
import mujoco.viewer
print("Pinocchio version:", pin.__version__)


# from motion.motion_planners import rrt
# import random
from pybullet_tools.utils import add_data_path, create_box, create_cylinder, quat_from_euler, connect, dump_body, disconnect, wait_for_user, \
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
TEST_Q = False
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
VIZ = False
NO_DOT = False
COL_COST_RATIO = 0.05
DISTANCE_THRESHOLD = 0.1
MAX_PENETRATION = 0.03
RANDOM_IK_CHANCE = 0
PAUSE_TIME = 0.3
JS_EXTEND_MAX = 1.5
SAVE_DATA_PREFIX = "reach_agg_manip_cost"
DATA_FOLDER_NAME = SAVE_DATA_PREFIX
# CONTACT_SAMPLE_PROB = 0
ANGLE_DIF_ALLOWED = math.radians(45)
DEFAULT_IK_RETRIES = 0
DEFAULT_CONTACT_IK_RETRIES = 100
REDUCE_OBJECTS = True
DEBUG_LINE_LIFETIME = 0 #0.1
GIF_FRAME_DURATION = 300 #milliseconds
TACTILE_KINOVA_URDF = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12_w_taxels.urdf"
TACTILE_KINOVA_URDF_FOR_PIN = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12_for_pin.urdf"
# TACTILE_KINOVA_URDF_FOR_MUJOCO= "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12_for_mujoco.urdf"
TACTILE_KINOVA_URDF_FOR_MUJOCO="/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/urdfs/gen3_taxels_with_sites_mjcf.xml"
CSV_FOLDER_LOCATION = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/data/manip_data"
SCREENSHOT_FOLDER = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/data/screenshots"
TEMP_SCREENSHOT_FOLDER = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/data/temp_screenshots"
OUTPUT_GIF_FOLDER= "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/data/output_gifs"
# WEIGHT_FOR_DISTANCE = 0.85
# MANIP_DIF_THRESHOLD = 0
# MIN_PERCENT_MANIP_INCREASE = 0 #out of 100%
# LAZY = False
# UPDATE_PATH = False
# OLD_COST = True
# INCREASE_WEIGHT = False
RANDOM_SEED = 25

def new_main():
    data_gathering()
    trial_num = 1
    sIM_type = 5
    wEIGHT_FOR_DISTANCE = 0.9
    mIN_PERCENT_MANIP_INCREASE = 0
    min_iterations = 500
    lAZY = False
    oLD_COST = True
    uPDATE_PATH = False
    iNCREASE_WEIGHT = False
    test_main(trial_num, sIM_type, min_iterations, wEIGHT_FOR_DISTANCE, 0.02, 0.0, mIN_PERCENT_MANIP_INCREASE, lAZY, oLD_COST, uPDATE_PATH, iNCREASE_WEIGHT)

def data_gathering():
    
    trial_num = 30
    save = True
    lAZY = False
    oLD_COST = True
    uPDATE_PATH = False
    iNCREASE_WEIGHT = False
    min_iterations = 2500
    sIM_type = 5
    # thresholds = [0,5,10,15,20,25]
    weight_val = [0.8, 0.9, 1, 0.95,0.97,0.7]
    object_reduction = [0.02,0.0]
    for obj in object_reduction:
        for w in weight_val:
            test_main(trial_num, sIM_type, min_iterations, w, obj, 0.0, 0, lAZY, oLD_COST, uPDATE_PATH, iNCREASE_WEIGHT,save=save)
        # test_main(trial_num, sIM_type, w, 0.0, 0.2, 0, lAZY, oLD_COST, uPDATE_PATH, iNCREASE_WEIGHT,save=save)
        
def test_main(trial_num, sIM_type, min_iterations, wEIGHT_FOR_DISTANCE, object_reduction, contact_sample_chance, mIN_PERCENT_MANIP_INCREASE, lAZY, oLD_COST, uPDATE_PATH, iNCREASE_WEIGHT,save = False):
    sIM_2D = False
    if sIM_type == 0:
        sIM_2D = True
    elif sIM_type >= 2:
        obstacle_num = True

    if sIM_2D:
        zERO_ANGLES = [0,2,4,5,6]
        gOAL_AREA_SAMPLE_PROB = 0.1
        # gOAL_AREA_DELTA = 0.6
        gOAL_AREA_DELTA = 0.25
        eXTEND_STEP_SIZE = 0.07
        r_FOR_PARENT = eXTEND_STEP_SIZE*3
    else:
        zERO_ANGLES = []
        gOAL_AREA_SAMPLE_PROB = 0.05
        gOAL_AREA_DELTA = 0.6
        if sIM_type >= 2:
            gOAL_AREA_SAMPLE_PROB = 0.2
            gOAL_AREA_DELTA = 0.2
        eXTEND_STEP_SIZE = 0.05
        r_FOR_PARENT = eXTEND_STEP_SIZE*2.5
    rRT_MAX_ITERATIONS = 10000
    rRT_MIN_ITERATIONS = min_iterations
    xyz_gOAL_TOLERANCE = 0.01
    gOAL_SAMPLE_PROB = 0.05
    
    eDGE_COLLISION_RESOLUTION = eXTEND_STEP_SIZE
    if save == False:
        rng = np.random.default_rng(15)
        main(1,15, rng, sIM_type, zERO_ANGLES,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,xyz_gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
                r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,mIN_PERCENT_MANIP_INCREASE,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT,object_reduction,contact_sample_chance)
    else:
        for i in range(1,trial_num+1):
            rANDOM_SEED = i*15
            rng = np.random.default_rng(rANDOM_SEED)
            main(i,rANDOM_SEED, rng, sIM_type, zERO_ANGLES,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,xyz_gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
                r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,mIN_PERCENT_MANIP_INCREASE,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT,object_reduction,contact_sample_chance, save = True)
    
    # rng = np.random.default_rng(RANDOM_SEED)
    

    

def main(trial_num, rANDOM_SEED, rng,sim_type, zERO_ANGLES,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,xyz_gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
             r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,mIN_PERCENT_MANIP_INCREASE,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT,object_reduction,contact_sample_chance, save = False):
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable GUI elements
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # Ensure rendering is enabled
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # Optional
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # Optional
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)  # Ensure RGB rendering is enabled

    add_data_path(pybullet_data.getDataPath())

    print(f"Running trial {trial_num} with sim type {sim_type}, min iterations {rRT_MIN_ITERATIONS}, contact sample prob {contact_sample_chance}")

    # draw_pose(Pose(), length=1.)
    xyz_quat_goal = None
    xyz_limits = None
    reference_quat = None
    yzx_euler_angle_limits = None
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(TACTILE_KINOVA_URDF, [0, 0, 0],useFixedBase=1)
    # m = pin.buildSampleModelManipulator()
    pin_model = pin.buildModelFromUrdf(TACTILE_KINOVA_URDF_FOR_PIN)
    
    # viz_mujoco()
    # viz_mujoco_collision_bodies()
            
    if object_reduction > 0.0 and sim_type != 3 and sim_type != 5 and sim_type != 6:
        print(f"setting for reducing objects in simulation for sim type {sim_type} not configured yet")
        wait_for_user()
        
    obstacles = []
    if sim_type == 0:
        #2D
        set_camera_pose(camera_point=[0, -1, 0.7], target_point = [0, 0, 0.7])
        box_position = [-0.1, 0, 1]
        box_dims = [0.1,0.1,0.2]
        goal = [0,-1.1,0,-0.1,0,-0.2,0]
        start = [0,1.1,0,0.1,0,0.2,0]
        col_box_id = create_box(box_position,box_dims)
        obstacles = [col_box_id]
    if sim_type == 1 or sim_type == 2:
        #initial 3D
        set_camera_pose(camera_point=[-1.2, 0.9, 0.7], target_point = [0, 0, 0.5])
        box1_position = [-0.7, 0.1, 0.7]
        box1_dims = [0.05,0.05,0.1]
        box2_position = [0.7, 0.1, 0.7]
        box2_dims = [0.05,0.05,0.1]
        box3_position = [0, 0, 1]
        box3_dims = [1,1,0.05]
        goal = [1.7,-1,0.5,0.3,-0.2,0.2,-0.2]
        start = [-1.7,-1,0.2,-0.1,0.3,0.2,0.1]
        if sim_type == 2:
            col_box_id1 = create_box(box1_position,box1_dims)
            col_box_id2 = create_box(box2_position,box2_dims)
            col_box_id3 = create_box(box3_position,box3_dims)
            obstacles = [col_box_id1,col_box_id2,col_box_id3]
    if sim_type == 3:
        #human laying down
        set_camera_pose(camera_point=[-1.2, -0.4, 1.2], target_point = [-0.45, 0, 0.3])
        
        
        box1_position = [-0.35, 0., 0.03]
        box1_dims = [0.28-object_reduction,1-object_reduction,0.04-object_reduction]
        col_box_id1 = create_box(box1_position,box1_dims)
        cyl_position1 = (-0.35,0,0.08)
        cyl_quat1 = p.getQuaternionFromEuler([3.14/2, 0, 0])
        rad1 = 0.17 - object_reduction
        h1 = 1 - 2*object_reduction
        col_cyl_id1 = create_cylinder(rad1, h1, cyl_position1, cyl_quat1)
        p.changeVisualShape(col_cyl_id1, -1, rgbaColor=[0.2,0.2,0.2,1])
        arm_dif = 0.3
        cyl_position2 = (-0.5+arm_dif,0,0.12)
        cyl_quat2 = p.getQuaternionFromEuler([3.14/2, 0, 3.14/4])
        cyl_position3 = (-0.5-arm_dif,0,0.12)
        cyl_quat3 = p.getQuaternionFromEuler([3.14/2, 0, -3.14/4])
        rad2 = 0.05
        h2 = 0.3
        # col_cyl_id2 = create_cylinder(rad2, h2, cyl_position2, cyl_quat2)
        # col_cyl_id3 = create_cylinder(rad2, h2, cyl_position3, cyl_quat3)
        obstacles = [col_cyl_id1,col_box_id1]

        # goal = [0.4,-1.4,0.2,-0.5,0.3,-1,0]
        # start = [-2.2,-1.7,1.3,-1.7,-0.1,-1,0]

        start = [0.4,-1.45,1,-0.4,-0.1,-1.7,0.5]
        goal = [0.85,0.9,-1,0.5,-0.5,1.2,0]

        # goal = [-0.59131794, -1.41156384,  0.64389399, -0.35700311, -0.85622942,-2.23725663, -1.95666927]
        g_pos = [-0.5,  -0.35,  0.19]
        g_quat = [0.9239,0.0,0.3827,0.0]
        xyz_quat_goal = [np.array(g_pos),np.array(g_quat)]
        d = 0.1
        xyz_limits = [[-0.5-d*2,  -0.43-d,  max(0,0.19-d*3)],[-0.5+d*2,  0.43+d,  0.19+d*3]]
        y_ref = np.pi/2+np.pi/5
        alpha = 0.0000001
        
        #TODO: get working 
        reference_quat = np.array(euler_yzx_to_quaternion([y_ref,0,0]))
        yzx_euler_angle_limits = [[y_ref-alpha,-np.pi,-(ANGLE_DIF_ALLOWED-math.radians(1))],[y_ref+alpha,np.pi,ANGLE_DIF_ALLOWED-math.radians(1)]]
        # yzx_euler_angle_limits = [[y_ref-alpha,-alpha,-alpha],[y_ref+alpha,alpha,alpha]]

        
    if sim_type == 4:
        #human in wheelchair
        set_camera_pose(camera_point=[-0.3, 0.3, 1.1], target_point = [0.4, 0, 0.7])
        box1_position = [0.5, 0, 0.15]
        box1_dims = [0.25,0.3,0.25]
        col_box_id1 = create_box(box1_position,box1_dims)
        cyl_position1 = (0.5, -0.1, 0.5)
        cyl_quat1 = p.getQuaternionFromEuler([0, 0, 0])
        rad1 = 0.17
        h1 = 1.3
        col_cyl_id1 = create_cylinder(rad1, h1, cyl_position1, cyl_quat1)
        obstacles = [col_box_id1,col_cyl_id1]

        # goal = [0.4,-1.4,0.2,-0.5,0.3,-1,0]
        # start = [-2.2,-1.7,1.3,-1.7,-0.1,-1,0]
        goal = [0.85,0.9,-1,0.5,-0.5,1.2,0]
        start = [-0.6,0.8,1,0.6,0.5,0.5,0]
        
    if sim_type == 5:
        #reach over body
        set_camera_pose(camera_point=[0.9, 0.2, 1], target_point = [0.35, -0.2, 0.13])
        box1_position = [0.35, -0.3, 0.13]
        box1_dims = [0.26-object_reduction,1-object_reduction,0.14-object_reduction]
        col_box_id1 = create_box(box1_position,box1_dims)
        cyl_position1 = (0.35,-0.3,0.3)
        cyl_quat1 = p.getQuaternionFromEuler([3.14/2, 0, 0])
        rad1 = 0.18 - object_reduction
        h1 = 1 - 2*object_reduction
        col_cyl_id1 = create_cylinder(rad1, h1, cyl_position1, cyl_quat1)
        p.changeVisualShape(col_cyl_id1, -1, rgbaColor=[0.2,0.2,0.2,1])
        
        obstacles = [col_cyl_id1,col_box_id1]
        
        #MUJOCO obstacle setup
        box_inputs = [{'pos': box1_position, 'dims': box1_dims}]
        cylinder_inputs = [{'pos': cyl_position1, 'radius': rad1, 'height': h1, 'quat': cyl_quat1}]
        mj_obstacle_list = build_mujoco_obstacle_list(box_inputs, cylinder_inputs)
    
        # goal = [0.4,-1.4,0.2,-0.5,0.3,-1,0]
        # start = [-2.2,-1.7,1.3,-1.7,-0.1,-1,0]
        start = [-0.3,0.3,0.4,1.2,0.5,1.2,0]
        # goal = [0.85,0.9,-1,0.5,-0.5,1.2,0]

        # goal = [-0.59131794, -1.41156384,  0.64389399, -0.35700311, -0.85622942,-2.23725663, -1.95666927]
        g_pos = [0.60, -0.42,  0.41]
        g_quat = [ 0.35454866,  0.8174026,  -0.00976152,  0.45393062]
        xyz_quat_goal = [np.array(g_pos),np.array(g_quat)]
        
        xyz_limits = [[0.3,  -0.7,  0.28],[0.8,  0,  0.7]]
        yzx_euler_angle_limits = [[-np.pi,-np.pi,-np.pi],[np.pi,np.pi,np.pi]]
        
    if sim_type == 6:
        #goign through cylinders
        set_camera_pose(camera_point=[-0.3, 0.3, 1.1], target_point = [0.4, 0, 0.7])
        # box1_position = [0.5, 0, 0.15]
        # box1_dims = [0.25,0.3,0.25]
        # col_box_id1 = create_box(box1_position,box1_dims)
        cyl_position1 = (0.45, -0.09, 0.5)
        cyl_quat1 = p.getQuaternionFromEuler([0, 0, 0])
        rad1 = 0.1 - object_reduction
        h1 = 1- 2*object_reduction
        col_cyl_id1 = create_cylinder(rad1, h1, cyl_position1, cyl_quat1)
        cyl_position2 = (0.22, 0.2, 0.5)
        cyl_quat2 = p.getQuaternionFromEuler([0, 0, 0])
        rad2 = 0.1 - object_reduction
        h2 = 1- 2*object_reduction
        col_cyl_id2 = create_cylinder(rad2, h2, cyl_position2, cyl_quat2)
        cyl_position3 = (0.15, -0.19, 0.5)
        cyl_quat3 = p.getQuaternionFromEuler([0, 0, 0])
        rad3 = 0.1 - object_reduction
        h3 = 1- 2*object_reduction
        col_cyl_id3 = create_cylinder(rad3, h3, cyl_position3, cyl_quat3)
        obstacles = [col_cyl_id1,col_cyl_id2,col_cyl_id3]
        
        #MUJOCO obstacle setup
        box_inputs = []
        cylinder_inputs = [{'pos': cyl_position1, 'radius': rad1, 'height': h1, 'quat': cyl_quat1},
                           {'pos': cyl_position2, 'radius': rad2, 'height': h2, 'quat': cyl_quat2},
                           {'pos': cyl_position3, 'radius': rad3, 'height': h3, 'quat': cyl_quat3}]
        mj_obstacle_list = build_mujoco_obstacle_list(box_inputs, cylinder_inputs)
        # goal = [0.4,-1.4,0.2,-0.5,0.3,-1,0]
        # start = [-2.2,-1.7,1.3,-1.7,-0.1,-1,0]
        # goal = [0.85,0.8,-1,0.5,-0.5,0.5,0]
        start = [0,-0.7,0,1.5,0,1.5,0]
        g_pos = [0.45, 0.25, 0.5]
        g_quat = [ 0.35454866,  0.8174026,  -0.00976152,  0.45393062]
        xyz_quat_goal = [np.array(g_pos),np.array(g_quat)]
        xyz_limits = [[-0.1,  -0.3,  0.25],[0.7, 0.53, 0.85]]
        yzx_euler_angle_limits = [[-np.pi,-np.pi,-np.pi],[np.pi,np.pi,np.pi]]

    # col_box_id = create_box(box_position,box_dims)
    # obstacles = [col_box_id]
    
    # taxel_ids = get_taxels(robot_id)



    # dump_body(robot_id)
    # print('Start?')
    # wait_for_user()

    joint_indices = [i for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
    
    #MinkIKSolver setup
    # mj_obstacle_list = []
    new_mjcf_path, obstacle_name_list = inject_obstacles(TACTILE_KINOVA_URDF_FOR_MUJOCO,mj_obstacle_list)
    ik_solver = MinkIKSolver(new_mjcf_path, obstacle_name_list)    
    
    if xyz_quat_goal is None:
        q_goal = goal
        set_joint_state(robot_id, q_goal, joint_indices)
        if VIZ:
            time.sleep(0.2)
        goal_xyz,goal_quat = q_to_endeff(robot_id, q_goal)
        xyz_quat_goal = [goal_xyz,goal_quat]
    print("goal xyz quat")
    print(xyz_quat_goal)    
    dot(xyz_quat_goal[0], [0.5,0.9,0.5,1])
    
    q_init = start
    set_joint_state(robot_id, q_init, joint_indices)
    start_pos, start_orientation = q_to_endeff(robot_id, q_init)
    xyz_quat_start = [start_pos, start_orientation]
    print(xyz_quat_start) 
    dot(xyz_quat_start[0], [0.8,0.8,0.8,1])
    if VIZ:
        time.sleep(0.2)
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
    if xyz_limits is None:
        xyz_limits = [[-1,-1,0],[0.1,1,0.8]]
        # yzx_euler_angle_limits = [[-np.pi,-np.pi,-np.pi],[np.pi,np.pi,np.pi]]
        # reference_quat = np.array(p.getQuaternionFromEuler([0,0,0]))
    rrt_obj = RRT_BASE(rng, robot_id, ik_solver,pin_model, joint_indices, 
                 obstacles, eXTEND_STEP_SIZE, q_init, xyz_quat_goal, xyz_quat_start, gOAL_SAMPLE_PROB, gOAL_AREA_SAMPLE_PROB, 
                 gOAL_AREA_DELTA, rRT_MAX_ITERATIONS, rRT_MIN_ITERATIONS, eDGE_COLLISION_RESOLUTION, PAUSE_TIME, 
                 wEIGHT_FOR_DISTANCE, xyz_gOAL_TOLERANCE, r_FOR_PARENT, zERO_ANGLES,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT, 
                 mIN_PERCENT_MANIP_INCREASE, xyz_limits, yzx_euler_angle_limits, reference_quat, object_reduction,contact_sample_chance, max_time=INF)
    start_time = time.time()
    node_path = rrt_obj.rrt_star_search()
    print("Time taken for RRT search: " + str(elapsed_time(start_time)))
    if VIZ:
        wait_for_user()
    path = configs(node_path)
    print("done with trial num: " + str(trial_num))
    if path:
        # path = smooth(robot_id, path, joint_indices, obstacles)
        print("Found path:")
        print_path(path)
        set_joint_state(robot_id, q_init)
        if VIZ:
            print_path_stuff(node_path)
            rrt_obj.visualize_path(path, line_color=[1, 0, 0])
        else:
            rrt_obj.dot_path(node_path)
    else:
        print("No path found")

    print('Done?')
    if VIZ:
        wait_for_user()
    # save_screenshot(trial_num, node_path, rANDOM_SEED, sIM_2D,zERO_ANGLES,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
    #          r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,mIN_PERCENT_MANIP_INCREASE,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT)
    if save:
        save_data (trial_num, node_path, rANDOM_SEED, sim_type,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,xyz_gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
                r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,object_reduction,contact_sample_chance)
    disconnect()

def print_path_stuff(node_path):
    i = 1
    for node in node_path:
        rounded_distances = [round(dist, 4) for dist in node.lowest_signed_distances]
        print(f"node {i} lsd: {str(rounded_distances)}, cumulative manip cost: {str(round(node.manip_cost,4))}")
        i += 1

class RRT_BASE(object):
    def __init__(self, rng, robot_id, ik_solver, pin_model, joint_indices, 
                 obstacles, extend_step, q_init, xyz_quat_goal, xyz_quat_start, agressive_goal_sampling_prob, goal_area_probability,  
                 goal_area_delta, max_samples, min_samples, res, pause_time, weight_dist, goal_tolerance,
                 R, zERO_ANGLES,lAZY,uPDATE_PATH,oLD_COST,iNCREASE_WEIGHT, mIN_PERCENT_MANIP_INCREASE, xyz_limits, yzx_euler_angle_limits, reference_quat, 
                 object_reduction,contact_sample_chance, classic_rrt = False, max_time=INF, prc=0.01):
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
        self.pin_model = pin_model
        self.pin_model_data = pin_model.createData()
        self.R = R #radius for parent finding
        self.joint_indices = joint_indices
        self.goal_tolerance = goal_tolerance
        self.agressive_goal_sampling_prob = agressive_goal_sampling_prob
        self.goal_area_probability = goal_area_probability
        self.goal_area_delta = goal_area_delta
        self.obstacles = obstacles
        self.samples_taken = 0
        self.res = res
        self.prc = prc
        self.object_reduction = object_reduction
        self.contact_sample_chance = contact_sample_chance
        self.q_init = np.array(q_init)
        self.xyz_quat_goal = xyz_quat_goal
        self.xyz_quat_start = xyz_quat_start
        [self.upper_limits,self.lower_limits] = self.get_joint_limits()
        self.xyz_limits = xyz_limits
        self.reference_quat = reference_quat
        self.yzx_euler_angle_limits = yzx_euler_angle_limits
        manipulabilities, lowest_signed_distances,closest_taxel_ids = self.calculate_taxel_manip_and_dist(self.q_init)
        self.nodes = [TreeNode(self.xyz_quat_start, self.q_init, manipulabilities=manipulabilities,lowest_signed_distances=lowest_signed_distances, closest_taxel_ids=closest_taxel_ids)]
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
        self.classic_rrt = classic_rrt
        self.ik_solver =ik_solver


    

    def rrt_star_search(self):
        start_time = time.time()
        # self.test_mink_IK()
        # wait_for_user()
        draw_xyz_bounding_box(self.xyz_limits, color=[1, 0, 0], line_width=1, life_time=0)
        if VIZ:
            wait_for_user()
        # pos = [0.4,  0.4,  0.2]
        # target_taxel_xyz_quat = [np.array(pos), np.array([0.9239,0,0.3827,0])]
        # target_end_eff_xyz_quat = [np.array([0.7,  0.1,  0.3]), np.array([0.9239,0,0.3827,0])]
        # taxel_link_id = 13
        # initial_q = self.q_init

        # q = self.taxel_IK(target_taxel_xyz_quat, target_end_eff_xyz_quat, taxel_link_id, initial_q)
        # dot(pos)
        # set_joint_state(self.robot_id, q, self.joint_indices)
        # p.addUserDebugLine([pos[0],-5,pos[2]], [pos[0],5,pos[2]], lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0)
        # p.addUserDebugLine([-5,pos[1],pos[2]], [5,pos[1],pos[2]], lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0)
        # p.addUserDebugLine([pos[0],pos[1],-5], [pos[0],pos[1],5], lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0)
        # wait_for_user()
        
        # print(self.xyz_quat_start)
        # print(self.xyz_quat_goal)
        # # xyz_quat = [np.array([-0.7,  0.1,  0.3]), np.array([0.9239,0,0.3827,0])]
        # xyz = np.array([0.2,  0.1,  0.3])
        # xyz_quat = [xyz, self.reference_quat]
        # # self.display_point_and_vector(quat,xyz)
        # q = self.get_closest_config_from_x(xyz_quat, self.q_init, max_retries=0,perturbation_scale=1)
        # wait_for_user()
        # for i in range(10000):
        #     xyz_quat = [xyz, self.get_random_quat()]
        #     q = self.get_closest_config_from_x(xyz_quat, self.q_init, max_retries=0,perturbation_scale=1)
        #     wait_for_user()
        # # # set_joint_state(self.robot_id, q, self.joint_indices)
        # # # print(q)
        #detect initial collision
        
        if self.collision_fn(self.q_init):# or self.collision_fn(self.xyz_quat_goal):
            print("collision at start")
            return None
        #checks sample count and compares to max sample
        while self.samples_taken < self.max_samples:
            c_sample = False
            if elapsed_time(start_time) >= self.max_time:
                print("max time reached")
                break
            # tries more aggressively directly sampling goal (trying to get a solution) only when enough samples were taken
            if self.samples_taken >= self.min_samples and (self.samples_taken == self.min_samples or self.rng.random() < self.agressive_goal_sampling_prob): 
                target_xyz_quat = self.xyz_quat_goal
                print("sampling goal")
                sampling_goal = True
                # dot(q_to_endeff(self.robot_id, target_config), [1,0,0,1]) anbd # visualizing sampled point in red             
            else:
                target_xyz_quat = self.sample_goal_area() if self.rng.random() < self.goal_area_probability else self.sample_fn() #sample goal a certain prob
                sampling_goal = False
            # dot(q_to_endeff(self.robot_id, target_config), [1,0,0,1]) # visualizing sampled point in red
            self.samples_taken = self.samples_taken + 1
            # print(f"sampled target xyz: {target_xyz_quat}")
            closest_node = self.get_closest_node(target_xyz_quat) # getting nearest node
            closest_q = closest_node.config
            # print(f"found closest node: {closest_node}")
            if sampling_goal:
                set_joint_state(self.robot_id, closest_node.config, self.joint_indices)
                if self.samples_taken == self.min_samples+1:
                    print("sample min reached")
                    print(closest_q)
                    # self.visualize_path(configs(closest_node.retrace()), line_color=[1, 0, 0])
                if VIZ:
                    current_pos, current_orientation = q_to_endeff(self.robot_id, closest_q)
                    print(f"Position: {current_pos}, Orientation: {current_orientation}")
                    wait_for_user()
                if self.goal_test(closest_node.xyz_quat):
                    print("FOUND PATH!")
                    print("Goal conditions met by closest node.")
                    return closest_node.retrace()
            if closest_node.c_sample == True or self.rng.random() > self.contact_sample_chance:
                extended_target_xyz_quat = self.extend_fn(closest_node, target_xyz_quat) # extending from nearest node
                if self.check_if_xyz_in_obstacle(extended_target_xyz_quat[0]):
                    q = None #if the xyz sample is in an obstacle don't use 
                else:
                    random_start = self.rng.random() < RANDOM_IK_CHANCE #chance to use random IK initialization
                    q = self.mink_collision_ik(
                        target_pos=target_xyz_quat[0],
                        target_quat=None,
                        initial_guess=closest_q,
                        taxel_target = None,
                        random_start = random_start
                        )
                    # q = self.ee_IK(extended_target_xyz_quat,closest_node.config,random_start = random_start)
            else:
                c_sample = True
                closest_node.c_sample = True
                closest_idx = np.argmin(closest_node.lowest_signed_distances)
                closest_taxel_ids = closest_node.closest_taxel_ids
                closest_taxel_id = closest_taxel_ids[closest_idx]
                contact_target_xyz_quat, line_id = self.contact_sample(closest_node,closest_taxel_id)
                if contact_target_xyz_quat is None or closest_taxel_id < 7:
                    q = None
                else:
                    print("contact IK start")
                    if VIZ:
                        set_joint_state(self.robot_id, closest_q, self.joint_indices)
                        wait_for_user()
                    q = self.mink_collision_ik(
                        target_pos=contact_target_xyz_quat[0],
                        target_quat=None,
                        initial_guess=closest_q,
                        taxel_target = closest_taxel_id-9,
                        random_start = False
                        )
                    # q = self.taxel_IK(contact_target_xyz_quat, self.xyz_quat_goal, closest_taxel_id, closest_node.config)
                    print("contact IK done")
                    if q is not None:
                        print("contact IK SUCCESS")
                        if VIZ:
                            wait_for_user()
                    else:
                        p.removeUserDebugItem(line_id)
                        print("contact IK FAIL")
                        if VIZ:
                            wait_for_user()
            if q is not None:
                set_joint_state(self.robot_id, q, self.joint_indices)
                manips, lowest_signed_distances,closest_taxel_ids = self.calculate_taxel_manip_and_dist(q)
                if all(lowest_signed_distances) >= -MAX_PENETRATION:
                    if self.classic_rrt:
                        new_parent = closest_node
                    else:
                        new_parent = None
                    [_,node_path] = self.create_node(extended_target_xyz_quat,q,manips, lowest_signed_distances,closest_taxel_ids, new_parent = new_parent, c_sample = c_sample) # TODO maybe dont compute new parent
                    if self.samples_taken > self.min_samples and node_path is not None:
                        print("FOUND PATH!")
                        # if self.will_update_path:
                        #     node_path = self.update_path(node_path) #computes new parents and costs starting from goal
                        return node_path
            print(f"finished loop\n")
            
        return None
    
    def print_all_node_xyzquats(self):
        for i,node in enumerate(self.nodes):
            print(f"Node {i}: {node.xyz_quat}\n")

    def print_all_nodes(self):
        for i,node in enumerate(self.nodes):
            print(f"Node {i}: {node}\n")
    

    
    def test_mink_IK(self):
        time.sleep(0.5)
        print("\nRunning Mink IK test...")

        # Reset robot to start pose
        set_joint_state(self.robot_id, self.q_init, self.joint_indices)
        p.stepSimulation()
        # time.sleep(0.5)
        
        # Draw target goal as green sphere
        target_pos = self.xyz_quat_goal[0]
        target_quat = self.xyz_quat_goal[1]
        initial_guess = self.q_init
        # initial_guess = np.array([-0.6,  1.5,  2.5, 1.27,  0.5,  0.84, -0.57])
        set_joint_state(self.robot_id, initial_guess, self.joint_indices)
        print(q_to_endeff(self.robot_id, initial_guess))  # Ensure initial guess is valid
        
        target_quat = None
        # target_pos[0] += -0.1
        # target_pos[2] += 0.1
        # target_pos = [0.5,-0.1,0.6]
        dot(target_pos, [1, 0, 0, 1])  # red dot at goal
        wait_for_user()
        # Small pause to visualize starting state and target
        # time.sleep(1)

        # Call Mink IK
        q_solution = self.mink_collision_ik(
            target_pos=target_pos,
            target_quat=target_quat,
            initial_guess=initial_guess,
            taxel_target = -1,
            random_start = True
            )
        
        
        
        # Visualize result
        if q_solution is not None:
            print("Mink IK succeeded!")
            # Print the solution's end-effector position and orientation
            ee_position, ee_orientation = q_to_endeff(self.robot_id, q_solution)
            print(f"End-effector position: {ee_position}")
            print(f"End-effector orientation (quaternion): {ee_orientation}")

            for i in range(50):  # Smooth move to target
                interp_q = (1 - i / 50) * np.array(self.q_init) + (i / 50) * np.array(q_solution)
                set_joint_state(self.robot_id, interp_q, self.joint_indices)
                p.stepSimulation()
                time.sleep(0.02)
        else:
            print("Mink IK failed to find solution.")
            
        wait_for_user()


    def get_config_from_x(self, xyz_quat): #using pybullet IK
        if xyz_quat[0][2] < 0: 
            print(f"xyz_quat: {xyz_quat}")
        q = p.calculateInverseKinematics(self.robot_id, 7, tuple(xyz_quat[0]))
        if q is None: return None
        return np.array(q)
    
    def mink_collision_ik(self, target_pos, target_quat=None, 
                          initial_guess=None, taxel_target = None, 
                          max_attempts=1,iterations =50,random_start = False, viz = False):
        """
        Compute collision-free IK using Mink.

        Args:
            target_pos: list or np.array, [x, y, z] end-effector position
            target_quat: list or np.array, [x, y, z, w] quaternion (optional)
            initial_guess: np.array of joint angles (optional)
            max_attempts: int, number of retries with random initial guesses
            taxel_target: int, taxel ID to target, otherwise end effector(optional)

        Returns:
            np.array of joint angles if solution found, otherwise None
        """

        if initial_guess is None:
            initial_guess = deepcopy(self.q_init)

        j_delta = 1
        if random_start:  
            initial_guess = initial_guess + self.rng.normal(scale=j_delta, size=initial_guess.shape)
            initial_guess = np.clip(initial_guess, self.lower_limits, self.upper_limits)

        guess = initial_guess
        
        for attempt in range(max_attempts):
            print(f"Mink IK attempt {attempt+1}/{max_attempts}")
            
            solution = self.ik_solver.solve(
                goal_pos=target_pos,
                goal_quat=target_quat,
                initial_qpos=guess,
                verbose=False,
                site_target = taxel_target,
                iterations = iterations,
                viz = viz
            )

            if solution is None:
                print("Mink failed to find collison-free IK solution.")
                guess = initial_guess + self.rng.normal(scale=j_delta, size=guess.shape)
                guess = np.clip(guess, self.lower_limits, self.upper_limits)
                # return None
            else:
                print("Found collision-free Mink IK solution.")
                return solution

        print("Failed to find collision-free Mink IK after retries.")
        return None

    
    # def pin_ee_IK(self, target_xyz_quat, initial_q, random_start = True, max_retries=DEFAULT_IK_RETRIES,perturbation_scale=JS_EXTEND_MAX*2):
    #     """
    #     Computes the closest joint configuration using inverse kinematics, retrying if the solution is in collision.
        
    #     Args:
    #         target_xyz_quat: Desired end-effector position and orientation.
    #         initial_q: Initial joint configuration.
    #         max_retries: Maximum number of retries with different initial guesses.
        
    #     Returns:
    #         np.array: A valid joint configuration or None if no valid solution is found.
    #     """
    #     # print(f"running IK with retries: {max_retries}")
    #     q_attempt = initial_q
    #     if random_start:  
    #         q_attempt = self.rng.uniform(self.lower_limits, self.upper_limits)
    #     for tryIK in range(max_retries+1):
    #         q_eigen = pin.utils.zero(len(q_attempt))
    #         q_eigen[0:len(q_attempt)] = q_attempt  

    #         target_pos = target_xyz_quat[0]
    #         target_quat = target_xyz_quat[1]
    #         model = self.pin_model
    #         data = self.pin_model_data
    #         max_iters = 200
    #         tolerance = 1e-3
    #         alpha = 0.1  
    #         ee_frame_id = model.getFrameId("EndEffector")

    #         for i in range(max_iters):
    #             # Forward Kinematics
    #             pin.forwardKinematics(model, data, q_eigen)
    #             pin.updateFramePlacements(model, data)
    #             # Compute position error
                
    #             ee_pos = np.array(data.oMf[ee_frame_id].translation)
    #             pos_error = target_pos - ee_pos

    #             # Compute orientation error using logarithm map
                
    #             ee_rot = data.oMf[ee_frame_id].rotation
    #             target_rot = pin.Quaternion(target_quat).matrix()
    #             rot_error_matrix = target_rot @ ee_rot.T
    #             rot_error = pin.log3(rot_error_matrix)  # Rotation vector (axis-angle representation)
                
    #             # Concatenate position and orientation errors
    #             error = np.concatenate((pos_error, rot_error))
                
    #             # Compute Jacobian
    #             J = pin.computeFrameJacobian(model, data, q_eigen, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)

    #             J_position = J[:3, :]
    #             J_orientation = J[3:, :]
    #             J_full = np.vstack((J_position, J_orientation))

    #             # Solve for q update using pseudo-inverse
    #             q_update = alpha * np.linalg.pinv(J_full) @ error
    #             q_eigen += q_update
                

    #             # Check for convergence
    #             if np.linalg.norm(error) < tolerance:
    #                 q = np.array(q_eigen)

    #                 # Collision check
    #                 # print(f"IK collision check for try {tryIK+1}")
    #                 if not self.collision_fn(q):
    #                     print(f"IK Converged in {i+1} iterations (Try {tryIK+1}).")
    #                     return q
    #                 else:
    #                     break

    #         # Modify initial guess and retry
    #         q_attempt = initial_q+self.rng.uniform(-perturbation_scale, perturbation_scale, size=initial_q.shape)
    #         q_attempt = np.clip(initial_q, self.lower_limits, self.upper_limits)  # Ensure within joint limits

    #     print("IK failed to find a collision-free solution.")
    #     return None
    
    
    # def pin_taxel_IK(self, target_taxel_xyz_quat, target_end_eff_xyz_quat, taxel_link_id, initial_q, max_retries=DEFAULT_CONTACT_IK_RETRIES,perturbation_scale=JS_EXTEND_MAX*10):
    #         """
    #         Computes the closest joint configuration using inverse kinematics for a taxel contact location, retrying if the solution is in collision.
            
    #         Args:
    #             target_taxel_xyz_quat: Desired taxel position and orientation.
    #             target_end_eff_xyz_quat: Desired end-effector position and orientation (solved in null space).
    #             taxel_link_id: Link ID of the taxel.
    #             initial_q: Initial joint configuration.
    #             max_retries: Maximum number of retries with different initial guesses.
            
    #         Returns:
    #             np.array: A valid joint configuration or None if no valid solution is found.
    #         """
            
    #         if taxel_link_id < 7 or taxel_link_id > 9+28:
    #             return None
            
    #         # print(f"running IK with retries: {max_retries}")
    #         model = self.pin_model
    #         data = self.pin_model_data
    #         max_iters = 200
    #         pos_tolerance = 1e-3
    #         rot_tolerance = 1e-2
    #         alpha = 0.1 

    #         # for taxel_link_id in range(9+14, 9+28):
    #         joint_info = p.getJointInfo(self.robot_id, taxel_link_id)
    #         link_name = joint_info[12].decode("utf-8")
    #         print(f"running IK for link: {link_name}")
    #         taxel_frame_id = model.getFrameId(link_name)

    #         q_eigen = pin.utils.zero(len(initial_q))
    #         q_eigen[0:len(initial_q)] = initial_q  
    #         pin.forwardKinematics(model, data, q_eigen)
    #         pin.updateFramePlacements(model, data)
    #         frame_placement = data.oMf[taxel_frame_id]
    #         taxel_frame_placement = frame_placement
            
    #         # #DEBUG VIZ
    #         # frame_position = taxel_frame_placement.translation
    #         # frame_rotation = taxel_frame_placement.rotation 
    #         # p.addUserDebugLine([frame_position[0],-5,frame_position[2]], [frame_position[0],5,frame_position[2]], lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=0)
    #         # p.addUserDebugLine([-5,frame_position[1],frame_position[2]], [5,frame_position[1],frame_position[2]], lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=0)
    #         # p.addUserDebugLine([frame_position[0],frame_position[1],-5], [frame_position[0],frame_position[1],5], lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=0)
    #         # local_normal = np.array([1, 0, 0])
    #         # world_normal = frame_rotation @ local_normal  # rotate normal into world frame
    #         # line_length = 5  # adjust as needed
    #         # end_point = frame_position + line_length * world_normal
    #         # p.addUserDebugLine(
    #         #     frame_position.tolist(),
    #         #     end_point.tolist(),
    #         #     lineColorRGB=[0, 0, 1],
    #         #     lineWidth=2,
    #         #     lifeTime=0
    #         # )
    #         # # time.sleep(2)
    #         # wait_for_user()

    #         target_taxel_pos = target_taxel_xyz_quat[0]
    #         target_taxel_quat = target_taxel_xyz_quat[1]
            
            
    #         q_attempt = initial_q
    #         for tryIK in range(max_retries+1):
    #             print(f"IK attempt {tryIK+1} of {max_retries+1}")
    #             q_eigen = pin.utils.zero(len(q_attempt))
    #             q_eigen[0:len(q_attempt)] = q_attempt  

    #             for i in range(max_iters):
    #                 # Forward Kinematics
    #                 pin.forwardKinematics(model, data, q_eigen)
    #                 pin.updateFramePlacements(model, data)
    #                 taxel_pos = np.array(taxel_frame_placement.translation)
    #                 pos_error = target_taxel_pos - taxel_pos

    #                 # taxel_rot = taxel_frame_placement.rotation
    #                 # target_rot = pin.Quaternion(target_taxel_quat).matrix()
    #                 # rot_error_matrix = target_rot @ taxel_rot.T
    #                 # rot_error = pin.log3(rot_error_matrix)  # Rotation vector (axis-angle representation)
    #                 # error = np.concatenate((pos_error, rot_error))
    #                 # error = np.concatenate((pos_error, [0, 0, 0]))  # Assuming no orientation error for now]))
    #                 # J, _ = p.calculateJacobian(self.robot_id, taxel_link_id, [0, 0, 0], q_attempt.tolist(), [0]*7, [0]*7)
    #                 # J_orientation = J[3:, :]
    #                 # J_full = np.vstack((J_position, J_orientation))
                    
    #                 error = pos_error # not dealing with orientation error for now
    #                 J = pin.computeFrameJacobian(model, data, q_eigen, taxel_frame_id, pin.LOCAL_WORLD_ALIGNED)
    #                 J_position = J[:3, :]
    #                 q_update = alpha * np.linalg.pinv(J_position) @ error
    #                 q_eigen += q_update
    #                 # Check for convergence
    #                 if np.linalg.norm(pos_error) < pos_tolerance: #and np.linalg.norm(rot_error) < rot_tolerance:
    #                     q = np.array(q_eigen)
    #                     # print(f"IK collision check for try {tryIK+1}")
    #                     if not self.collision_fn(q):
    #                         print(f"IK Converged in {i+1} iterations (Try {tryIK+1}).")
    #                         # print(q)
    #                         return q
    #                     else:
    #                         break

    #             # Modify initial guess and retry
    #             q_attempt = initial_q+self.rng.uniform(-perturbation_scale, perturbation_scale, size=initial_q.shape)
    #             q_attempt = np.clip(initial_q, self.lower_limits, self.upper_limits)  # Ensure within joint limits

    #         print("IK failed to find a collision-free solution.")
    #         return None
    #         return np.array(q_eigen)
        
    def create_node(self, xyz_quat, q, manips, lowest_signed_distances,closest_taxel_ids, c_sample = False, new_parent = None):
            same_node_idx, new_node = self.check_for_same_node(q)
            if same_node_idx < 0:
                new_node = TreeNode(xyz_quat, q, manipulabilities=manips,lowest_signed_distances=lowest_signed_distances, closest_taxel_ids=closest_taxel_ids, c_sample = c_sample)
                set_joint_state(self.robot_id, q, self.joint_indices)
                if VIZ:
                    closest_idx = np.argmin(lowest_signed_distances)
                    dot_id = gradient_dot(self.robot_id, manips[closest_idx])
                else:
                    dot_id = None
                # dot(xyz_quat[0], [0.9,0,0,1])
                print(f"created new tree node: {new_node}")
                # wait_for_user()
                prev_node = False
            else: #removes and readds node if it already exists
                print(f"node already exists at: {same_node_idx}")
                new_node = self.nodes[same_node_idx]
                prev_node = True
                new_parent = None
                dot_id = None
            if new_parent is None:
                new_parent = self.find_parent(new_node,self.threshold)
                if new_parent is None:
                    if dot_id is not None:
                        p.removeBody(dot_id)
                    return None, None
            new_node.parent = new_parent
            new_node.num_in_path = new_parent.num_in_path + 1
            [new_node.total_cost, new_node.d_cost, new_node.dist_to_last, new_node.manip_cost, new_node.node_manip_cost] = self.cost_fn(new_parent,new_node)
            if prev_node == False: 
                self.nodes.append(new_node)
                print(f"NODE NUMBER: {len(self.nodes)}")
                if (c_sample):
                    set_joint_state(self.robot_id, new_node.config, self.joint_indices)
                    print("valid contact sample!")
                    if VIZ:
                        wait_for_user()
                # set_joint_state(self.robot_id, new_node.config, self.joint_indices)
                # time.sleep(PAUSE_TIME)
            else:
                self.nodes[same_node_idx] = new_node
            if self.goal_test(new_node.xyz_quat):
                return new_node, new_node.retrace()
            return new_node, None
    
    def check_for_same_node(self, q): #returns index of node in self.nodes if it already exists for given q
        for i, node in enumerate(self.nodes):  
            if np.allclose(node.config, np.array(q), atol=1e-3):  
                return i,node   # Return the index of the matching node  
        return -1, None

    def find_parent(self, new_node, threshold):
        print(f"finding parent")
        # self.print_all_nodes()
        nodes_in_radius = [
            node for node in self.nodes 
            if not np.allclose(new_node.config, node.config, atol=1e-3)
            and self.euclid_distance_fn(new_node.xyz_quat, node.xyz_quat) < self.R
        ]
        # for i,node in enumerate(nodes_in_radius):
        #     print(f"Radius Node {i}: {node}")
        # print(f"New Node: {node}")
        # wait_for_user()
        if nodes_in_radius:
            # lowest_dist_parent = self.argmin(lambda n: self.cost_fn(n, new_node)[1], nodes_in_radius) # lowest dist cost node
            full_cost_parent = self.argmin(lambda n: self.cost_fn(n, new_node)[0], nodes_in_radius) # lowest total cost node
            # lowest_dist_manip_cost = self.cost_fn(lowest_dist_parent, new_node)[3]
            # full_cost_parent_manip_cost = self.cost_fn(full_cost_parent, new_node)[3]
            parent = full_cost_parent
            print(f"found parent")
           
            #interpolating between nodes if config space is greater than step size, creating new ones and checking for collisions
            #also ensuring that the two nodes allow for a path within bounds
            #if interpolation doesn't work (or no nodes in range) returns None to indicate that parent cannot be found
            if self.lazy is False and np.max(np.abs(np.array(parent.config)[:]-np.array(new_node.config)[:])) > JS_EXTEND_MAX:
                print(f"interpolating")
                num_steps = math.floor(np.max(np.abs(np.array(parent.config)[:]-np.array(new_node.config)[:]))/JS_EXTEND_MAX)
                q_list = self.interpolate_configs(parent.config, new_node.config, num_steps = num_steps)
                parent_interp_node = parent
                for i in range(len(q_list)):
                    q_interp = np.array(q_list[i])
                    col = self.collision_fn(q_interp)
                    link_state = p.getLinkState(self.robot_id, 7)
                    xyz_quat =[np.array(link_state[4]), np.array(link_state[5])]
                    manips, lowest_signed_distances,closest_taxel_ids = self.calculate_taxel_manip_and_dist(q_interp)
                    in_xyz_quat_lims = self.check_xyz_quat_lims(xyz_quat)
                    if col or any(lowest_signed_distances) < -MAX_PENETRATION or not in_xyz_quat_lims:
                        print("failed interpolation")
                        return None
                    else:
                        [parent_interp_node,_] = self.create_node(xyz_quat, q_interp, manips, lowest_signed_distances, closest_taxel_ids, new_parent = parent_interp_node)
                        if parent_interp_node is None:
                            print("failed interpolation")
                            return None
                return parent_interp_node
            else:
                return parent
        return None


    def get_closest_node(self,xyz_quat):
        # manipulability, lowest_signed_distance = self.calculate_taxel_manip_and_dist(q)
        # target_node = TreeNode(xyz_quat, manipulability=manipulability,lowest_signed_distance=lowest_signed_distance)
        return self.argmin(lambda n: self.euclid_distance_fn(n.xyz_quat, xyz_quat), self.nodes)



    def sample_fn(self):
        # Generate random xyz_quat combinations within the limits
        random_xyz = np.array([self.rng.uniform(low, high) for low, high in zip(self.xyz_limits[0][:], self.xyz_limits[1][:])])
        random_quat = self.get_random_quat()
        print(f"randomly sampled")
        # wait_for_user()
        return [random_xyz,random_quat]
    
    def contact_sample(self, node, closest_taxel_id):
        '''
        Sample closest point on the surface of objects
        returning None if inside objects or too far away
        returns xyz of point, and the same quat as closest node
        '''
        xyz_quat = node.xyz_quat
        ee_xyz = xyz_quat[0]
        quat = xyz_quat[1]
        
        if closest_taxel_id < 7 or closest_taxel_id > 9+28:
            return None, -1
        
        model = self.pin_model
        data = self.pin_model_data
        joint_info = p.getJointInfo(self.robot_id, closest_taxel_id)
        link_name = joint_info[12].decode("utf-8")
        taxel_frame_id = model.getFrameId(link_name)
        q_eigen = pin.utils.zero(len(node.config))
        q_eigen[0:len(node.config)] = node.config
        pin.forwardKinematics(model, data, q_eigen)
        pin.updateFramePlacements(model, data)
        frame_placement = data.oMf[taxel_frame_id]
        taxel_frame_placement = frame_placement
        tax_xyz = taxel_frame_placement.translation
        tax_rot = taxel_frame_placement.rotation
        
        print(f"contact sampling for {tax_xyz}")
        clearance = self.extend_step/2
        radius = 0.00001  # Small radius for proxy object
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        # Create the body for the visual shape (this places the sphere at the end-effector position)
        proxy_id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex=collision_shape,basePosition=tax_xyz)
        max_dist = self.extend_step*2.5 #TODO choose a good max dist val
        obstacle_id = self.obstacles[0]
        closest_points = p.getClosestPoints(bodyA=proxy_id, bodyB=obstacle_id, distance=max_dist)
        print(closest_points)
        if closest_points:
            closest_point = min(closest_points, key=lambda point: point[8])
            xyz_dist = closest_point[8]  # Distance between the two objects
            # The closest point will be the first one in the list
            closest_xyz = np.array(closest_point[6])  # Point on the obstacle
            # Check if the point is inside the object (based on distance)
            if xyz_dist > clearance:  # If distance is greater than 0 (account for buffer), it's outside the object
                print(f"contact sample found: from {tax_xyz} to {closest_xyz}")
                direction = tax_xyz - closest_xyz
                closest_xyz = closest_xyz + (direction/np.linalg.norm(direction))*clearance
                line_id = p.addUserDebugLine(closest_xyz, tax_xyz, lineColorRGB=[0,1, 0], lineWidth=2, lifeTime=0)
                p.removeBody(proxy_id)
                return [closest_xyz,quat], line_id
        print(f"no contact sample found")
        p.removeBody(proxy_id)
        return None, -1
    
    def get_random_quat(self):
        random_yzx_euler_angles = [self.rng.uniform(low, high) for low, high in zip(self.yzx_euler_angle_limits[0][:], self.yzx_euler_angle_limits[1][:])]
        return np.array(euler_yzx_to_quaternion(random_yzx_euler_angles))
    
    def check_xyz_quat_lims(self, xyz_quat):
        xyz = xyz_quat[0]
        quat = xyz_quat[1]
        for i, (low, high) in enumerate(zip(self.xyz_limits[0], self.xyz_limits[1])):
            if not (low <= xyz[i] <= high):
                return False
        # if abs(angle_between_z_axes(quat, self.reference_quat)) > ANGLE_DIF_ALLOWED:
        #     return False
        # if abs(angle_between_z_axes(quat, self.reference_quat)) > ANGLE_DIF_ALLOWED:
        #     return False
        # TODO FIX THIS
        return True
    
    def sample_goal_area(self):
        direction = self.rng.normal(size=3) 
        direction /= np.linalg.norm(direction)  
        distance = self.rng.uniform(0, self.goal_area_delta)
        new_xyz = self.xyz_quat_goal[0] + direction * distance
        random_quat = self.get_random_quat()
        print(f"goal area sampled")
        # wait_for_user()
        return [new_xyz, random_quat] #TODO maybe restrict quat

    def distance_fn(self, q1, q2):
        # distance between two configurations (joint angles)
        return np.linalg.norm(np.array(q1) - np.array(q2))

    def euclid_distance_fn(self, xyz_quat1, xyz_quat2):
        # distance between two configurations (joint angles)
        xyz_dist = np.linalg.norm(np.array(xyz_quat1[0]) - np.array(xyz_quat2[0]))
        #for now not interpolating quat
        return xyz_dist

    def cost_fn(self, start_node, target_node):
        distance = self.euclid_distance_fn(start_node.xyz_quat, target_node.xyz_quat)
        max_distance = self.euclid_distance_fn((self.xyz_limits[1],[np.pi,np.pi/2,np.pi]), (self.xyz_limits[0],[-np.pi,-np.pi/2,-np.pi]))
        normalized_dist_to_last = distance/max_distance
        parent_dist_cost = start_node.d_cost
        distance_cost = normalized_dist_to_last + parent_dist_cost
        
        lowest_signed_distances = start_node.lowest_signed_distances
        manips = start_node.manipulabilities
        epsilon = 1e-10
        mu_min = 1e-10  
        close_cost_factor1 = 1.5
        closet_cost_factor2 = 4
        
        manip_costs = []
        for i, lowest_signed_distance in enumerate(lowest_signed_distances):
            manip = manips[i]
            mu = max(manip, epsilon)
            scaled_manip_cost = np.log(mu) / np.log(mu_min)
            # maps manips from [0,1] linearly ish, 0 being best manip and thus lowest cost
            scaled_manip_cost = -math.exp(-2 * scaled_manip_cost) + 1
            # maps manips from [0,1] exponentially, so that cost is goes up faster closer when manip is closer to 1
            scaled_manip_cost = np.clip(scaled_manip_cost, 0.0, 1.0) # just in case
            
            adjusted_lsd = lowest_signed_distance + MAX_PENETRATION
            if lowest_signed_distance >= DISTANCE_THRESHOLD:
                penetration_cost = 0
                closeness_cost = 0
            elif lowest_signed_distance >=0:
                closeness_cost = math.exp(-(close_cost_factor1*(adjusted_lsd)/DISTANCE_THRESHOLD))
                # penetration_cost = 0
            else:
                # penetration_cost = math.exp(-(2.5*lowest_signed_distance/MAX_PENETRATION))/(10)
                closeness_cost = math.exp(-(closet_cost_factor2*(adjusted_lsd)/DISTANCE_THRESHOLD))
            
            final_manip_cost = 2*closeness_cost*scaled_manip_cost
            manip_costs.append(final_manip_cost)
        
        node_weighted_manip_cost = np.sum(manip_costs)/len(manip_costs) if manip_costs else 0
        total_manip_cost = (node_weighted_manip_cost + start_node.manip_cost*start_node.num_in_path)/(1+start_node.num_in_path)
        # normalized_final_manip_cost = final_manip_cost * distance_cost
        total_cost = self.weight_dist*distance_cost + distance_cost*total_manip_cost*(1-self.weight_dist) #TODO is this legit, otherwise farther goals will weight distance more??
        # total_cost = (1-COL_COST_RATIO)*total_cost + COL_COST_RATIO*penetration_cost

        return total_cost, distance_cost, distance, total_manip_cost, node_weighted_manip_cost
        

    def argmin(self, function, nodes):
        min_node = min(nodes, key=function)
        return  min_node
        # values = list(nodes)
        # scores = [function(x) for x in values]
        # return values[scores.index(min(scores))]


    def interpolate_configs(self, config1, config2, num_steps=10):
        """Interpolate between two configurations. exclusive of start, exclusive of end"""
        return [
            [(1 - t) * c1 + t * c2 for c1, c2 in zip(config1, config2)]
            for t in (i / float(num_steps + 1) for i in range(1, num_steps + 1))
        ]
    
    # def interpolate_nodes_euclid(self, node1, node2, num_steps=10):
    #     """Interpolate between two configurations. exclusive of start, exclusive of end"""
    #     xyz_list = [
    #         [(1 - t) * c1 + t * c2 for c1, c2 in zip(node1.xyz_quat[0], node2.xyz_quat[0])]
    #         for t in (i / float(num_steps + 1) for i in range(1, num_steps + 1))
    #     ]
    #     xyz_quat_list = [[np.array(xyz), node2.xyz_quat[1]] for xyz in xyz_list] #TODO INTERPOLATE QUAT
    #     q_list = []
    #     q = node1.config
    #     for xyz_quat in xyz_quat_list:
    #         # q = self.ee_IK(xyz_quat, q, random_start = False)
    #         q = self.mink_collision_ik(
    #                     target_pos=xyz_quat[0],
    #                     target_quat=xyz_quat[1],
    #                     initial_guess=q,
    #                     taxel_target = None,
    #                     random_start = False
    #                     )
    #         if q is None:
    #             return None, None
    #         q_list.append(np.array(q))

    #     return q_list, xyz_quat_list


    def extend_fn(self, closest_node, xyz_quat_target):
        xyz_start = np.array(closest_node.xyz_quat[0])
        quat_start = np.array(closest_node.xyz_quat[1])
        xyz_target = np.array(xyz_quat_target[0])
        quat_target = np.array(xyz_quat_target[1])
        
        # Calculate the direction vector and its norm
        direction = xyz_target - xyz_start
        distance = np.linalg.norm(direction)
        
        if distance <= self.extend_step:
            return xyz_quat_target
        else:
            # Normalize the direction vector and scale by the extend step
            normalized_direction = direction / distance
            new_xyz = xyz_start + self.extend_step * normalized_direction
            
            # Compute the interpolation factor (percentage of total distance)
            interp_factor = self.extend_step / distance
            
            # Perform SLERP using Scipy's Slerp
            key_rots = R.from_quat([quat_start, quat_target])
            key_times = [0, 1]
            slerp = Slerp(key_times, key_rots)
            new_quat = slerp([interp_factor]).as_quat()[0]
            
            return [np.array(new_xyz), np.array(new_quat)]
        
    # def check_collision_and_manip(self,q,debug_life = -1):
    #     #returns collision, manipulability, lowest signed distance
    #     collision = False
    #     # Set the robot's joints to the given configuration
    #     set_joint_state(self.robot_id, q, self.joint_indices)
    #     if not NOCOL:
    #         # Check for collisions between the robot and obstacles
    #         for obstacle in self.obstacles:
    #             for link_index in range(9):  # Check first 9 links (not taxels)
    #                 contact_points = p.getClosestPoints(self.robot_id, obstacle, linkIndexA=link_index, distance=0)
    #                 if len(contact_points) > 0:
    #                     return True, None, None  # Collision detected for this link

    #     lowest_signed_distance = 10000
    #     signed_dist = None
    #     start1 = 0
    #     end1 = 0
    #     closest_link_index = None
    #     for obstacle_id in self.obstacles:
    #         for link_index in range(9,p.getNumJoints(self.robot_id)):
    #             signed_dist,start,end = compute_signed_distance_for_link(self.robot_id, link_index, obstacle_id, distance_threshold=DISTANCE_THRESHOLD)
    #             if signed_dist is not None and signed_dist < lowest_signed_distance:
    #                 lowest_signed_distance = signed_dist
    #                 closest_link_index = link_index
    #                 end1 = end
    #                 start1 = start
    #     if lowest_signed_distance > DISTANCE_THRESHOLD: return False, 1, 10000
    #     if debug_life >= 0:
    #         p.addUserDebugLine(start1, end1, lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=debug_life)
    #         # if TEST_Q:
    #         #     p.addUserDebugLine([end1[0],-5,end1[2]], [end1[0],5,end1[2]], lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0)
    #         #     p.addUserDebugLine([-5,end1[1],end1[2]], [5,end1[1],end1[2]], lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0)
    #         #     p.addUserDebugLine([end1[0],end1[1],-5], [end1[0],end1[1],5], lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0)
    #         #     p.addUserDebugLine([start1[0],-5,start1[2]], [start1[0],5,start1[2]], lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=0)
    #         #     p.addUserDebugLine([-5,start1[1],start1[2]], [5,start1[1],start1[2]], lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=0)
    #         #     p.addUserDebugLine([start1[0],start1[1],-5], [start1[0],start1[1],5], lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=0)

    #     # time.sleep(1)
    #     J, _ = p.calculateJacobian(self.robot_id, closest_link_index, [0,0,0], q, [0]*7, [0]*7)
    #     J = correct_J_form(self.robot_id, J)
    #     JJT = np.dot(J, np.transpose(J))
    #     det_JJT = np.linalg.det(JJT)
    #     if det_JJT < 0:
    #         manipulability = 1 
    #     else:
    #         manipulability = np.sqrt(det_JJT)
        
    #     return manipulability, lowest_signed_distance


    def collision_fn(self, q):
        # print(f"checking for collision for q of: {q}")
        if NOCOL:
            return False
        # Set the robot's joints to the given configuration
        set_joint_state(self.robot_id, q, self.joint_indices)
        # Check for collisions between the robot and obstacles
        for obstacle in self.obstacles:
            for link_index in range(9):  # Check first 9 links (not taxels)
                contact_points = p.getClosestPoints(self.robot_id, obstacle, linkIndexA=link_index, distance=0)
                if len(contact_points) > 0:
                    # print(f"collision at link: {link_index}")
                    return True  # Collision detected for this link
        return False

    def check_if_xyz_in_obstacle(self, xyz):
        """
        Checks whether the xyz point [x, y, z] is inside any of the obstacles.
        Uses PyBullet's collision detection for efficiency.
        """
        ray_from = xyz
        ray_to = [xyz[0], xyz[1], xyz[2] + 0.0000001]  # Small distance along the z-axis

        # Perform the ray test
        result = p.rayTest(ray_from, ray_to)
        hit_id = result[0][0]
        
        if hit_id > 0:
            return True  # The point is in the obstacle
        return False  # The point is not inside any obstacle

    
    def calculate_taxel_manip_and_dist(self,q,debug_life = -1):
        set_joint_state(self.robot_id, q, self.joint_indices)
        lowest_signed_distances = []
        closest_taxel_ids = []
        manipulabilities = []
        ends = []
        starts = []
        signed_dist = None
        start1 = 0
        end1 = 0
        closest_taxel_id = None
        for obstacle_id in self.obstacles:
            for link_index in range(9,p.getNumJoints(self.robot_id)):
                signed_dist,start,end = compute_signed_distance_for_link(self.robot_id, link_index, obstacle_id, distance_threshold=DISTANCE_THRESHOLD)
                if signed_dist is not None and signed_dist < DISTANCE_THRESHOLD:
                    lowest_signed_distances.append(signed_dist)
                    closest_taxel_ids.append(link_index)
                    ends.append(end)
                    starts.append(start)
        if not lowest_signed_distances: 
            return [1], [10000], [-1]
       
        for i, closest_taxel_id  in enumerate(closest_taxel_ids):
            if debug_life >= 0:
                p.addUserDebugLine(starts[i], ends[i], lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=debug_life)
        
            J, _ = p.calculateJacobian(self.robot_id, closest_taxel_id, [0, 0, 0], q.tolist(), [0]*7, [0]*7)
            J = correct_J_form(self.robot_id, J)
            JJT = np.dot(J, np.transpose(J))
            det_JJT = np.linalg.det(JJT)
            det_JJT = max(det_JJT, 0)
            manipulability = np.sqrt(det_JJT)
            manipulabilities.append(manipulability)
            
        return manipulabilities, lowest_signed_distances, closest_taxel_ids

    def goal_test(self, xyz_quat):
        # Check if the current configuration is within the tolerance of the goal
        if self.euclid_distance_fn(xyz_quat, self.xyz_quat_goal) < self.goal_tolerance:# and self.quaternion_angle_dif(xyz_quat[1],self.xyz_quat_goal[1])<45:
            return True
        # TODO check quat too maybe
        return False
    
    def quaternion_angle_dif(self,quat1, quat2):
        """
        Computes the angle difference between two unit quaternions in degrees.
        """
        dot_product = sum(q1 * q2 for q1, q2 in zip(quat1, quat2))
        dot_product = max(-1.0, min(1.0, dot_product))
        angle_rad = 2 * math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    

    def test_q(self):
        q1 = [0.0, -0.116, 0.0, -1.848, 0.0, 0.0, 0.0]
        set_joint_state(self.robot_id, q1, self.joint_indices)
        manips, lowest_signed_distances,closest_taxel_ids = self.calculate_taxel_manip_and_dist(q1,0)
        print(f"manips are : {manips}")
        print(f"lowest signed difs are : {lowest_signed_distances}")
        wait_for_user()
        disconnect()

    # def update_path(self, node_path):
    #     return
    #     node_path2 = []
    #     for node in node_path:
    #         if not node_path2 or node_path2[-1].config != node.config:
    #             node_path2.append(node)
    #     node_path = node_path2

    #     upper = 2
    #     lower = 0
    #     while upper < len(node_path) and lower >= 0:
    #         parent = node_path[lower]
    #         target = node_path[upper]
    #         initial_cost  = target.total_cost + node_path[upper-1].total_cost + parent.total_cost
    #         new_nodes = []
    #         new_nodes.append(parent)
    #         num_steps = math.floor(np.max(np.abs(np.array(parent.config)[:]-np.array(target.config)[:]))/self.extend_step) 
    #         q_list = self.interpolate_configs(parent.config, target.config, num_steps = num_steps)
    #         parent_interp_node = parent
    #         fail = False
    #         for i in range(len(q_list)):
    #             q_interp = q_list[i]
    #             col = self.collision_fn(q_interp)
    #             manip, lowest_signed_distance = self.calculate_taxel_manip_and_dist(q_interp)
    #             if not col and lowest_signed_distance >= -MAX_PENETRATION:
    #                 [parent_interp_node,_] = self.create_node(q_interp, manip, lowest_signed_distance=lowest_signed_distance) 
    #                 new_nodes.append(parent_interp_node)
    #             else:
    #                 fail = True
    #                 i = len(q_list)
    #         new_cost = target.total_cost
    #         for node in new_nodes:
    #             new_cost = new_cost + node.total_cost
    #         if fail == False and new_cost < initial_cost:
    #             node_path = node_path[0:lower] + new_nodes + node_path[upper:]
    #         lower = lower + 1
    #         upper = upper + 1
                    
        
    #     # goal = node_path[-1]
    #     # node = goal
    #     # while node.dist_to_last > 0:
    #     #     new_parent = self.find_parent(node,0)
    #     #     if new_parent is not None:
    #     #         node.parent = new_parent
    #     #         [node.total_cost, node.d_cost, node.dist_to_last, node.manip_cost] = self.cost_fn(new_parent,node)
    #     #     node = node.parent
    #     return node_path

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
            closest_idx = np.argmin(node.lowest_signed_distances)
            manips = node.manipulabilities
            gradient_dot(self.robot_id, manips[closest_idx])

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
        print("\nvisualizing path\n")
        
        if REDUCE_OBJECTS:
            box1_position = [0.35, -0.3, 0.13]
            box1_dims = [0.26,1,0.14]
            col_box_id1 = create_box(box1_position,box1_dims)
            p.changeVisualShape(col_box_id1, -1, rgbaColor=[0.2,0.2,0.2,0.7])
            cyl_position1 = (0.35,-0.3,0.3)
            cyl_quat1 = p.getQuaternionFromEuler([3.14/2, 0, 0])
            rad1 = 0.18 
            h1 = 1
            col_cyl_id1 = create_cylinder(rad1, h1, cyl_position1, cyl_quat1)
            p.changeVisualShape(col_cyl_id1, -1, rgbaColor=[0.2,0.2,0.2,0.7])
        
        # Remove any existing lines
        line_ids = []
        screenshots = []
        # Loop through each configuration in the path
        for i, q in enumerate(path):
            screenshots.append(save_screenshot())
            # time.sleep(0.2)
    
            # If it's not the first configuration, draw a line from the previous one to the current one, and interpolate for visuals
            if i > 0:
                prev_q = path[i - 1]

                #interpolate and to visualize moving from waypoint to waypoint
                num_steps = math.ceil(5*np.max(np.abs(np.array(q)[:]-np.array(prev_q)[:]))/self.res) 
                q_list = self.interpolate_configs(prev_q, q, num_steps = num_steps)
                q_list.append(q)
                for q_interp in q_list:
                    time.sleep(0.2/num_steps)
                    set_joint_state(self.robot_id, q_interp)
                    # for joint_idx, joint_angle in zip(self.joint_indices, q_interp):
                        # p.setJointMotorControl2(self.robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=joint_angle)
                    p.stepSimulation()

                # Get the 3D world positions of the end-effector for the previous and current configuration
                prev_pos = q_to_endeff(self.robot_id, prev_q)[0]
                curr_pos = q_to_endeff(self.robot_id, q)[0]
                # Draw a line between the two configurations
                if isinstance(prev_pos, np.ndarray): prev_pos = prev_pos.tolist()
                if isinstance(curr_pos, np.ndarray): curr_pos = curr_pos.tolist()
                line_id = p.addUserDebugLine(prev_pos, curr_pos, lineColorRGB=line_color, lineWidth=1.5)
                line_ids.append(line_id)

                
            set_joint_state(self.robot_id, q)
                # p.setJointMotorControl2(self.robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=joint_angle)
            # Step the simulation to allow the robot to move
            p.stepSimulation()
        turn_screenshots_to_gif(screenshots)
        # move_through_path(robot_id, path, joint_indices, time_step=pause_time)
        return line_ids
    
    


def print_path(path):
    i = 1
    for config in path:
        # Round each joint angle in the configuration to 3 decimal points
        rounded_config = [round(angle, 3) for angle in config]
        # Print the rounded configuration
        print(f"node {str(i)}: {str(rounded_config)}")
        i += 1
    return

def set_joint_state(robot_id, q, joint_indices=[0,1,2,3,4,5,6]):
        for joint_index in joint_indices:
            p.resetJointState(robot_id, joint_index, q[joint_index])
        p.stepSimulation()

def configs(nodes):
    if nodes is None:
        return None
    return list(map(lambda n: n.config, nodes))

def q_to_endeff(robot_id, q):
        set_joint_state(robot_id, q)
        link_state = p.getLinkState(robot_id, 7)
        return np.array(link_state[4]), np.array(link_state[5])
    
def faster_q_to_endeff(robot_id, q, pin_model, pin_model_data):
    """
    Computes the end-effector position and orientation using Pinocchio forward kinematics.

    Args:
        robot_id: PyBullet robot ID (not used here, kept for compatibility).
        q: Joint configuration (list or numpy array).
        pin_model: Pinocchio model object.
        pin_model_data: Pinocchio data object.

    Returns:
        Tuple (end_effector_position, end_effector_orientation_quaternion).
    """
    # Convert the joint configuration to Pinocchio format
    q_eigen = pin.utils.zero(len(q))
    q_eigen[:len(q)] = q

    # Perform forward kinematics
    pin.forwardKinematics(pin_model, pin_model_data, q_eigen)
    pin.updateFramePlacements(pin_model, pin_model_data)

    # Get the end-effector frame ID
    ee_frame_id = pin_model.getFrameId("EndEffector")

    # Extract the position and orientation of the end-effector
    ee_position = pin_model_data.oMf[ee_frame_id].translation
    ee_rotation = pin_model_data.oMf[ee_frame_id].rotation
    ee_orientation_quaternion = pin.Quaternion(ee_rotation).coeffs()

    return np.array(ee_position), np.array(ee_orientation_quaternion)

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

def gradient_dot(robot_id, manip,alpha=1,ee_num=7):
    if NO_DOT is True: return None
    # Get the position of the end-effector (assuming it's the last joint/link)
    link_state = p.getLinkState(robot_id, ee_num)
    end_effector_position = link_state[4]  # Position is index [4] in the result
    manip = min(1,manip/0.2)
    # Create a small visual sphere at the end-effector's position
    dot_radius = 0.008  # A small dot
    color = [1-manip, 0, manip, alpha]  # RGBA color for blue

    # Create a visual shape for the dot (small sphere)
    dot_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=dot_radius, rgbaColor=color)

    # Create the body for the visual shape (this places the sphere at the end-effector position)
    dot_id = p.createMultiBody(
        baseMass=0,  # Mass of 0 means it's static and won't be affected by physics
        baseVisualShapeIndex=dot_visual_shape,
        basePosition=end_effector_position
    )
    return dot_id 

def dot(pos,color = [1,0,0,1],dot_radius = 0.01):
    if NO_DOT is True: return
    # Create a small visual sphere at the end-effector's position

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

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.
    :param quaternion: A list or array [qx, qy, qz, qw]
    :return: A 3x3 rotation matrix
    """
    qx, qy, qz, qw = quaternion
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw),     1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx**2 + qy**2)]
    ])
    return R

def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    :param R: A 3x3 rotation matrix
    :return: A list [qx, qy, qz, qw]
    """
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return [qx, qy, qz, qw]

def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

# def compare_world_frames(robot_id, q, model, data, ee_site_name):
#     """
#     Compares MuJoCo and PyBullet end-effector world positions for same joint config.
#     """
#     pybullet_pos, pybullet_quat = q_to_endeff(robot_id, q)
#     dot(pybullet_pos, color=[0, 1, 0, 1], dot_radius=0.01)
#     wait_for_user()
#     mujoco_pos, mujoco_quat = get_mujoco_ee_world_pose(model, data, q, ee_site_name)

#     print("PyBullet EE World Pos:", pybullet_pos)
#     print("MuJoCo EE World Pos:", mujoco_pos)

#     pos_diff = np.linalg.norm(pybullet_pos - mujoco_pos)
#     print("Position Difference (meters):", pos_diff)

#     return pos_diff

def rotate_vector_by_quaternion(vector, quaternion):
    """Rotate a vector by a quaternion."""
    q = quaternion / np.linalg.norm(quaternion)  # Normalize the quaternion if necessary
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])  # Conjugate of quaternion
    
    # Convert the vector to a quaternion (w = 0 for vector)
    vector_quat = np.array([0, *vector])
    
    # Perform the rotation using quaternion multiplication: q * vector * q_conj
    rotated_vector = quaternion_multiply(quaternion_multiply(q, vector_quat), q_conj)
    
    return rotated_vector[1:]  # Return the x, y, z components of the rotated vector

def angle_between_z_axes(q1, q2):
    """Compute the angle between the z-axis of two rotated frames by quaternions q1 and q2."""
    # Original z-axis
    original_z = np.array([0, 0, 1])
    
    # Rotate the z-axis by both quaternions
    rotated_z1 = rotate_vector_by_quaternion(original_z, q1)
    rotated_z2 = rotate_vector_by_quaternion(original_z, q2)
    
    # Compute the cosine of the angle between the two vectors
    cos_angle = np.dot(rotated_z1, rotated_z2) / (np.linalg.norm(rotated_z1) * np.linalg.norm(rotated_z2))
    
    # Clamp cos_angle to avoid numerical errors that may result in values slightly outside the range [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Compute the angle in radians
    angle = np.arccos(cos_angle)
    
    return angle

def euler_yzx_to_quaternion(euler_angles):
    # Extract YZX Euler angles (y, z, x)
    y, z, x = euler_angles
    
    # Precompute sine and cosine values for efficiency
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    cz, sz = np.cos(z / 2), np.sin(z / 2)
    cx, sx = np.cos(x / 2), np.sin(x / 2)
    
    # Create quaternions for each rotation axis
    q_y = np.array([0, sy, 0, cy])  # Rotation about Y-axis
    q_z = np.array([0, 0, sz, cz])  # Rotation about Z-axis
    q_x = np.array([sx, 0, 0, cx])  # Rotation about X-axis
    
    # Apply the intrinsic rotations in X -> Z -> Y order
    q_final = quaternion_multiply(q_y, quaternion_multiply(q_z, q_x))
    
    return q_final

def quaternion_multiply(q1, q2):
    
    # Quaternion multiplication (Hamilton product)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([x3, y3, z3, w3])

def get_taxels(robot_id, yaml_file=r'C:\Users\arbxe\OneDrive\Desktop\code stuff\EmPRISE repos\Manip_planning\mp-osc\multipriority\configs\real_taxel_data_v2.yaml'):
    # Load YAML data
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    taxel_ids = []

    # Lists for all links' data to be passed to createMultiBody at once
    link_masses = []
    link_collision_shapes = []
    link_visual_shapes = []
    link_positions = []
    link_orientations = []
    link_parent_indices = []
    link_joint_types = []
    link_joint_axis = []
    baseMass=[],  # No mass for the base (it will be the robot body, not used in this case)
    baseVisualShapeIndex=[],  # No visual for the base
    basePosition=[],
    baseOrientation = [],

    # Create objects based on YAML data
    for key, values in data.items():
        link = values['Link']  # The robot's link index where the taxel will be attached
        local_position = np.array(values['Position'])  # Local position in the link frame
        normal = np.array(values['Normal'])  # Normal vector for the taxel orientation

        # Get the current world position and orientation of the link
        link_state = p.getLinkState(bodyUniqueId=robot_id, linkIndex=link)
        link_world_pos = np.array(link_state[0])  # World position of the link
        link_world_orientation = link_state[1]     # World orientation of the link (quaternion)

        # Calculate the world position of the taxel based on the link's transform
        link_rotation_matrix = np.array(p.getMatrixFromQuaternion(link_world_orientation)).reshape(3, 3)
        world_position = link_world_pos + link_rotation_matrix.dot(local_position)

        # Convert the normal vector to quaternion orientation
        R = quaternion_to_rotation_matrix(normal_to_quaternion(normal))  # Get rotation matrix from normal
        orientation = rotation_matrix_to_quaternion(np.dot(link_rotation_matrix, R))  # Final orientation

        # Define a small rectangular box shape for the taxel
        half_extents = [0.01, 0.02, 0.03]  # Adjust dimensions as needed
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=half_extents, 
            rgbaColor=[0, 0, 1, 1]  # Blue for visibility
        )

        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)  # Collision shape

        # Append the data for this taxel to the respective lists
        link_masses.append(0)  # No mass for taxel link
        link_collision_shapes.append(collision_shape)  # Collision shape for the taxel
        link_visual_shapes.append(visual_shape)  # Visual shape for the taxel
        link_positions.append(local_position.tolist())  # Position relative to the parent link (robot's link)
        link_orientations.append(orientation)  # Orientation relative to the parent link
        link_parent_indices.append(link)  # Parent link index (robot's link)
        link_joint_types.append(p.JOINT_FIXED)  # Fixed joint type to attach the taxel as a fixed link
        link_joint_axis.append([0, 0, 0])  # No axis for the fixed joint
        # baseMass.append(0)  # No mass for the base (it will be the robot body, not used in this case)
        # baseVisualShapeIndex.append(-1)  # No visual for the base
        # basePosition.append([0, 0, 0])
        # baseOrientation.append([0, 0, 0, 1])
        

    # Debugging: print the lists to check for consistency
    print(f"link_masses length: {len(link_masses)}")
    print(f"link_collision_shapes length: {len(link_collision_shapes)}")
    print(f"link_visual_shapes length: {len(link_visual_shapes)}")
    print(f"link_positions length: {len(link_positions)}")
    print(f"link_orientations length: {len(link_orientations)}")
    print(f"link_parent_indices length: {len(link_parent_indices)}")
    print(f"link_joint_types length: {len(link_joint_types)}")
    print(f"link_joint_axis length: {len(link_joint_axis)}")

    # Check the first few values of each list to ensure they match expectations
    print(f"First link position: {link_positions[0]}")
    print(f"First link orientation: {link_orientations[0]}")
    print(f"First link parent index: {link_parent_indices[0]}")
    print(f"First link joint type: {link_joint_types[0]}")

    # Ensure that all lists have the same length before proceeding
    if len(link_masses) == len(link_collision_shapes) == len(link_visual_shapes) == len(link_positions) == len(link_orientations) == len(link_parent_indices) == len(link_joint_types) == len(link_joint_axis):
        # Create the taxels as child links to the robot in a single call to createMultiBody
        taxel_id_temp = p.createMultiBody(
            # baseMass=baseMass,  # No mass for the base (it will be the robot body, not used in this case)
            # baseVisualShapeIndex=baseVisualShapeIndex,  # No visual for the base
            # basePosition=basePosition,  # Base position (not relevant as we have child links)
            # baseOrientation=baseOrientation,  # Base orientation (identity quaternion)
            linkMasses=link_masses,  # List of masses for each taxel link
            linkCollisionShapeIndices=link_collision_shapes,  # List of collision shapes for each taxel link
            linkVisualShapeIndices=link_visual_shapes,  # List of visual shapes for each taxel link
            linkPositions=link_positions,  # List of positions for each taxel relative to its parent link
            linkOrientations=link_orientations,  # List of orientations for each taxel relative to its parent link
            linkParentIndices=link_parent_indices,  # List of parent indices for each taxel (same link index)
            linkJointTypes=link_joint_types,  # Joint types (all fixed)
            linkJointAxis=link_joint_axis,  # Joint axis (fixed joints have no axis)
        )
        # Add taxel ID to the list
        taxel_ids.append(taxel_id_temp)
    else:
        print("Error: The lengths of the link attributes do not match!")
    
    return taxel_ids


def compute_signed_distance_for_link(robot_id, link_index, obstacle_id, distance_threshold=DISTANCE_THRESHOLD):
    closest_points = p.getClosestPoints(bodyA=robot_id, bodyB=obstacle_id, linkIndexA=link_index, distance=distance_threshold)
    if closest_points:
        # Extract the closest point (smallest distance)
        closest_point = min(closest_points, key=lambda point: point[8])  # point[8] is the distance or penetration depth (-)
        point_start = closest_point[5]  # Position on robot
        point_end = closest_point[6]    # Position on obstacle
        closest_distance = closest_point[8]  # Closest distance
        
        return closest_distance,point_start, point_end  # Positive for separation (no collision), negative for penetraction
    # If no contact or closest points, return None
    return None,None,None


def get_distance_between_bodies(body_id1, body_id2):
    # Get positions of both bodies
    pos1, _ = p.getBasePositionAndOrientation(body_id1)
    pos2, _ = p.getBasePositionAndOrientation(body_id2)
    
    # Calculate the Euclidean distance between the two positions
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
    return distance
    
def viz_mujoco():
    # MUJOCO viz test
    model_m = mujoco.MjModel.from_xml_path(TACTILE_KINOVA_URDF_FOR_MUJOCO)
    # for geom_id in range(model_m.ngeom):
    #     model_m.geom_rgba[geom_id, 3] = 0.5  # Set alpha (transparency) to 0.5 for half transparency
    data_m = mujoco.MjData(model_m)
    model_m.opt.gravity[:] = 0
    model_m.geom_conaffinity[:] = 1
    model_m.geom_contype[:] = 1
    with mujoco.viewer.launch_passive(model_m, data_m) as viewer:
        print("Press ESC to exit")
        while viewer.is_running():
            mujoco.mj_step(model_m, data_m)
            viewer.sync()
    wait_for_user()    
    
def viz_mujoco_collision_bodies():
    # Load model
    model = mujoco.MjModel.from_xml_path(TACTILE_KINOVA_URDF_FOR_MUJOCO)
    data = mujoco.MjData(model)
    
    # Disable gravity to keep robot still
    model.opt.gravity[:] = 0
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    # Set visual geoms fully transparent (group 2 = visuals in your MJCF)
    for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] == 2:  # visual meshes
            model.geom_rgba[geom_id, 3] = 0.6  # fully transparent
        
        if model.geom_group[geom_id] == 3:  # collision capsules/boxes
            print(f"Geom {geom_id} is a collision geom with type {model.geom_type[geom_id]}")
            model.geom_group[geom_id] = 0
            model.geom_rgba[geom_id] = [1.0, 0.1, 0.1, 0.6]  # red, semi-transparent

        # Optional: obstacles (group 4 or others)
        if model.geom_group[geom_id] == 4:
            model.geom_rgba[geom_id] = [0.0, 1.0, 0.0, 0.5]  # green obstacles

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Visualizing collision geoms — Press ESC to exit")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

def draw_xyz_bounding_box(xyz_limits, color=[1, 0, 0], line_width=1, life_time=0):
    """
    Draws a bounding box in PyBullet using the xyz limits defined in xyz_limits.

    Args:
        color: RGB color for the box lines (default: red)
        line_width: Thickness of the lines
        life_time: Duration in seconds (0 = permanent)
    """
    x_min, y_min, z_min = xyz_limits[0]
    x_max, y_max, z_max = xyz_limits[1]

    # 8 corner points of the box
    corners = [
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max]
    ]

    # 12 edges of the box (as index pairs into the corners list)
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    for start, end in edges:
        p.addUserDebugLine(
            lineFromXYZ=corners[start],
            lineToXYZ=corners[end],
            lineColorRGB=color,
            lineWidth=line_width,
            lifeTime=life_time
        )


class TreeNode(object):

    def __init__(self, xyz_quat, config, parent=None,manipulabilities=[], lowest_signed_distances = [], 
                 closest_taxel_ids = [], dist_to_last = 0, d_cost = 0, 
                 manip_cost = 0, node_manip_cost = 0, num_in_path = 1, 
                 c_sample = False):
        self.xyz_quat = xyz_quat
        self.config = np.array(config)
        self.parent = parent
        self.manipulabilities = manipulabilities
        self.lowest_signed_distances = lowest_signed_distances
        self.dist_to_last = dist_to_last
        self.manip_cost = manip_cost
        self.node_manip_cost = node_manip_cost
        self.d_cost = d_cost
        self.total_cost = manip_cost+d_cost
        self.num_in_path = num_in_path
        self.closest_taxel_ids = closest_taxel_ids
        self.c_sample = c_sample

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
        return self.manipulabilities

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

def save_screenshot(mult=2):
    width, height = mult * 640, mult * 480  # Set the desired screenshot resolution
    view_matrix = p.getDebugVisualizerCamera()[2]
    projection_matrix = p.getDebugVisualizerCamera()[3]
    image_data = p.getCameraImage(width, height, view_matrix, projection_matrix)
    rgb_array = np.array(image_data[2], dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    image = Image.fromarray(rgb_array)
    return image

def turn_screenshots_to_gif(frames):
    output_folder = OUTPUT_GIF_FOLDER
    os.makedirs(output_folder, exist_ok=True)
    # Get all existing GIF filenames in the output folder
    existing_files = os.listdir(output_folder)
    numbers = []
    
    # Find the highest number used in existing gif files
    for filename in existing_files:
        match = re.search(r"screenshot_animation_(\d+)\.gif", filename)
        if match:
            numbers.append(int(match.group(1)))
    
    # Determine the next number for the new GIF
    next_number = max(numbers, default=0) + 1
    gif_name = f"screenshot_animation_{next_number}.gif"

    # Save the GIF
    gif_path = os.path.join(output_folder, gif_name)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_DURATION,
        loop=0
    )

    print(f"GIF created at {gif_path}")

    # # Open the GIF
    # if os.name == "posix":  # Linux or macOS
    #     os.system(f'xdg-open "{gif_path}"')
    # else:  # Windows
    #     os.system(f'start "" "{gif_path}"')

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Delete the file or symbolic link
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Remove an empty directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    

def save_data (trial_num, node_path, rANDOM_SEED, sim_type,gOAL_AREA_SAMPLE_PROB,rRT_MAX_ITERATIONS,rRT_MIN_ITERATIONS,eXTEND_STEP_SIZE,xyz_gOAL_TOLERANCE,gOAL_AREA_DELTA,gOAL_SAMPLE_PROB,
             r_FOR_PARENT,eDGE_COLLISION_RESOLUTION,wEIGHT_FOR_DISTANCE,object_reduction,contact_sample_chance):
    
    manip = []
    node_manip_cost = []
    total_costs = []
    dist_to_last = []
    if node_path is not None and len(node_path) != 0:
        path = configs(node_path)
        joint_path_matrix = np.empty((len(path[0]), 0))
        for config in path:
            # Round each joint angle in the configuration to 3 decimal points
            rounded_config = np.array([round(angle, 3) for angle in config])
            rounded_config = rounded_config.reshape(-1, 1)
            joint_path_matrix = np.hstack((joint_path_matrix, rounded_config))

        for node in node_path:
            closest_idx = np.argmin(node.lowest_signed_distances)
            manips = node.manipulabilities
            manip.append(manips[closest_idx])
        manip = np.array(manip)
        manip = [round(x, 20) for x in manip]
        
        for node in node_path:
            node_manip_cost.append(node.node_manip_cost)
        node_manip_cost = np.array(node_manip_cost)
        node_manip_cost = [round(x, 20) for x in node_manip_cost]
        
        for node in node_path:
            total_costs.append(node.total_cost)
        total_costs = np.array(total_costs)
        total_costs = [round(x, 5) for x in total_costs]

        for node in node_path:
            dist_to_last.append(node.dist_to_last)
        dist_to_last = np.array(dist_to_last)
        dist_to_last = [round(x, 3) for x in dist_to_last]
        
        final_dist_cost = node_path[-1].d_cost
        final_total_cost = node_path[-1].total_cost
        final_manip_cost = node_path[-1].manip_cost
    else:
        final_dist_cost = "N/A"
        final_total_cost = "N/A"
        final_manip_cost = "N/A"
        joint_path_matrix = np.empty((0, 0))

    test_type = "".join([
        SAVE_DATA_PREFIX,"_weight",str(wEIGHT_FOR_DISTANCE),"_contactsamplechance",str(contact_sample_chance),"_objreduction", str(object_reduction), "_Min Iterations", str(rRT_MIN_ITERATIONS)
        ])
    csv_filepath = os.path.join(CSV_FOLDER_LOCATION, DATA_FOLDER_NAME, test_type + ".csv")
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)

    with open(csv_filepath, 'a', newline='') as fd:
        writer = csv.writer(fd)
    
        # Write the `manip` list with title in the first column
        writer.writerow(['Trial'] + [str(trial_num)] + ["Sim Type"] + [str(sim_type)] 
                        + ["Distance Weight"] + [str(wEIGHT_FOR_DISTANCE)] + ["Random Seed"] + [str(rANDOM_SEED)]
                        + ["Min Iterations"] + [str(rRT_MIN_ITERATIONS)] 
                        + ["Distance cost"] + [str(final_dist_cost)] + ["Manip Cost"] + [str(final_manip_cost)] 
                        + ["Total cost"] + [str(final_total_cost)] + ["Object Reduction"] + [str(object_reduction)]
                        + ["Contact Sample Chance"] + [str(contact_sample_chance)]
                        + ["Parent finding radius R"] + [str(round(r_FOR_PARENT, 3))] + ["Step Size"] + [str(eXTEND_STEP_SIZE)]
                        + ["Goal Area Sample Probability"] + [str(gOAL_AREA_SAMPLE_PROB)] + ["Goal Area Delta"] + [str(gOAL_AREA_DELTA)])
        
        if len(manip) != 0:
            writer.writerow(['Closest Taxel Manip'] + manip)
            writer.writerow(['Distance to Last Node'] + dist_to_last)
            writer.writerow(['Total Cost'] + total_costs)
            writer.writerow(['Local Manip Cost'] + node_manip_cost)
        for i in range(joint_path_matrix.shape[0]):
            row = joint_path_matrix[i,:]
            row = [str(x) for x in row]
            joint_text = 'Joint ' + str(i) + ': '
            writer.writerow([joint_text] + row)

if __name__ == '__main__':
    new_main()


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