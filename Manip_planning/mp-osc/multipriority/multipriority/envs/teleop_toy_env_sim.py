import pygame
import sys
import numpy as np
import time
try:
    from pyrcareworld.envs import RCareWorld
except:
    print("RCareWorld not found (if possible: conda activate rcare)")
from enum import Enum, auto
from scipy.spatial.transform import Rotation as R
try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
except:
    print("ROS and friends not found")

import pinocchio as pin
import math
import random
import cv2
import tempfile
import sys
import os
import os.path as osp
import open3d as o3d
import atexit
import yaml
import pathlib
from multipriority.agents import UnityRobot, RealRobot

FORCE_THRESHOLD = 0.01

def filter_active_taxels(forces, ids, positions, threshold=0.0):
    # filter out taxels with force magnitude below threshold
    forces = np.array(forces)
    ids = np.array(ids)
    positions = np.array(positions)
    mask = np.where(forces > threshold)
    return forces[mask], ids[mask], positions[mask]

def get_skeleton_id(skeleton_pose, taxel_pose, threshold=0.2):
    min_distance = 100000
    min_id = None
    for i, pose in enumerate(skeleton_pose):
        dist = np.linalg.norm(pose - taxel_pose)
        if dist < min_distance:
            min_id = i
            min_distance = dist
    return min_id

class Taxel:
    def __init__(self, taxel_id, linkID):
        self.taxel_id = taxel_id
        self.linkID = linkID
        self.pos = None # global pos
        self.local_pos = None # local pos
        self.contactF = None
    
    def get_contact_info(self):
        return self.pos, self.vel, self.contactF, self.local_pos, self.linkID
    
    def update(self, current_force, global_pos, local_pos):
        # update contact info
        self.contactF = current_force
        self.pos = global_pos
        self.local_pos = local_pos

global failure
failure = 0

#Backend Enum 
class Backend(Enum):
    SIM = auto()
    REAL = auto()

#Input Enum - used to abstract input from pygame, allows for easy switching between keyboard and joystick
class Input(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    ONE = auto()
    TWO = auto()
    THREE = auto()
    FOUR = auto()
    POS_X = auto()
    NEG_X = auto()
    POS_Y = auto()
    NEG_Y = auto()
    POS_Z = auto()
    NEG_Z = auto()
    EE_MODE = auto()
    VECTOR_MODE = auto()

#ROS_Interface class - used to read and write from ROS without worrying about instantializing ROS nodes yourself 
class ROS_Interface():

    MAX_V_SCALE = 0.3
    MAX_EE_STEP = 0.001

    def __init__(self):
        #Pubs and Subs. These are the topics that the ROS_Interface class will read from and write to
        rospy.init_node('joint_state_listener') 
        self.jointSub = rospy.Subscriber("/joint_states", JointState, self.__joint_states_callback__)
        self.jointPub = rospy.Publisher('/new_velocity_controller/command', Float64MultiArray)

        #Current joint states. This is updated by the joint_states_callback function
        self.current_joint_states = None

        #Pinocchio model and data. Used to convert between ROS and Pinocchio joint configurations. 
        #This helps with IK and debatably FK
        self.model = pin.buildModelFromUrdf("gen3_7dof_vision.urdf")
        self.data = self.model.createData()
        
        #for safety, we don't want to write a high velocity to the robot. This is the threshold for the velocity
        self.thresh = 0.2

    #Callback function for the jointSub subscriber. This updates the current_joint_states variable
    def __joint_states_callback__(self, msg):
        self.current_joint_states = msg

    #Get the maximum scale factor for a given mode
    def get_max_scale_factor(self, mode):
        if mode == Mode.ENDEFFECTOR:
            return self.MAX_EE_STEP
        elif mode == Mode.VECTOR:
            return self.MAX_V_SCALE
        
    #Read the robots current joint configuration from ROS
    def read_joints(self):
        try:
            return self.current_joint_states.position[1:]
        except:
            print ("Empty message")
    
    #Convert a ROS joint configuration to a Pinocchio joint configuration
    def __ros_joints_to_pin__(self, config):
        q = pin.neutral(self.model)

        # Convert ROS joint configuration to pin configuration
        for i in range(len(self.model.joints)-1):
            jidx = self.model.getJointId(self.model.names[i + 1])
            qidx = self.model.idx_qs[jidx]
    
            # nqs[i] is 2 for continuous joints in pin
            if self.model.nqs[jidx] == 2:
                q[qidx] = np.cos(config[i])
                q[qidx + 1] = np.sin(config[i])
            else:
                q[qidx] = config[i]
        
        return  q
    
    #Convert a Pinocchio joint configuration to a ROS joint configuration
    def __pin_joints_to_ros__(self, q):
        config = np.zeros(len(self.model.joints) - 1)  # Subtract 1 because Pinocchio adds an extra joint for the universe

        for i in range(len(config)):
            jidx = self.model.getJointId(self.model.names[i + 1])
            qidx = self.model.idx_qs[jidx]

            if self.model.nqs[jidx] == 2:
                # For continuous joints, extract angle from sine and cosine components
                config[i] = np.arctan2(q[qidx + 1], q[qidx])
            else:
                config[i] = q[qidx]

        return config

    #Given a pin joint configuration, get the end effector position vector and rotation matrix
    def get_endeffector_pose(self, q):
        # Update the robot model and data
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get the end effector position and rotation
        end_effector_id = self.model.getFrameId("end_effector_link")
        end_effector_pose = self.data.oMf[end_effector_id]
        pos = end_effector_pose.translation
        rot = end_effector_pose.rotation

        return pos, rot
    
    #given a ROS joint configuration, right that configuration to a ROS topic for velocity control
    def write_joints(self, vector):
        
        print ("vector: ", vector)

        # Check if the vector is within the threshold (safety DO NOT DELETE or risk writing a high velocity to the robot)
        #If you want robot to move faster, slowly change threshold 
        if np.all(np.abs(vector) <= self.thresh):
            msg = Float64MultiArray()
            msg.data = vector
            print("write pos: ", msg.data)
            self.jointPub.publish(msg)
        #If the vector is outside the threshold, normalize it and write it to the robot
        else:
            vector = vector / np.linalg.norm(vector)
            msg = Float64MultiArray()
            msg.data = vector
            print("write pos: ", msg.data)
            self.jointPub.publish(msg)

    #Given a position and rotation, compute the joint configuration that will move the end effector to that pose
    #Note we pass in the inital joint configuration to help the IK solver converge faster AND compute velocity for control
    def __pose_to_joints__(self, pos, rot, inital):
        JOINT_ID = 7
        oMdes = pin.SE3(rot, pos)
        
        #eps should be pretty low since we want an accurate solution or else the velocity will be higher than it should be
        q = self.__ros_joints_to_pin__(inital)
        eps = 1e-4
        IT_MAX = 8000
        DT = 0.2
        damp = 1e-12

        #Set the bounds for the IK solver
        low = np.array([-np.pi]*11, dtype=np.float64)
        high = np.array([np.pi]*11, dtype=np.float64)

        for i in range(IT_MAX):
            # Update the robot model and data
            pin.forwardKinematics(self.model, self.data, q)
            pin.framesForwardKinematics(self.model, self.data, q)
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # Compute the error between the current and desired end effector pose
            frame_id = self.model.getFrameId("end_effector_link")
            dMi = oMdes.actInv(self.data.oMf[frame_id])
            err = pin.log(dMi).vector
            
            #if your failure count is high the ik solver should be tuned more OR a bug has been introduced into the method
            if i >= IT_MAX:
                global failure
                failure += 1
                print ("Failure count: ", failure)
                break
            
            #solution is written to the robot given that the error is acceptable
            if np.linalg.norm(err) < eps:
                self.write_joints((self.__pin_joints_to_ros__(q)-inital)/DT)
                pos, rot = self.get_endeffector_pose(q)
                return True
            
            #compute Jacobian and velocity to find the next joint configuration
            J = pin.getFrameJacobian(self.model,self.data,frame_id,pin.LOCAL)
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * DT)
            q = np.clip(q, low, high)
 
        
        #if the IK solver fails, write a zero velocity to the robot
        #DO NOT DELETE OR having the robot continously move 
        self.write_joints([0,0,0,0,0,0,0])
        return False

    #Given a velocity vector, write it to the robot
    def write_backend(self, mode, output):
        
        #if the mode is a vector, output will be a vector of joint velocities
        if mode == Mode.VECTOR:
            output_v = np.zeros(len(output[0]))
            for x in output:
                output_v = np.add(output_v, x)
            self.write_joints(output_v)
        
        #if the mode is a pose, output will be a tuple of position and rotation
        if mode == Mode.ENDEFFECTOR:
            diff_pos, dif_rot = output

            #get joint velocities, convert to pinnochio, and finally get end effector pose
            #This could be done with ROS and tf 
            pos, rot = self.get_endeffector_pose(self.__ros_joints_to_pin__(self.read_joints()))
            inital = np.array(self.read_joints())

            #calculate the new position and rotation
            pos += diff_pos
            rot = np.dot(rot, dif_rot)
            
            #write the new position and rotation to the robot
            self.__pose_to_joints__(pos, rot, inital)

#This class is used to interface with RCareWorld
class RCare_Interface():
    
    MAX_V_SCALE = 0.4
    MAX_EE_STEP = 0.01

    #initalize sim
    def __init__(self):
        self.env = RCareWorld()
        atexit.register(self.save_npz)

        self.env.robot = self.create_robot(
            id=315893,
            gripper_list=[3158930],
            robot_name="kinova_gen3_7dof-robotiq85",
            base_pos=[0, 0, 0])
        #right now the cube is used to move the robot end effector around the scene
        #theoretically RcWorld should be able to move the robot end effector around the scene, but it is not working
        self.cube = self.env.create_object(id=315867, name="Cube", is_in_scene=True)
        
        self.model = pin.buildModelFromUrdf("gen3_7dof_vision.urdf")
        self.data = self.model.createData()
        
        # Create skin to collect robot taxel data
        self.skin = self.env.create_skin(id=114514, name="Skin", is_in_scene=True)
        self.bar_poses = []
        # self.taxel1 = self.env.create_object(id=1004, name="Taxel1", is_in_scene=True)
        # print ("Taxel1: ", self.taxel1.getPosition())        
                     
        # Initializing environment objects (softbody bars and end goal cube)
        try:
            # Create bars to get position data to use for goal cube
            self.bar1 = self.env.create_object(id=91, name="Bar1", is_in_scene=True)
            b1 = self.bar1.getPosition()
            # b1.pop(1)
            self.bar_poses.append(b1)
            self.bar2 = self.env.create_object(id=92, name="Bar2", is_in_scene=True)
            b2 = self.bar2.getPosition()
            # b2.pop(1)
            self.bar_poses.append(b2)
            self.bar3 = self.env.create_object(id=93, name="Bar3", is_in_scene=True)
            b3 = self.bar3.getPosition()
            # b3.pop(1)
            self.bar_poses.append(b3)
            self.bar4 = self.env.create_object(id=94, name="Bar4", is_in_scene=True)
            b4 = self.bar4.getPosition()
            # b4.pop(1)
            self.bar_poses.append(b4)
            # self.bar5 = self.create_object(id=95, name="Bar5", is_in_scene=True) #camera bar
            # b5 = self.bar5.getPosition()
            # b5.pop(1)
            self.bar6 = self.env.create_object(id=96, name="Bar6", is_in_scene=True)
            b6 = self.bar6.getPosition()
            # b6.pop(1)
            self.bar_poses.append(b6)
            self.bar7 = self.env.create_object(id=97, name="Bar7", is_in_scene=True)
            b7 = self.bar7.getPosition()
            # b7.pop(1)
            self.bar_poses.append(b7)
            self.bar8 = self.env.create_object(id=98, name="Bar8", is_in_scene=True)
            b8 = self.bar8.getPosition()
            # b8.pop(1)
            self.bar_poses.append(b8)
            self.bar9 = self.env.create_object(id=99, name="Bar9", is_in_scene=True)
            b9 = self.bar9.getPosition()
            # b9.pop(1)
            self.bar_poses.append(b9)
            print ("Bar poses: ", self.bar_poses)
            exit(0)
        except:
            print("Running IK.")
            # Generate end-effector goal cube with random position within certain bounds
            # Check and regenerate until goal cube has no collision
            # self.goal = self.create_object(id=100000, name="End Goal Cube", is_in_scene=True)
            # goal_x = round(random.uniform(-0.7755, 0.1158), 5)
            # goal_y = round(random.uniform(0.5730, 0.9500), 5)
            # goal_z = round(random.uniform(0.2872, 1.5320), 5)
            # goal_pos = [goal_x, goal_z]
            # while (math.dist(b1, goal_pos) < 0.12 or 
            #     math.dist(b2, goal_pos) < 0.12 or 
            #     math.dist(b3, goal_pos) < 0.12 or 
            #     math.dist(b4, goal_pos) < 0.12 or 
            #     #math.dist(b5, goal_pos) < 0.12 or 
            #     math.dist(b6, goal_pos) < 0.12 or 
            #     math.dist(b7, goal_pos) < 0.12 or 
            #     math.dist(b8, goal_pos) < 0.12 or 
            #     math.dist(b9, goal_pos) < 0.12):
            #     print("REGENERATING")
            #     goal_x = round(random.uniform(-0.7755, 0.1158), 5)
            #     goal_y = round(random.uniform(0.5730, 0.9500), 5)
            #     goal_z = round(random.uniform(0.2872, 1.5320), 5)
            #     goal_pos = [goal_x, goal_z]
            # self.goal.setTransform([goal_x, goal_y, goal_z])
            # print("\nEnd Goal Position:", [goal_x, goal_y, goal_z])
        # If environment does not exist, run inverse kinematics on random end goal position
        # except:
        #     print("Running Inverse Kinematics.")
            # self.goal = self.create_object(id=100000, name="End Goal Cube", is_in_scene=True)
            # goal_x = random.uniform(-0.7755, 0.1158)
            # goal_y = random.uniform(0.7, 0.9500)
            # goal_z = random.uniform(0.2872, 1.5320)
            # position = [goal_x, goal_y, goal_z]
            # # self.instance_channel.set_action(
            # #     "SetTransform",
            # #     id=100000,
            # #     position=position,
            # #     scale=[0.05, 0.05, 0.05],
            # # )
            # self.inverseKinematics(position, [90, 0, 0])

        # Create overhead camera and collect RGB and depth data
        intrinsic_matrix = [600, 0, 0, 0, 600, 0, 240, 240, 1]
        # self.camera = self.env.create_camera(id=1234, name='example_camera', intrinsic_matrix=intrinsic_matrix, width=680, height=480, fov=57)

        #self.forceCube = self.create_object(id=54321, name="Force Cube", is_in_scene=True)

        # Initialize data collection list (will be converted to npz array later)
        self.ep_data = np.array([])

        self.count = 0
        self.time = 0

        #step the sim 100 times to make sure everything is initalized
        for i in range(100):
            self.env.step()

        init_pos = [180.21, 293.64, 182, 230.15, 359.25, 332.14, 90.87]
        # init_pos = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(init_pos)):
            if init_pos[i] > 180:
                init_pos[i] -= 360
        self.env.robot.setJointPositionsDirectly(init_pos)
        self.env.step()
        # skin = self.skin.getInfo()['forces']
        # skin_ids = self.skin.getInfo()['ids']
        # pos = self.skin.getInfo()['positions']
        # positions = {}
        # for i, idx in enumerate(skin_ids):
        #     positions[idx] = pos[i]
        # # print ("Skin positions: ", self.skin.getInfo()['positions'])
        # robot_state = self.robot.getRobotState()
        
        # def read_yaml(file_path):
        #     with open(file_path, 'r') as file:
        #         return yaml.safe_load(file)

        # def write_yaml(data, file_path):
        #     with open(file_path, 'w') as file:
        #         yaml.dump(data, file)

        # input_file_path = 'taxel_locations.yaml'
        # output_file_path = 'taxel_location_local.yaml'

        # data = read_yaml(input_file_path)
        # for joint_id in taxel_data:
        #     link_id = int(joint_id[-1])
        #     taxels = taxel_data[joint_id]['Taxels']
        #     for idx, taxel in taxels.items():
        #         taxel_id = taxel['ID']
        #         data[joint_id]['Taxels'][idx]['POS'] = (np.array(positions[taxel_id]) - np.array(robot_state['positions'][link_id])).tolist()

        # write_yaml(data, output_file_path)
        # exit(0)
    
    def create_robot(
        self,
        id: int,
        gripper_list: list = None,
        robot_name: str = None,
        urdf_path: str = None,
        base_pos: list = [0, 0, 0],
        base_orn=[0, 0, 0, 1],
    ) -> None:
        """
        Create a robot in the scene
        :param id: robot id
        :param gripper_list: list of gripper ids
        :param robot_type: robot type, str, check robot.py
        :param urdf_path: path to urdf file, needed if robot_type is None
        :param base_pos: base position of the robot (x, y, z) same as unity
        :param base_orn: base orientation of the robot (x, y, z) same as unity
        """
        if urdf_path is None:
            self.env.robot_dict[id] = Robot(
                self.env,
                id=id,
                gripper_id=gripper_list,
                robot_name=robot_name,
                base_pose=base_pos,
                base_orientation=base_orn,
            )
        else:
            self.env.robot_dict[id] = Robot(
                self.env,
                id=id,
                gripper_id=gripper_list,
                robot_name=robot_name,
                urdf_path=urdf_path,
                base_pose=base_pos,
                base_orientation=base_orn,
            )
        this_robot = self.env.robot_dict[id]
        return this_robot

    # Function registered at script termination for saving data collected
    def save_npz(self):
        import datetime
        np.savez(f"ee_trajectory_{datetime.datetime.now()}.npz", self.ep_data)
        print("Data Saved")

    # Function to test latency of force sensors
    def test_latency(self):
        cube_pos = self.forceCube.getPosition()
        self.forceCube.setTransform(position=[0.7430508136749268, 1.1135352849960327, 0.9985001087188721])

    #Get the maximum scale factor for a given mode
    def get_max_scale_factor(self, mode):
        if mode == Mode.ENDEFFECTOR:
            return self.MAX_EE_STEP
        elif mode == Mode.VECTOR:
            return self.MAX_V_SCALE

    #read the joint positions from the sim
    def read_sim(self):
        return self.env.robot.getJointPositions()
    
    #write the joint positions to the sim then step
    def write_sim_joints(self, vector):
        print ("vector: ", vector)
        for i in range(len(vector)):
            if vector[i] > 180:
                vector[i] -= 360
            if vector[i] < -180:
                vector[i] += 360
        self.env.robot.setJointPositionsDirectly(vector)
        self.env._step()
        
    #write the end effector position and rotation to the sim then step
    def write_sim_transform(self, pos, rot):
        self.env.robot.directlyMoveTo(pos, rot)
        self.env._step()

        #Given a pin joint configuration, get the end effector position vector and rotation matrix
    def get_endeffector_pose(self, q):
        # Update the robot model and data
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get the end effector position and rotation
        end_effector_id = self.model.getFrameId("end_effector_link")
        end_effector_pose = self.data.oMf[end_effector_id]
        pos = end_effector_pose.translation
        rot = end_effector_pose.rotation

        return pos, rot

    def __rcare_joints_to_pin__(self, config):
        config = np.radians(config)
        q = pin.neutral(self.model)

        # Convert ROS joint configuration to pin configuration
        for i in range(len(self.model.joints)-1):
            jidx = self.model.getJointId(self.model.names[i + 1])
            qidx = self.model.idx_qs[jidx]
    
            # nqs[i] is 2 for continuous joints _step()in pin
            if self.model.nqs[jidx] == 2:
                q[qidx] = np.cos(config[i])
                q[qidx + 1] = np.sin(config[i])
            else:
                q[qidx] = config[i]

        return q

        #Convert a Pinocchio joint configuration to a ROS joint configuration
    def __pin_joints_to_rcare__(self, q):
        config = np.zeros(len(self.model.joints) - 1)  # Subtract 1 because Pinocchio adds an extra joint for the universe

        for i in range(len(config)):
            jidx = self.model.getJointId(self.model.names[i + 1])
            qidx = self.model.idx_qs[jidx]

            if self.model.nqs[jidx] == 2:
                # For continuous joints, extract angle from sine and cosine components
                config[i] = np.arctan2(q[qidx + 1], q[qidx])
            else:
                config[i] = q[qidx]

        config = np.degrees(config)

        return config

    def __pose_to_joints__(self, pos, rot, initial):
        JOINT_ID = 7
        oMdes = pin.SE3(rot, pos)
        
        #eps should be pretty low since we want an accurate solution or else the velocity will be higher than it should be
        q = self.__rcare_joints_to_pin__(initial)
        eps = 1e-4
        IT_MAX = 8000
        DT = 0.2
        damp = 1e-12

        #Set the bounds for the IK solver
        low = np.array([-np.pi]*11, dtype=np.float64)
        high = np.array([np.pi]*11, dtype=np.float64)

        for i in range(IT_MAX):
            # Update the robot model and data
            pin.forwardKinematics(self.model, self.data, q)
            pin.framesForwardKinematics(self.model, self.data, q)
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Compute the error between the current and desired end effector pose
            frame_id = self.model.getFrameId("end_effector_link")
            dMi = oMdes.actInv(self.data.oMf[frame_id])
            err = pin.log(dMi).vector
            
            #if your failure count is high the ik solver should be tuned more OR a bug has been introduced into the method
            if i >= IT_MAX:
                global failure
                failure += 1
                print ("Failure count: ", failure)
                break
            
            #solution is written to the robot given that the error is acceptable
            if np.linalg.norm(err) < eps:
                # self.write_joints((self.__pin_joints_to_rcare__(q)-inital)/DT)
                pos, rot = self.get_endeffector_pose(q)
                q = self.__pin_joints_to_rcare__(q)
                return q
            
            #compute Jacobian and velocity to find the next joint configuration
            J = pin.getFrameJacobian(self.model,self.data,frame_id,pin.LOCAL)
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * DT)
            q = np.clip(q, low, high)
 
        
        #if the IK solver fails, write a zero velocity to the robot
        #DO NOT DELETE OR__pose_to_joints__ having the robot continously move 
        return initial

    #write to the sim regardless of mode
    def write_backend(self, mode, output):
        if mode == Mode.VECTOR:
            initial = np.array(self.read_sim())
            for x in output:
                finalV = np.add(initial, x)
            initial[initial < 0] += 360
            finalV[finalV < 0] += 360
            finalV = initial + ((finalV - initial) / (np.linalg.norm(finalV - initial) + 0.00001)) * self.MAX_V_SCALE
            self.write_sim_joints(finalV)
        
        if mode == Mode.ENDEFFECTOR:
            # rot = self.robot.getGripperRotation()
            # #TODO: fix this once RCAREWORLD is fixed
            # #the robots position is not actually correct in the sim, so writing to the sim causes a drift in the end effector position
            # pos = self.cube.getPosition()
            # out_pos = output[0]
            # out_rot = output[1]
            # pos[0] += out_pos[0]
            # pos[1] += out_pos[1]
            # pos[2] += out_pos[2]

            # self.write_sim_transform(pos, rot)
            tmp, dif_rot = output
            diff_pos = [tmp[1], -tmp[0], tmp[2]]

            #get joint velocities, convert to pinnochio, and finally get end effector pose
            initial = np.array(self.read_sim())
            #This could be done with ROS and tf 
            pos, rot = self.get_endeffector_pose(self.__rcare_joints_to_pin__(initial))


            #calculate the new position and rotation
            pos += diff_pos
            rot = np.dot(rot, dif_rot)
            
            #write the new position and rotation to the robot
            final = self.__pose_to_joints__(pos, rot, initial)
            print ("Final before scaling: ", final)
            initial[initial < 0] += 360
            final[final < 0] += 360
            final = initial + ((final - initial) / (np.linalg.norm(final - initial) + 0.0000001)) * self.MAX_V_SCALE
            print ("Final after scaling: ", final)
            self.write_sim_joints(final)

    def collectData(self):
        # self.robot.setJointPositionsDirectly(action)
        # self.env.instance_channel.set_action(
        #         "GetRGB", id=1234, width=680, height=480, fov=57
        #     )
        # self.camera.initializeRGB()

        # # Get RGB camera data
        # rgb_img = self.camera.getRGB()
        # rgb_img = np.array(rgb_img)
        
        # # Get DepthEXR camera data
        # self.env.instance_channel.set_action(
        #     "GetDepthEXR",
        #     id=1234,
        #     width=680,
        #     height=480,
        #     fov=57
        # )
        # # self.robot.setJointPositionsDirectly(action)
        # self.camera.initializeDepthEXR()
        # depth_bytes = self.env.instance_channel.data[1234]["depth_exr"]
        # temp_file_path = osp.join(tempfile.gettempdir(), "temp_img.exr")
        # with open(temp_file_path, "wb") as f:
        #     f.write(depth_bytes)
        # depth_exr = cv2.imread(temp_file_path, cv2.IMREAD_UNCHANGED)
        # os.remove(temp_file_path)
        # depth_img = (depth_exr).astype(np.float32)[:,:]    

        # # Get Joint Position data of robot's current pose       
        # joint_pos = self.robot.getJointPositions()      

        # # Get Skin forces data      
        # skin_info = self.skin.getInfo()
        # skin = skin_info['forces']
        # skin_ids = skin_info['ids']
        # skin_pos = skin_info['positions']
        # for i, idx in enumerate(skin_ids):
        #     if idx == 1000:
        #         print ("Taxel ID: ", i)
        #         print ("Position for 1000: ", self.skin.getInfo()['positions'][i])
        # # print ("Skin positions: ", self.skin.getInfo()['positions'])
        
        robot_state = self.env.robot.getRobotState()
        
        pos = robot_state['positions'][7]
        ori = self.env.robot.ik_controller.get_link_states_pybullet(robot_state['joint_positions'])[1][7]
        
        # forces, active_taxel_ids, positions = filter_active_taxels(skin, skin_ids, skin_pos, FORCE_THRESHOLD)
        # pos = {}
        # for i, idx in enumerate(active_taxel_ids):
        #     pos[idx] = positions[i]
        # # active_taxel_ids = skin_ids
        # # forces = skin
        # # tmp = np.array([0.35, -0.031, 1.008])
        # if len(forces) > 0:
        #     for taxel_id, force in zip(active_taxel_ids, forces):
        #         taxel_dict[taxel_id].pos = pos[taxel_id]
        #         skeleton_id = get_skeleton_id(self.bar_poses, taxel_dict[taxel_id].pos)
        #         print ("Skeleton id: ", skeleton_id)
        
        data_entry = {"ee_pos": pos, "ee_ori": ori}
        # # Single data entry for one step        
        # data_entry = {"RGB": rgb_img, "Depth": depth_img, "joint-pos": joint_pos, "skin": skin}    
        self.ep_data = np.append(self.ep_data, data_entry)   
            
#This class abstracts how to update the robot's joint states
#This class should work regardless of the backend 
class Mode(Enum):
    VECTOR = auto()
    ENDEFFECTOR = auto()
    eigenvectors = [[0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0, 0]]

    # eigenvectors = [[0, 1, 0, 0, 0, 0, 0],
    #             [1, 0, /home/rishabh/kortex_description0, 0, 0, 0, 0],
    #             [0, 0, 0, 1, 0, 0.75, 0],
    #             [0, 0, 1, 0, -0.7, 0, 0],
    #             [0, 0, -0.15, 0, 0, 1, 0],]

    #this returns the appropriate mode given the current user input
    def get_mode(curr_input, mode):
        if Input.EE_MODE in curr_input:
            return Mode.ENDEFFECTOR
        elif Input.VECTOR_MODE in curr_input:
                return Mode.VECTOR
        return mode

    #this method is used to calculate the delta between the current state and the desired state, regardless of mode
    def delta(mode, inputs, max_step_factor):
        if mode == Mode.VECTOR:
            return Mode.delta_vector(inputs, max_step_factor)
        else:
            return Mode.delta_ee(inputs, max_step_factor)

    #maps the input to an approriate scale factor
    def map_to_scale(input, max_scale):
        return np.abs(input * max_scale)
             
    #this method is used to calculate the delta between the current state and the desired state, given the mode is ENDEFFECTOR
    def delta_vector(inputs, max_scale):
        output = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]
        
        vectors = np.array(Mode.eigenvectors.value)
        
        #change this to change mapping between input and eigenvector
        for input in inputs:
            
            #if the input is a tuple, the second value mapped to a scale factor. Otherwise, the scale factor is max_scale
            if isinstance(input, tuple):
                scale = Mode.map_to_scale(input[1], max_scale)
                input = input[0]
            else:
                scale = max_scale
            
            if input == Input.FORWARD:
                output[0] = -1 * scale*vectors[0]
            if input == Input.BACKWARD:
                output[0] = 1 * scale*vectors[0]
            if input == Input.RIGHT:
                output[1] = scale*vectors[1]
            if input == Input.LEFT:
                output[1] = -1 * scale*vectors[1]
            if input == Input.UP:
                output[2] = scale*vectors[2]
            if input == Input.DOWN:
                output[2] = -1 * scale*vectors[2]
            if input == Input.ONE:
                output[3] = scale*vectors[3]
            if input == Input.TWO:
                output[3] = -1 * scale*vectors[3]
            if input == Input.THREE:
                output[3] = scale*vectors[4]
            if input == Input.FOUR:
                output[3] = -1 * scale*vectors[4]
        return output
   
   #this method is used to calculate the delta between the current state and the desired state, given the mode is VECTOR
    def delta_ee(inputs, max_step):
        pos_output = [0]*3
        rot_euler = [0]*3

        #change this to change mapping between input and axis/rotation direction
        for input in inputs:
            
            #if the input is a tuple, the second value mapped to a scale factor. Otherwise, the scale factor is max_scale
            if isinstance(input, tuple):
                step = Mode.map_to_scale(input[1], max_step)
                input = input[0]
            else:
                step = max_step

            if input == Input.LEFT:
                pos_output[1] = step
            if input == Input.RIGHT:
                pos_output[1] = -1 * step
            if input == Input.FORWARD:
                pos_output[0] = step
            if input == Input.BACKWARD:
                pos_output[0] = -1 * step
            if input == Input.UP:
                pos_output[2] = step
            if input == Input.DOWN:
                pos_output[2] = -1 * step
            if input == Input.POS_X:
                rot_euler[1] = 3 * step
            if input == Input.NEG_X:
                rot_euler[1] = -3 * step
            if input == Input.POS_Y:
                rot_euler[0] = 3 * step
            if input == Input.NEG_Y:
                rot_euler[0] = -3 * step
            if input == Input.POS_Z:
                rot_euler[2] = 6 * step
            if input == Input.NEG_Z:
                rot_euler[2] = -6 * step
        return pos_output, R.from_rotvec(rot_euler).as_matrix()


#This class abstracts how to get inputs from the user
class Controller:
    def __init__(self):
        self.joystick = None
        self.update_joysticks()

        #change this to change mapping between key and input
        self.keymapping = {
            pygame.K_UP: Input.UP,
            pygame.K_DOWN: Input.DOWN,
            pygame.K_RIGHT: Input.RIGHT,
            pygame.K_LEFT: Input.LEFT,
            pygame.K_f: Input.FORWARD,
            pygame.K_b: Input.BACKWARD,
            pygame.K_1: Input.ONE,
            pygame.K_2: Input.TWO,
            pygame.K_w: Input.POS_Y,
            pygame.K_s: Input.NEG_Y,
            pygame.K_a: Input.NEG_X,
            pygame.K_d: Input.POS_X,
            pygame.K_q: Input.NEG_Z,
            pygame.K_e: Input.POS_Z,
            pygame.K_m: Input.EE_MODE,
            pygame.K_n: Input.VECTOR_MODE
        }

    #sets the controller to use the most recently connected joystick
    def update_joysticks(self):
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        else:
            self.joystick = None

    #reads inputs from the user regardless of controller type
    def read_inputs(self):
        if self.joystick is not None:
            return self._read_joystick_inputs()
        else:
            return self._read_keyboard_inputs()

    #reads inputs from the connected joystick and maps them to inputs
    def _read_joystick_inputs(self):
        left_x = self.joystick.get_axis(0)
        left_y = self.joystick.get_axis(1)
        right_x = self.joystick.get_axis(3)
        right_y = self.joystick.get_axis(4)

        x_pressed = self.joystick.get_button(1)
        a_pressed = self.joystick.get_button(2)

        left_bumper = self.joystick.get_button(4)
        right_bumper = self.joystick.get_button(5)

        lt = (self.joystick.get_axis(2) + 1) / 2
        rt = (self.joystick.get_axis(5) + 1) / 2

        dpad_x, dpad_y = self.joystick.get_hat(0)

        mapped_inputs = []
        threshold = 0.3

        if left_y < -threshold:
            mapped_inputs.append((Input.FORWARD, left_y))
        elif left_y > threshold:
            mapped_inputs.append((Input.BACKWARD, left_y))

        if left_x < -threshold:
            mapped_inputs.append((Input.LEFT, left_x))
        elif left_x > threshold:
            mapped_inputs.append((Input.RIGHT, left_x))
           
        if right_y < -threshold:
            mapped_inputs.append((Input.UP, right_y))
        elif right_y > threshold:
            mapped_inputs.append((Input.DOWN, right_y))
         
        if right_x < -threshold:
            mapped_inputs.append((Input.TWO, right_x))
        elif right_x > threshold:
            mapped_inputs.append((Input.ONE, right_x))     

        if dpad_x < 0:
            mapped_inputs.append(Input.NEG_X)
        elif dpad_x > 0:
            mapped_inputs.append(Input.POS_X)
        
        if dpad_y < 0:
            mapped_inputs.append(Input.NEG_Y)
        elif dpad_y > 0:
            mapped_inputs.append(Input.POS_Y)
        
        if x_pressed:
            mapped_inputs.append(Input.POS_Z)  
        if a_pressed:
            mapped_inputs.append(Input.NEG_Z)  

        if left_bumper:
            mapped_inputs.append(Input.VECTOR_MODE)
        if right_bumper:
            mapped_inputs.append(Input.EE_MODE)

        if lt > threshold:
            mapped_inputs.append(Input.THREE)
        if rt > threshold:
            mapped_inputs.append(Input.FOUR)

        return mapped_inputs

    #reads inputs from the keyboard, maps it to corresponding input, and filters it besided on valid inputs
    def _read_keyboard_inputs(self):
        keys = pygame.key.get_pressed()
        mapped_inputs = []

        for key, input_value in self.keymapping.items():
            if keys[key]:
                mapped_inputs.append(input_value)

        return mapped_inputs


class Teleoperate:
    #initialize pygame, the backend, how the user controls the robot, and control mode
    def __init__(self, backendType, mode):
        pygame.init()
        pygame.display.set_mode((300, 300))
        
        self.backendType = backendType
        self.mode = mode
        self.controller = Controller()
        self.last_time = time.perf_counter()

        if self.backendType == Backend.SIM:
            self.backend = RCare_Interface()
        else:
            self.backend = ROS_Interface()
        
        self.freq_time = time.perf_counter()
        
    #update loop for teleop
    def update(self):
        #write to robot at 40hz if not in sim
        while time.perf_counter() - self.last_time > 0.01 or self.backendType == Backend.SIM:
            self.last_time = time.perf_counter()
            
            for event in pygame.event.get():
                #stop program if user closes window, make sure robot is stopped given not in sim
                if event.type == pygame.QUIT:
                    if self.backendType == Backend.REAL:
                        self.backend.write_joints([0,0,0,0,0,0,0])
                    pygame.quit()
                    sys.exit()

                #update joysticks if user adds or removes one
                if event.type == pygame.JOYDEVICEADDED or event.type == pygame.JOYDEVICEREMOVED:
                    self.controller.update_joysticks()

            #calculate delta from user input
            curr_input = self.controller.read_inputs()
            self.mode = Mode.get_mode(curr_input, self.mode)  
            delta = Mode.delta(self.mode, curr_input, self.backend.get_max_scale_factor(self.mode))

            self.backend.collectData()
            #write delta to the backend
            if self.backendType is Backend.REAL and self.backend.current_joint_states is not None:
                self.backend.write_backend(self.mode, delta)
            elif self.backendType is Backend.SIM:
                self.backend.write_backend(self.mode, delta)
            
            print ("Frequency: ", 1/(time.perf_counter() - self.freq_time))
            self.freq_time = time.perf_counter()


# # Example of how to use an instance of Teleop class
t = Teleoperate(Backend.SIM, Mode.ENDEFFECTOR)
while True:
    t.update()
