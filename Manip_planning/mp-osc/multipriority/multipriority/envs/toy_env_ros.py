import rospy
from .toy_env_sim import ToyOneEnv
import numpy as np
import pybullet as p
import math
import random
import cv2
import tempfile
import sys
import os
import os.path as osp
import open3d as o3d
import atexit
import time
import pygame

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

try:
    from multipriority_ros.srv import FeedbackService, FeedbackServiceResponse
except:
    print ("Multipriority ROS package not found")

from multipriority.agents import RealRobot

class ToyOneEnvROS(ToyOneEnv):
    def __init__(
        self,
        executable_file: str = None,
        scene_file: str = None,
        custom_channels: list = [],
        assets: list = [],
        **kwargs
    ):
        ToyOneEnv.__init__(
            self,
            executable_file=executable_file,
            scene_file=scene_file,
            custom_channels=custom_channels,
            assets=assets,
            **kwargs,
        )

        rospy.init_node('feedback_server')
        s = rospy.Service('feedback_watcher', FeedbackService, self.receiving)
        print("Ready to receive feedback.")
        self.feedback_received = False
        self.feedback = None

    def receiving(self, req):
        self.feedback = req.input_data
        
        print ("Feedback received: ", self.feedback)
        self.feedback_received = True
        return FeedbackServiceResponse("success")
    

class RealToyEnv:
    def __init__(self, **kwargs):
        self.robot = self.create_robot(robot_name="kinova_gen3_7dof_real-robotiq85", sensor_threshold=2.0)
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        self.skin_sub = rospy.Subscriber("/calibration", Float32MultiArray, self.skin_callback)
    
    def create_robot(self, robot_name, sensor_threshold=1):
        return RealRobot(
            self,
            robot_name=robot_name,
            sensor_threshold=sensor_threshold,
        )
    
    def joint_state_callback(self, data):
        self.robot.updateRobotState(data.position[1:], data.velocity[1:])

    def skin_callback(self, data):
        self.robot.updateSkinState(data.data)

    def get_obs(self):
        if self.robot.current_joint_positions is None:
            print("\n[INFO]: Waiting for joint state publisher.\n")
            time.sleep(1)
            return None
        if self.robot.taxel_forces is None:
            print("\n[INFO]: Waiting for skin data publisher.\n")
            time.sleep(1)
            return None

        obs = {}
        obs['robot'] = {'joint_positions': self.robot.current_joint_positions, 'joint_velocities': self.robot.current_joint_velocities, 'positions':self.robot.link_pos, 'orientations':self.robot.link_ori}
        obs['skin'] = {'forces': self.robot.taxel_forces}
        
        # TODO: Add logic to update taxel skeleton id. For now skeleton id is 0
        return obs

    def get_skeleton_id(self, taxel):
        # TODO: Implement this
        return 0

