import os
from multipriority.envs.toy_env_sim import ToyOneEnv
import pygame
import numpy as np
import yaml
import pathlib
import tqdm
import rospy

from multipriority.controllers import MultiPriorityController, TaskController, ContactController
from multipriority.utils import *
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int16

class DigitalTwin:
    def __init__(self):
        self.env = ToyOneEnv()
        self.env_base = self.env.create_object(id=100, name="envBase", is_in_scene=True)
        self.env_base_sub = rospy.Subscriber('/natnet_ros/EnvBase/pose', PoseStamped, self.env_base_sub_cb)
        self.env_base_pose = np.array([0., 0., 0.])

        self.robot_base = self.env.create_object(id=101, name="robotBase", is_in_scene=True)
        self.robot_base_sub = rospy.Subscriber('/natnet_ros/ManipulatorBase/pose', PoseStamped, self.robot_base_sub)
        self.robot_base_pose = np.array([10., 10., 10.])

        self.cylinder_offset = np.array([0, -0.6, 0])
        self.cylinder_subs = []
        self.cylinder_poses = []
        self.cylinder_bases = []
        for i in range(8):
            self.cylinder_subs.append(rospy.Subscriber(f'/natnet_ros/Cylinder0{i+1}/pose', PoseStamped, eval(f'self.cyl{i}_sub')))
            self.cylinder_poses.append(np.array([0., 0., 0.]))
            self.cylinder_bases.append(self.env.create_object(id=102+i, name=f"cylBase{i}", is_in_scene=True))

        init_pos = [90.21, 293.64, 182, 230.15, 359.25, 332.14, 90.87]
        self.pub = rospy.Publisher('/counter', Int16, queue_size=1)
        # init_pos = [0] * 7
        for i in range(len(init_pos)):
            if init_pos[i] > 180:
                init_pos[i] -= 360

        for i in range(100):
            self.env.robot.setJointPositionsDirectly(init_pos)
            self.env._step()

    def env_base_sub_cb(self, msg):
        pose = msg.pose.position
        self.env_base_pose[0] = -pose.y
        self.env_base_pose[1] = pose.z # 0.6
        self.env_base_pose[2] = pose.x
        
    def robot_base_sub(self, msg):
        pose = msg.pose.position
        self.robot_base_pose[0] = -pose.y
        self.robot_base_pose[1] = pose.z
        self.robot_base_pose[2] = pose.x

    def cyl0_sub(self, msg):
        pose = msg.pose.position
        self.cylinder_poses[0][0] = -pose.y
        self.cylinder_poses[0][1] = pose.z 
        self.cylinder_poses[0][2] = pose.x

    def cyl1_sub(self, msg):
        pose = msg.pose.position
        self.cylinder_poses[1][0] = -0.037 + pose.y
        self.cylinder_poses[1][1] = pose.z 
        self.cylinder_poses[1][2] = pose.x + 0.005

    def cyl2_sub(self, msg):
        pose = msg.pose.position
        self.cylinder_poses[2][0] = -0.025 + pose.y
        self.cylinder_poses[2][1] = pose.z 
        self.cylinder_poses[2][2] = pose.x + 0.015

    def cyl3_sub(self, msg):
        pose = msg.pose.position
        self.cylinder_poses[3][0] = -pose.y
        self.cylinder_poses[3][1] = pose.z 
        self.cylinder_poses[3][2] = pose.x

    def cyl4_sub(self, msg):
        pose = msg.pose.position
        self.cylinder_poses[4][0] = -pose.y
        self.cylinder_poses[4][1] = pose.z 
        self.cylinder_poses[4][2] = pose.x

    def cyl5_sub(self, msg):
        pose = msg.pose.position
        self.cylinder_poses[5][0] = -0.040 + pose.y
        self.cylinder_poses[5][1] = pose.z 
        self.cylinder_poses[5][2] = pose.x + 0.021

    def cyl6_sub(self, msg):
        pose = msg.pose.position
        self.cylinder_poses[6][0] = -0.045 + pose.y
        self.cylinder_poses[6][1] = pose.z  
        self.cylinder_poses[6][2] = pose.x #+ 0.015

    def cyl7_sub(self, msg):
        pose = msg.pose.position
        self.cylinder_poses[7][0] = -pose.y
        self.cylinder_poses[7][1] = pose.z 
        self.cylinder_poses[7][2] = pose.x

    def step(self):
        obs = {}
        # obs['robot'] = self.env.robot.getRobotState()
        ##############################################
        self.env_base.setTransform(position=(self.env_base_pose - self.robot_base_pose).tolist())
        for i in range(8):
            self.cylinder_bases[i].setTransform(position=(self.cylinder_poses[i] - self.robot_base_pose + self.cylinder_offset).tolist())

        init_pos = [90.21, 293.64, 182, 230.15, 359.25, 332.14, 90.87]
        # init_pos = [0] * 7
        for i in range(len(init_pos)):
            if init_pos[i] > 180:
                init_pos[i] -= 360
        self.env.robot.setJointPositionsDirectly(init_pos)
        # target = env.create_object(id=315867, name="Cube", is_in_scene=True)
        # target.setTransform(position=env.robot.ik_controller.get_unity_pos_from_bullet(obs['robot']['positions'][-1]))
        # target.setTransform(obs['robot']['positions'][-1], obs['robot']['rotations'][-1])
        # max_force = 0
        self.env._step()

if __name__ == "__main__":
    rospy.init_node('dt_node')
    print("Ready to create Digital Twin")
    twin = DigitalTwin()
    while True:
        twin.step()