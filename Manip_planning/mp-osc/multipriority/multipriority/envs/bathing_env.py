from pyrcareworld.envs import RCareWorld
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from multipriority.agents import UnityRobot, RealRobot
from multipriority.utils import *
import math
import rospy

try:
    from multipriority_ros.srv import FeedbackService, FeedbackServiceResponse
except:
    print ("Multipriority ROS package not found")

from pyrcareworld.utils.skeleton_visualizer import SkeletonVisualizer


class SkeletonEnv(RCareWorld):
    def __init__(
        self,
        executable_file: str = None,
        scene_file: str = None,
        custom_channels: list = [],
        assets: list = [],
        **kwargs
    ):
        RCareWorld.__init__(
            self,
            executable_file=executable_file,
            scene_file=scene_file,
            custom_channels=custom_channels,
            assets=assets,
            **kwargs,
        )

        # Create robot object
        self.robot = self.create_robot(
            id=315892,
            gripper_list=[3158920],
            robot_name="kinova_gen3_7dof-robotiq85",
            base_pos=[0, 0, 0],
        )
        
        self.init_pose_obj = self.create_object(6666, "Ini", is_in_scene=True)
        ini_world_pose = self.init_pose_obj.getPosition()
        ini_world_rot = self.init_pose_obj.getQuaternion()
        self.robot.moveTo(ini_world_pose, ini_world_rot)
        self.skin = self.create_skin(id=114514, name="Skin", is_in_scene=True)
        self.visualizer = SkeletonVisualizer()
        
        # Create overhead camera and collect RGB and depth data
        intrinsic_matrix = [600, 0, 0, 0, 600, 0, 240, 240, 1]
        self.camera = self.create_camera(id=1234, name='example_camera', intrinsic_matrix=intrinsic_matrix, width=680, height=480, fov=57)


        rospy.init_node('feedback_server')
        s = rospy.Service('feedback_watcher', FeedbackService, self.receiving)
        print("Ready to receive feedback.")
        torque_limit = load_yaml("torque_limit.yaml")
        self.torque_limit = np.array([torque_limit[key] for key in torque_limit.keys()])
        self.feedback_received = False
        self.feedback = None

    def receiving(self, req):
        self.feedback = req.input_data
        
        print ("Feedback received: ", self.feedback)
        self.feedback_received = True
        return FeedbackServiceResponse("success")
    
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
            self.robot_dict[id] = UnityRobot(
                self,
                id=id,
                gripper_id=gripper_list,
                robot_name=robot_name,
                base_pose=base_pos,
                base_orientation=base_orn,
            )
        else:
            self.robot_dict[id] = UnityRobot(
                self,
                id=id,
                gripper_id=gripper_list,
                robot_name=robot_name,
                urdf_path=urdf_path,
                base_pose=base_pos,
                base_orientation=base_orn,
            )
        this_robot = self.robot_dict[id]
        return this_robot

    # Function for moving robot arm using inverse kinematics to position pos and rotation rot
    def inverseKinematics(self, pos, rot):
        self.instance_channel.set_action(
            "GripperOpen",
            id=3158920,
        )
        self.instance_channel.set_action(
            "IKTargetDoMove",
            id=315892,
            position=[pos[0], pos[1], pos[2]],
            duration=0.1,
            speed_based=False,
        )
        self.instance_channel.set_action(
            "IKTargetDoRotate",
            id=315892,
            vector3=rot,
            duration=0.1,
            speed_based=False,
        )

    def step(self):
        pose = self.init_pose_obj.getPosition()
        rot = self.init_pose_obj.getQuaternion()
        self.robot.moveTo(pose, rot)
        skin_info = self.skin.getInfo()

        force_on_skeleton = {i: 0 for i in range(7)}
        for i in range(len(skin_info["skeleton_ids"])):
            skeleton_id = skin_info["skeleton_ids"][i]
            if skeleton_id == -1:
                continue

            if skeleton_id not in force_on_skeleton:
                force_on_skeleton[skeleton_id] = 0
            force_on_skeleton[skeleton_id] += skin_info["forces"][i]
          
        if (not hasattr(self, "prev_force_on_skeleton") or force_on_skeleton != self.prev_force_on_skeleton):
          print("Forces along IDs are now:", force_on_skeleton)
          self.visualizer.update(force_on_skeleton)

        self.prev_force_on_skeleton = force_on_skeleton
        self._step()
        
    def demo(self):
        for i in range(10000000):
            self.step()
    
    def start_visualizer(self):
        self.visualizer.show()


if __name__ == "__main__":
    env = SkeletonEnv()
    env.start_visualizer()
    env.demo()