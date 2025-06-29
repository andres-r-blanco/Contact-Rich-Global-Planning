from pyrcareworld.envs import RCareWorld
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
import rospy

from multipriority.agents import UnityRobot, RealRobot
from multipriority.utils import *

USE_ROS = False

try:
    from multipriority_ros.srv import FeedbackService, FeedbackServiceResponse
    USE_ROS = True
    
except:
    print ("Multipriority ROS package not found")
    

class ToyOneEnv(RCareWorld):
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
            id=315893,
            gripper_list=[3158930],
            robot_name="kinova_gen3_7dof-robotiq85",
            base_pos=[0, 0, 0],
            base_orn=[0,0,0,1],
        )
        
        # self.viz_robot = self.create_robot(
        #     id=315894,
        #     gripper_list=[],
        #     robot_name="kinova_gen3_7dof_real-robotiq85",
        #     base_pos=[0, 0, 0],
        # )
      
        # Create skin to collect robot taxel data
        self.skin = self.create_skin(id=114514, name="Skin", is_in_scene=True)
        
        self.bar_poses = []
        try:
            self.viz_cube = self.create_object(id=315867, name="Viz Cube", is_in_scene=False)
        except:
            print ("Viz cube not found")
        # Initializing environment objects (softbody bars and end goal cube)
        try:
            # Create bars to get position data to use for goal cube
            self.bar1 = self.create_object(id=102, name="Bar1", is_in_scene=True)
            b1 = self.bar1.getPosition()
            self.bar_poses.append(b1)
            self.bar2 = self.create_object(id=103, name="Bar2", is_in_scene=True)
            b2 = self.bar2.getPosition()
            self.bar_poses.append(b2)
            self.bar3 = self.create_object(id=104, name="Bar3", is_in_scene=True)
            b3 = self.bar3.getPosition()
            self.bar_poses.append(b3)
            self.bar4 = self.create_object(id=105, name="Bar4", is_in_scene=True)
            b4 = self.bar4.getPosition()
            self.bar_poses.append(b4)
            # self.bar5 = self.create_object(id=95, name="Bar5", is_in_scene=True) #camera bar
            # b5 = self.bar5.getPosition()
            # b5.pop(1)
            self.bar6 = self.create_object(id=106, name="Bar6", is_in_scene=True)
            b6 = self.bar6.getPosition()
            self.bar_poses.append(b6)
            self.bar7 = self.create_object(id=107, name="Bar7", is_in_scene=True)
            b7 = self.bar7.getPosition()
            self.bar_poses.append(b7)
            self.bar8 = self.create_object(id=108, name="Bar8", is_in_scene=True)
            b8 = self.bar8.getPosition()
            self.bar_poses.append(b8)
            self.bar9 = self.create_object(id=109, name="Bar9", is_in_scene=True)
            b9 = self.bar9.getPosition()
            self.bar_poses.append(b9)

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
        except:
            print("Running Inverse Kinematics.")
            self.goal = self.create_object(id=100000, name="End Goal Cube", is_in_scene=True)
            goal_x = random.uniform(-0.7755, 0.1158)
            goal_y = random.uniform(0.7, 0.9500)
            goal_z = random.uniform(0.2872, 1.5320)
            position = [goal_x, goal_y, goal_z]
            # self.instance_channel.set_action(
            #     "SetTransform",
            #     id=100000,
            #     position=position,
            #     scale=[0.05, 0.05, 0.05],
            # )
            self.inverseKinematics(position, [90, 0, 0])

        # Create overhead camera and collect RGB and depth data
        intrinsic_matrix = [600, 0, 0, 0, 600, 0, 240, 240, 1]
        self.camera = self.create_camera(id=1234, name='example_camera', intrinsic_matrix=intrinsic_matrix, width=128, height=128, fov=45) # 57

        # Initialize data collection list (will be converted to npz array later)
        self.ep_data = np.array([])

        if USE_ROS:
            rospy.init_node('feedback_server')
            s = rospy.Service('feedback_watcher', FeedbackService, self.receiving)
            print("Ready to receive feedback.")
            
        torque_limit = load_yaml("torque_limit.yaml")
        self.torque_limit = np.array([torque_limit[key] for key in torque_limit.keys()])
        self.feedback_received = False
        self.feedback = None

        self.count = 0
        self.time = 0

        self._step()
    
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
        return UnityRobot(
                self,
                id=id,
                gripper_id=gripper_list,
                robot_name=robot_name,
                base_pose=base_pos,
                base_orientation=base_orn,
            )

    # Function for moving robot arm using inverse kinematics to position pos and rotation rot
    def inverseKinematics(self, pos, rot):
        self.instance_channel.set_action(
            "GripperOpen",
            id=3158930,
        )
        self.instance_channel.set_action(
            "IKTargetDoMove",
            id=315893,
            position=[pos[0], pos[1], pos[2]],
            duration=0.1,
            speed_based=False,
        )
        self.instance_channel.set_action(
            "IKTargetDoRotate",
            id=315893,
            vector3=rot,
            duration=0.1,
            speed_based=False,
        )
        # Collision avoidance?
        # self.instance_channel.set_action(
        #     "AddDistancePoint",
        #     position=[b1[0],20,b1[1]],
        #     radius=0.05,
        # )

    # Function registered at script termination for saving data collected
    def save_npz(self):
        import datetime
        np.savez(f"ee_trajectory_{datetime.datetime.now()}.npz", self.ep_data)
        print("Data Saved")

    # Function to test latency of force sensors
    def test_latency(self):
        cube_pos = self.forceCube.getPosition()
        print(cube_pos)
        self.forceCube.setTransform(position=[0.7430508136749268, 1.1135352849960327, 0.9985001087188721])

    # Function to recieve teleop control keyboard commands and return next position
    def teleop(self):
        # run teleop and recieve action
        zeroVector = [0, 0, 0, 0, 0, 0, 0]
        # zeroVector = [1.5, 1.5, .5, .7, 0.1, 0, 0]
        # self.robot.setJointPositionsDirectly(zeroVector)
        eigenIncrement = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]

        scaling = 2
        eig1 = scaling*np.array([0, 1, 0, 0, 0, 0, 0]) 
        eig2 = scaling*np.array([1, 0, 0, 0, 0, 0, 0]) 
        eig3 = scaling*np.array([0, 0, 1, 0, 1, 0, 1]) 
        eig4 = scaling*np.array([0, 0, 0, 1, 0, 1, 0]) 

        #print("Initial Pose")
        #print(self.robot.getRobotState()["positions"][8])

        mode = 'V' # vector control, change to 'E' if end-effector control

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if pygame.key.get_pressed()[pygame.K_1]:
                    eigenIncrement[0] = eig1
                elif pygame.key.get_pressed()[pygame.K_2]:
                    eigenIncrement[0] = -1 * eig1
                else:
                    eigenIncrement[0] = zeroVector

                if pygame.key.get_pressed()[pygame.K_3]:
                    eigenIncrement[1] = eig2
                elif pygame.key.get_pressed()[pygame.K_4]:
                    eigenIncrement[1] = -1 * eig2
                else:
                    eigenIncrement[1] = zeroVector

                if pygame.key.get_pressed()[pygame.K_5]:
                    eigenIncrement[2] = eig3  
                elif pygame.key.get_pressed()[pygame.K_6]:
                    eigenIncrement[2] = -1 * eig3
                else:
                    eigenIncrement[2] = zeroVector

                if pygame.key.get_pressed()[pygame.K_7]:
                    eigenIncrement[3] = eig4
                elif pygame.key.get_pressed()[pygame.K_8]:
                    eigenIncrement[3] = -1 * eig4
                else:
                    eigenIncrement[3] = zeroVector  

            finalV = np.array(self.robot.getJointPositions())
            for x in eigenIncrement:
                finalV = np.add(finalV, x)
            
            return finalV
        
    def collectData(self, action, steps): #, steps=1):
        time_start = time.perf_counter()
        # self.count = self.count + 1
        self.set_joint_sim(action)
        # Save RGB, Depth, Force, Joint, etc. data to numpy array for every 10 steps
        if self.count % steps == 0:
            # # Get RGB camera data
            # self.camera.initializeRGB()
            # self.instance_channel.set_action(
            #     "GetRGB", id=1234, width=680, height=480, fov=57
            # )
            # rgb_img = self.camera.getRGB()
            # rgb_img = np.array(rgb_img)
            
            # # Get DepthEXR camera data
            # self.camera.initializeDepthEXR()
            # self.instance_channel.set_action(
            #     "GetDepthEXR",
            #     id=1234,
            #     width=680,
            #     height=480,
            # )
            # depth_bytes = self.instance_channel.data[1234]["depth_exr"]
            # temp_file_path = osp.join(tempfile.gettempdir(), "temp_img.exr")
            # with open(temp_file_path, "wb") as f:
            #     f.write(depth_bytes)
            # depth_exr = cv2.imread(temp_file_path, cv2.IMREAD_UNCHANGED)
            # os.remove(temp_file_path)
            # depth_img = (depth_exr).astype(np.float32)[:,:]    

            # Get Joint Position data of robot's current pose       
            joint_state = self.robot.getJointStates()
            pos = joint_state['positions'][7]
            ori = self.env.robot.ik_controller.get_link_states_pybullet(joint_state['joint_positions'])[1][7]     

            # Get Skin forces data      
            skin = self.skin.getInfo()['forces']
            skin_ids = self.skin.getInfo()['ids']
            
            # Single data entry for one step        
            data_entry = {"ee_pos":pos, "ee_ori":ori, "joint_pos": joint_state['joint_positions'], "joint_vel": joint_state['joint_velocities'], "joint_force": joint_state['drive_forces'], "skin": skin, "skin_ids": skin_ids}   
            self.ep_data = np.append(self.ep_data, data_entry)   

            
    def set_joint_sim(self, joints):
        self.robot.setJointTorques(joint_torques = joints)
        return None
    
    def boost_torque(self, torque):
        torque = np.array(torque)
        for i, limit in enumerate(self.torque_limit):
            if torque[i] > limit:
                torque[i] = limit
                
        return [[0, item, 0] for item in torque]  

    # Function run every frame/constantly updating
    def step(self, action): #, action=None):
        action = self.boost_torque(action)
        self.set_joint_sim(action)
        self._step()

    def axiscreator(self, is_joint, bodyId, linkId = -1, unity_local = [0,0,0]):
        bullet_pos = self.robot.ik_controller.get_bullet_pos_from_unity(unity_local)
        # bullet_pos = unity_local

        if linkId == 5:
            x, y, z = bullet_pos
            bullet_pos = [z, -x, -y]
        if is_joint:
            print(f'axis creator at bodyId = {bodyId} and linkId = {linkId} as XYZ->RGB')
            x_axis = self.robot.ik_controller.bullet_client.addUserDebugLine(lineFromXYZ = [0, 0, 0] ,
                                                                    lineToXYZ = [0.1, 0, 0],
                                                                    lineColorRGB = [1, 0, 0] ,
                                                                    lineWidth = 0.1 ,
                                                                    lifeTime = 0 ,
                                                                    parentObjectUniqueId = bodyId ,
                                                                    parentLinkIndex = linkId )

            y_axis = self.robot.ik_controller.bullet_client.addUserDebugLine(lineFromXYZ = [0, 0, 0],
                                                                    lineToXYZ            = [0, 0.1, 0],
                                                                    lineColorRGB         = [0, 1, 0]  ,
                                                                    lineWidth            = 0.1        ,
                                                                    lifeTime             = 0          ,
                                                                    parentObjectUniqueId = bodyId     ,
                                                                    parentLinkIndex      = linkId     )

            z_axis = self.robot.ik_controller.bullet_client.addUserDebugLine(lineFromXYZ          = [0, 0, 0],
                                                                    lineToXYZ            = [0, 0, 0.1],
                                                                    lineColorRGB         = [0, 0, 1]  ,
                                                                    lineWidth            = 0.1        ,
                                                                    lifeTime             = 0          ,
                                                                    parentObjectUniqueId = bodyId     ,
                                                                    parentLinkIndex      = linkId     )
            return [x_axis, y_axis, z_axis]
        else:
            x_axis = self.robot.ik_controller.bullet_client.addUserDebugLine(lineFromXYZ        = bullet_pos ,
                                                                    lineToXYZ = [0.05, 0, 0],
                                                                    lineColorRGB = [1, 0, 0] ,
                                                                    lineWidth = 0.42 ,
                                                                    lifeTime = 0 ,
                                                                    parentObjectUniqueId = bodyId ,
                                                                    parentLinkIndex = linkId )

            y_axis = self.robot.ik_controller.bullet_client.addUserDebugLine(lineFromXYZ          = bullet_pos  ,
                                                                    lineToXYZ            = [0, 0.05, 0],
                                                                    lineColorRGB         = [0, 1, 0]  ,
                                                                    lineWidth            = 0.4       ,
                                                                    lifeTime             = 0          ,
                                                                    parentObjectUniqueId = bodyId     ,
                                                                    parentLinkIndex      = linkId     )

            z_axis = self.robot.ik_controller.bullet_client.addUserDebugLine(lineFromXYZ          = bullet_pos ,
                                                                    lineToXYZ            = [0, 0, 0.05],
                                                                    lineColorRGB         = [0, 0, 1]  ,
                                                                    lineWidth            = 0.4       ,
                                                                    lifeTime             = 0          ,
                                                                    parentObjectUniqueId = bodyId     ,
                                                                    parentLinkIndex      = linkId     )
            return [x_axis, y_axis, z_axis]

    def sub(self,a,b):
        return (np.array(a)-np.array(b)).tolist()

    def calc_taxel(self, expected_id):
        
        base = np.array([0, 0, 0])
        ids = self.skin.getInfo()["ids"]
        print(ids)
        global_positions = self.skin.getInfo()["positions"]
        taxel_global_pybullet = []
        link_global_pybullet = []

        for i in range(len(ids)):
            if ids[i] == 1004:
                print ("Test point unity: ", global_positions[i])
            pos = self.robot.ik_controller.get_bullet_pos_from_unity(np.array(global_positions[i]))
            taxel_global_pybullet += [pos]

            # if id < 1009:
            #     print("origin, ", id, " ", pos)
            #     taxel_global_pybullet[0] += [(id, pos)]
            # elif id < 1018:
            #     print("origin, ", id, " ", pos)
            #     taxel_global_pybullet[1] += [(id, pos)]
            # elif id < 1037:
            #     print("origin, ", id, " ", pos)
            #     taxel_global_pybullet[2] += [(id, pos)]
            # elif id <1056:
            #     print("origin, ", id, " ", pos)
            #     taxel_global_pybullet[3] += [(id, pos)]

        for i in range(2,6):
            link_global_pybullet += [self.robot.ik_controller.bullet_client.getLinkState(self.robot.ik_controller.robot,
                                                                                         i)[0]]
        
        rot_4 = np.array([[0,0,1], [0, -1, 0], [1, 0, 0]])
        pt = taxel_global_pybullet[expected_id-1000]
        print("Test point: ", np.array(pt))

        self.robot.ik_controller.bullet_client.addUserDebugLine(lineFromXYZ        = pt ,
                                                                    lineToXYZ = [pt[0], pt[1], pt[2]+0.05],
                                                                    lineColorRGB = [1, 0, 0] ,
                                                                    lineWidth = 5,
                                                                    lifeTime = 0 )
        # print(pt)

        # point_local = np.array(taxel_global_pybullet[expected_id-1000]) - np.array(link_global_pybullet[-1])
        point_local = np.dot(rot_4.T, pt - np.array(link_global_pybullet[-1]))
        self.robot.ik_controller.bullet_client.addUserDebugLine(lineFromXYZ        = point_local.tolist() ,
                                                                    lineToXYZ = [point_local[0], point_local[1]+0.05, point_local[2]],
                                                                    lineColorRGB = [0, 1, 0] ,
                                                                    lineWidth = 5,
                                                                    lifeTime = 0,
                                                                    parentObjectUniqueId = self.robot.ik_controller.robot,
                                                                    parentLinkIndex=5)
        return point_local

    def test(self):
        all_zeros = [90.21, 293.64, 182, 230.15, 359.25, 332.14, 90.87]
        pose = [90.21, 293.64, 182, 230.15, 359.25, 332.14, 90.87]

        self.robot.setJointPositionsDirectly(all_zeros)
        self._step()
        # poss = self.robot.getRobotState()["positions"]
        # print(poss)
        # print("-----skin data-----")
        # for id, pos in zip(self.skin.getInfo()["ids"], self.skin.getInfo()["positions"]):
        #     # 37-55, 3
        #     # 18-36, 4
        #     # 9-17, 5
        #     # 0-8, 6
        #     if id < 1009:
        #         print("origin, ", id, " ", pos)
        #         pos = self.sub(pos, poss[6])
        #     elif id < 1018:
        #         print("origin, ", id, " ", pos)
        #         pos = self.sub(pos, poss[5])
        #     elif id < 1037:
        #         print("origin, ", id, " ", pos)
        #         pos = self.sub(pos, poss[4])
        #     elif id <1056:
        #         print("origin, ", id, " ", pos)
        #         pos = self.sub(pos, poss[3])
        #     print([id, pos])

        for i in range(2,6):
            self.axiscreator(True, self.robot.ik_controller.robot, i)
        
        pt = self.calc_taxel(1004)
        
        # self.axiscreator(False, self.robot.ik_controller.robot, 5, pt)
        # 1004: [-0.05265629291534424, 0.00917041301727295, -1.1920928955078125e-07]
        # 1006: [-0.00016739964485168457, 0.08052682876586914, 0.03503572940826416]
        # 1007: [-0.04152598977088928, 0.043018341064453125, -0.014114856719970703]
        # self.axiscreator(False, self.robot.ik_controller.robot, 5, [-0.05265629291534424, 0.00917041301727295, -1.1920928955078125e-07])
        #print(self.robot.getRobotState())

        while True:
            # self.robot.setImmovable(True)
            # self.cube.setTransform(position=res.tolist())
            self.robot.setJointPositionsDirectly(pose)
            self._step()
            # print("Robot States")
            # print(self.robot.getRobotState())

    def close(self):
        super().close()

    def simulate(self):
        # initialize teleop display system
        pygame.init()
        display = pygame.display.set_mode((300, 300))
        # keep the simulation running
        for i in range(10000000):
            
            pos, vel = self.robot.getJointPositions(), self.robot.getJointVelocities()
            inc = np.ones(7)
        
            # perform data collection
            
            ## contact control
            Kp = 1
            desF = np.array([5, 0, 0])
            contactF = np.array([0, 0, 0])
            localPos = np.array([0,0,0])
            linkID = 5
        
            action, filter = self.robot.osc_contact_control(pos, vel, Kp, desF, contactF, localPos, linkID)
            
            
            ## end effector control
            desPos = np.array([0.62, 1.23, 0.8])
            # ee_action, filter = np.array(self.robot.osc_ee_control(pos, vel, desPos, np.zeros(3), 10,1))
            # print("torque_grv from ee_control: ", ee_action)
            
            currState = self.robot.getRobotState()
            print("end effector")
            print(currState['positions'][-1])

            # action = action + np.matmul(filter, ee_action)
            
            action = self.boost_torque(action)

            self.step(action)

    def get_obs(self):
        # populate using getRobotState and getSkinInfo
        robot_data = self.robot.getRobotState()
        self.robot.updateRobotState(self.robot.ik_controller.get_pybullet_joint_pos_from_unity(robot_data['joint_positions']), robot_data['joint_velocities'])
        skin_data = self.skin.getInfo()
        sorted_indices = np.argsort(skin_data['ids'])
        sorted_forces = np.array(skin_data['forces'])[sorted_indices]
        self.robot.updateSkinState(sorted_forces)
        obs = {}
        obs['robot'] = {'joint_positions': self.robot.current_joint_positions, 'joint_velocities': self.robot.current_joint_velocities, 'positions':self.robot.link_pos, 'orientations':self.robot.link_ori}
        obs['skin'] = {'forces': self.robot.taxel_forces}
        
        # TODO: Add logic to update taxel skeleton id. For now skeleton id is 0
        return obs


# if __name__ == "__main__":
#     # create the environment
#     env = ToyOneEnv()
#     # run the environment
#     env.simulate()