import os
import pathlib
import pyrcareworld
import time
from pyrcareworld.objects import RCareWorldBaseObject
from pyrfuniverse.utils.stretch_controller import RFUniverseStretchController
from pyrfuniverse.utils.controller import RFUniverseController
from pyrfuniverse.utils.jaco_controller import RFUniverseJacoController
from pyrfuniverse.utils.ur5_controller import RFUniverseUR5Controller
import pyrfuniverse.utils.rfuniverse_utility as utility
import numpy as np
from pyrr import quaternion
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from .osc_controller import PBAgent

from multipriority.utils import Taxel, load_yaml

def adaptive_damping(J, min_singular_value_threshold=0.01, max_damping=0.1):
    """
    Dynamically adjusts the damping factor based on the smallest singular value of the Jacobian.
    Parameters:
    - J: The Jacobian matrix.
    - min_singular_value_threshold: The threshold for singularity detection.
    - max_damping: The maximum allowable damping factor.
    
    Returns:
    - damping_factor: The adjusted damping factor.
    """
    U, singular_values, Vh = np.linalg.svd(J, full_matrices=False)
    min_singular_value = singular_values[-1]
    if min_singular_value < min_singular_value_threshold:
        damping_factor = min_singular_value_threshold / (min_singular_value + 1e-6)
        return min(damping_factor, max_damping)  # Cap the damping factor to prevent over-damping
    else:
        return 0.01  # Default small damping factor

def smooth_damping(min_singular_value, min_threshold, max_threshold, max_damping):
    """
    Smoothly adjusts the damping factor as the robot approaches singularity.
    Parameters:
    - min_singular_value: The smallest singular value of the Jacobian.
    - min_threshold: Threshold below which damping is applied.
    - max_threshold: Threshold beyond which maximum damping is applied.
    - max_damping: Maximum damping factor.
    
    Returns:
    - damping_factor: A smoothly varying damping factor.
    """
    if min_singular_value < min_threshold:
        return max_damping
    elif min_threshold <= min_singular_value <= max_threshold:
        return max_damping * (1 - (min_singular_value - min_threshold) / (max_threshold - min_threshold))
    else:
        return 0.01  # Default small damping factorss

class UnityRobot(RCareWorldBaseObject):
    def __init__(
        self,
        env,
        id: int,
        gripper_id: list,
        robot_name: str,
        urdf_path: str = None,
        base_pose: list = [0, 0, 0],
        # TODO: Verify taxel normals
        base_orientation: list = [0, 0, 0, 1],
        is_in_scene: bool = False,
        **kwargs
    ):
        super().__init__(env=env, id=id, name=robot_name, is_in_scene=is_in_scene)
        """
        Initialization function.
        @param env: The environment object
        @param id: ID of this robot
        @param gripper_id:  A list of IDs of the grippers. For bimanual manipulator there might be two grippers, so this is a list.
        @param hand_camera: Whether there is a camera on the hand
        """
        self.gripper_id = gripper_id
        self.hand_camera = False
        self.is_mobile = False
        self.first = True

        self.base_pose = base_pose
        self.base_orientation = base_orientation

        self.robot_name = robot_name
        self.robot_type = robot_name.split("-")[0]

        # TODO: Update UnityRobot to have the same link mass on server side
        self.robot_type_dict = {
            "franka": "Franka/panda.urdf",
            "kinova_gen3_6dof": "",
            "kinova_gen3_7dof": "GEN3_URDF_V12.urdf",
            "kinova_gen3_7dof_real": "gen3_7dof_vision_with_skin.urdf",
            "kinova_gen3_7dof_sim": "gen3_7dof_vision_with_skin_sim.urdf",
            # "kinova_gen3_7dof": "kinova_gen3/gen3_7dof_vision.urdf",
            "jaco_6dof": "",
            "jaco_7dof": "Jaco/j2s7s300_gym.urdf",
            "stretch": "Stretch/stretch_uncalibrated.urdf",
            "ur5": "UR5/ur5_robot.urdf",
        }
        self.urdf_path_prefix = os.path.join(
            pathlib.Path(os.path.dirname(__file__)).parents[1], "urdfs/"
        )
        urdf_path = os.path.join(
            self.urdf_path_prefix, self.robot_type_dict[self.robot_type]
        )
        print ("urdf_path: ", urdf_path)
        robot_prefix = self.robot_type.split("_")[0]
        self.ik_controller = PBAgent(
            robot_urdf=urdf_path,
            base_pos=self.base_pose,
            base_orn=self.base_orientation,
            render=True,
        )

        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.link_pos, self.link_ori = None, None
        self.current_ee_pos, self.current_ee_ori = None, None
        self.taxels = []
        self.taxel_forces = None
        
        # TODO: move ctrl variables to a better place
        self.prevErr = None
        self.prev_time = None
        self.integral_err = 0
        self.raw_dt = []
        self.ee_idx = 8
        self.sensor_threshold = 0.1
        
        taxel_config = load_yaml(kwargs.get("taxel_cfg", "real_taxel_data_v2.yaml"))
        for tid in taxel_config:
            taxel = taxel_config[tid]
            # TODO: Check this max force for each taxel. Safety!
            maxSensingForce = 25
            # TODO: Verify taxel ID
            self.taxels.append(Taxel(tid - 1000, taxel['Link'] + 1, taxel['Position'], taxel['Normal'], maxSensingForce))
        # sort self.taxels by taxel id
        self.taxels = sorted(self.taxels, key=lambda x: x.taxel_id)
        print ("Len taxels: ", len(self.taxels))

    def getInfo(self):
        """
        Get the information of this robot.
        @return: A dictionary containing the information of this robot
        """
        return {"id": self.id, "name": self.robot_name, "gripper_id": self.gripper_id}

    def getNumJoints(self):
        """
        Return number of movable joints
        @return: number of movable joints
        """
        num_joints = self.env.instance_channel.data[self.id][
            "number_of_moveable_joints"
        ]
        return num_joints

    def getRobotState(self) -> dict:
        """
        Returns a dictionary containing detailed information about the robot's current status.
        The information includes:
        - Robot name
        - Position, rotation, and quaternion for both global and local frames
        - Local to world transformation matrix
        - Number of joints and their respective positions and rotations
        - Local positions and rotations of the joints
        - Velocities of the joints and robot
        - Joint positions and velocities for the moveable joints
        - Stability and movement status of the robot

        Returns:
            dict: A dictionary containing the robot's current status, with keys such as 'name',
                  'position', 'rotation', 'quaternion', 'local_position', 'local_rotation',
                  'local_quaternion', 'local_to_world_matrix', 'number_of_joints', 'positions',
                  'rotations', 'local_positions', 'local_rotations', 'velocities',
                  'number_of_moveable_joints', 'joint_positions', 'joint_velocities',
                  'all_stable', 'move_done', 'rotate_done'.
        """
        info = self.env.instance_channel.data[self.id]
        # add link positions and orientations
        return info

    def getPBWorldToLinkTransforms(self, positions) -> list:
        link_tr_pb, link_ori_pb = self.ik_controller.get_link_states_pybullet(positions)
        world_to_link_trs = []
        for i in range(len(link_tr_pb)):
            pb_world_to_link_tr = np.eye(4)
            pb_world_to_link_tr[:3, :3] = np.asarray(link_ori_pb[i]).reshape(3,3)
            pb_world_to_link_tr[:3, 3] = link_tr_pb[i]
            world_to_link_trs.append(pb_world_to_link_tr)
        return world_to_link_trs

    def getJointStates(self, joint_id: int) -> dict:
        """
        Returns a dictionary containing detailed information about the robot's current status.
        The information includes:
        - Joint position
        - Joint velocity
        - Joint force

        Args:
            joint_id (int): The ID of the joint to get the state of.

        Returns:
            dict: A dictionary containing the joint's current status, with keys such as 'joint_position',
                    'joint_velocity', 'joint_force'.
        """
        assert (
            joint_id
            < self.env.instance_channel.data[id]["number_of_moveable_joints"]
            is True
        ), "Joint ID should be less than the number of movable joints"
        info = {}
        robot_info = self.getRobotState()
        info["joint_positions"] = robot_info["joint_positions"][joint_id]
        info["joint_velocity"] = robot_info["joint_velocities"][joint_id]
        if "drive_forces" in robot_info.keys():
            info["joint_forces"] = robot_info["drive_forces"][joint_id]
        return info

    def getJointPositions(self) -> list:
        """
        Returns a list containing the positions of all the joints.

        Returns:
            list: A list containing the positions of all the joints.
        """
        robot_info = self.getRobotState()
        return robot_info["joint_positions"]

    def getJointVelocities(self) -> list:
        """
        Returns a list containing the velocities of all the joints.

        Returns:
            list: A list containing the velocities of all the joints.
        """
        robot_info = self.getRobotState()
        return robot_info["joint_velocities"]

    def getJointForces(self):
        """
        Returns a list containing the forces of all the joints.

        Returns:
            list: A list containing the forces of all the joints.
        """
        joint_dynamics_forces = self.env.instance_channel.set_action(
            "GetJointInverseDynamicsForce", id=self.id
        )
        # if self.first:
        #     # self.env._step()
        #     self.first = False
        robot_info = self.getRobotState()
        return (
            robot_info["drive_forces"],
            robot_info["gravity_forces"],
            robot_info["coriolis_centrifugal_forces"],
        )

    def getJointAccelerations(self):
        """
        Returns a list containing the accelerations of all the joints.

        Returns:
            list: A list containing the accelerations of all the joints.
        """
        joint_acccelerations = self.env.instance_channel.set_action(
            "GetJointAccelerations", id=self.id
        )
        return joint_acccelerations

    def getJointPositionByID(self, joint_id: int) -> float:
        """
        Returns the current position of the joint.

        Args:
            joint_id (int): The ID of the joint to get the position of.

        Returns:
            float: The position of the joint.
        """
        assert (
            joint_id
            < self.env.instance_channel.data[id]["number_of_moveable_joints"]
            is True
        ), "Joint ID should be less than the number of movable joints"
        robot_info = self.getRobotState()
        return robot_info["joint_positions"][joint_id]

    def getJointVelocityByID(self, joint_id: int) -> float:
        """
        Returns the current velocity of the joint.

        Args:
            joint_id (int): The ID of the joint to get the velocity of.

        Returns:
            float: The velocity of the joint.
        """
        assert (
            joint_id
            < self.env.instance_channel.data[id]["number_of_moveable_joints"]
            is True
        ), "Joint ID should be less than the number of movable joints"
        robot_info = self.getRobotState()
        return robot_info["joint_velocities"][joint_id]

    def getJointForceByID(self, joint_id: int):
        """
        Returns the current force of the joint.

        Args:
            joint_id (int): The ID of the joint to get the force of.

        Returns:
            float: The force of the joint.
        """
        assert (
            joint_id
            < self.env.instance_channel.data[id]["number_of_moveable_joints"]
            is True
        ), "Joint ID should be less than the number of movable joints"

        joint_dynamics_forces = self.env.instance_channel.set_action(
            "GetJointInverseDynamicsForce", id=self.id
        )
        self.env._step()

        robot_info = self.getRobotState()
        return (
            robot_info["drive_forces"][joint_id],
            robot_info["gravity_forces"][joint_id],
            robot_info["coriolis_centrifugal_forces"][joint_id],
        )

    def getJointAccelerationByID(self, joint_id: int):
        """
        Returns the current acceleration of the joint.

        Args:
            joint_id (int): The ID of the joint to get the acceleration of.

        Returns:
            float: The acceleration of the joint.
        """
        assert (
            joint_id
            < self.env.instance_channel.data[id]["number_of_moveable_joints"]
            is True
        ), "Joint ID should be less than the number of movable joints"

        joint_acccelerations = self.env.instance_channel.set_action(
            "GetJointAccelerations", id=self.id
        )
        return joint_acccelerations[joint_id]

    def getGripperPosition(self) -> list:
        """
        Returns the current position of the gripper.

        Returns:
            list: The position of the gripper.
        """
        if len(self.gripper_id) == 1:
            # print(self.env.instance_channel.data[self.gripper_id[0]])
            return self.env.instance_channel.data[self.gripper_id[0]]["position"]

    def getGripperRotation(self) -> list:
        """
        Returns the current rotation of the gripper.

        Returns:
            list: The rotation of the gripper.
        """
        if len(self.gripper_id) == 1:
            return self.env.instance_channel.data[self.gripper_id[0]]["rotation"]

    def getGripperVelocity(self) -> list:
        """
        Returns the current velocity of the gripper.

        Returns:
            list: The velocity of the gripper.
        """
        if len(self.gripper_id) == 1:
            return self.env.instance_channel.data[self.gripper_id[0]]["velocities"][0]
    
    def getGripperGraspPointPosition(self) -> list:
        """
        Returns the current position of the gripper grasp point.
        """
        gripper_data = self.env.instance_channel.data[self.id]["positions"][-1]
        # print(gripper_data)
        return gripper_data

    def getGripperGraspPointRotation(self) -> list:
        """
        Returns the current position of the gripper grasp point.
        """
        gripper_data = self.env.instance_channel.data[self.id]["rotations"][-1]
        # print(gripper_data)
        return gripper_data

    def setJointPositions(self, joint_positions: list, speed_scales=None) -> None:
        """
        @param joint_positions: list of joint positions, starting from base, in degree TODO need check
        @param speed_scales: A list inferring each joint's speed scale.
        @return: does not return anything
        """
        if speed_scales is not None:
            self.env.instance_channel.set_action(
                "SetJointPosition",
                id=self.id,
                joint_positions=list(joint_positions),
                speed_scales=list(speed_scales),
            )
        if speed_scales is None:
            self.env.instance_channel.set_action(
                "SetJointPosition",
                id=self.id,
                joint_positions=list(joint_positions),
            )

    def setJointPositionsDirectly(self, joint_positions: list) -> None:
        """
        @param joint_positions: list of joint positions, starting from base, in degree TODO need check
        @return: does not return anything
        """
        self.env.instance_channel.set_action(
            "SetJointPositionDirectly", id=self.id, joint_positions=joint_positions
        )

    def setJointPositionsContinue(
        self, joint_positions: list, interval: int, time_joint_positions: int
    ) -> None:
        """
        @param joint_positions: list of joint positions, starting from base, in degree TODO need check
        @return:
        """
        self.env.instance_channel.set_action(
            "SetJointPositionContinue",
            id=self.id,
            interval=interval,
            time_joint_positions=time_joint_positions,
        )

    def setJointVelocity(self, joint_velocities: list) -> None:
        self.env.instance_channel.set_action(
            "SetJointVelocity", id=self.id, joint_velocitys=joint_velocities
        )

    def setJointForces(self, joint_forces: list) -> None:
        """

        @param joint_forces:
        @return:
        """
        self.env.instance_channel.set_action(
            "AddJointForce", id=self.id, joint_forces=joint_forces
        )

    def setJointTorques(self, joint_torques: list) -> None:
        """

        @param joint_torques:
        @return:
        """
        # print ("Mode: relative")
        # print ("Mode: global")
        self.env.instance_channel.set_action(
            "AddJointTorque", id=self.id, joint_torques=joint_torques
        )

    def setJointForcesAtPositions(
        self, joint_forces: list, force_positions: list
    ) -> None:
        """

        @param jiont_forces:
        @param force_positions:
        @return:
        """
        self.env.instance_channel.set_action(
            "AddJointForceAtPosition",
            id=self.id,
            joint_forces=joint_forces,
            forces_position=force_positions,
        )

    def setImmovable(self, status) -> None:
        """

        @return:
        """
        self.env.instance_channel.set_action("SetImmovable", id=self.id, immovable=status)

    def moveTo(self, targetPose: list, targetRot=None, currentPos=None) -> None:
        if targetRot != None:
            joint_positions = self.ik_controller.calculate_ik_recursive(
                targetPose, targetRot, currentPos
            )
        else:
            joint_positions = self.ik_controller.calculate_ik_recursive(targetPose, currentPos=currentPos)
        self.setJointPositions(joint_positions)
        self.env._step()
        
    def moveToCompliant(self, targetPose: list, targetRot=None, currentPos=None, currentVel=None) -> None:
        if targetRot != None:
            joint_positions = self.ik_controller.calculate_ik_recursive(
                targetPose, targetRot, currentPos
            )
        else:
            joint_positions = self.ik_controller.calculate_ik_recursive(targetPose)

        Kp = np.array([190] * 7)
        Kd = 18 * np.ones(7)
        currentPos = self.ik_controller.get_pybullet_joint_pos_from_unity(currentPos)           
        joint_positions = self.ik_controller.get_pybullet_joint_pos_from_unity(joint_positions)
        # joint_positions[joint_positions > np.pi] = joint_positions - 2 * np.pi

        currentVel = np.array(currentVel)
        print ("Current: ", currentPos)
        print ("Tgt: ", joint_positions)
        q_delta = (joint_positions - currentPos) % (2 * np.pi)
        for i in range(7):
            if q_delta[i] > np.pi:
                q_delta[i] -= 2 * np.pi
            if q_delta[i] < -np.pi:
                q_delta[i] += 2 * np.pi
        print ("Delta: ", q_delta)
        print ("Current vel: ", currentVel)
        # q_delta = q_delta / np.linalg.norm(q_delta)
        tau = Kp * q_delta + Kd * currentVel
        return tau   

    def directlyMoveTo(self, targetPose: list, targetRot: list = None, currentPos=None) -> None:
        if targetRot is not None:
            joint_positions = self.ik_controller.calculate_ik_recursive(
                targetPose, targetRot, currentPos
            )
        else:
            joint_positions = self.ik_controller.calculate_ik_recursive(targetPose)
        
        self.setJointPositionsDirectly(joint_positions)
        self.env._step()

    def BioIKMove(self, targetPose: list, duration: float, relative: bool) -> None:
        self.env.instance_channel.set_action(
            "IKTargetDoMove",
            id=self.id,
            position=targetPose,
            duration=duration,
            relative=relative,
        )
        self.env._step()
        # while not self.env.instance_channel.data[self.id]["move_done"]:
        #     self.env._step()

    def BioIKRotateQua(
        self, taregetEuler: list, duration: float, relative: bool
    ) -> None:
        self.env.instance_channel.set_action(
            "IKTargetDoRotateQuaternion",
            id=self.id,
            quaternion=utility.UnityEularToQuaternion(taregetEuler),
            duration=duration,
            relative=relative,
        )
        self.env._step()
        while (
            not self.env.instance_channel.data[self.id]["move_done"]
            or not self.env.instance_channel.data[self.id]["rotate_done"]
        ):
            self.env._step()

    def closeGripper(self) -> None:
        print("Make sure the gripper is Robotiq 2F")
        self.env.instance_channel.set_action(
            "SetJointPosition", id=self.gripper_id[0], joint_positions=[30, 30]
        )
        self.env._step()

    def updateRobotState(self, joint_positions, joint_velocities):
        self.ik_controller.ros_set_joint_positions(joint_positions)
        self.current_joint_positions = joint_positions
        self.current_joint_velocities = joint_velocities
        self.link_pos, self.link_ori, _, __= self.ik_controller.ros_get_link_states_pybullet(self.current_joint_positions)
        self.current_ee_pos, self.current_ee_ori = np.array(self.link_pos[self.ee_idx]), np.array(self.link_ori[self.ee_idx])

    def updateSkinState(self, taxel_forces):
        self.taxel_forces = taxel_forces
        for i in range(len(taxel_forces)):
            self.taxels[i].contactF = taxel_forces[i]
        # TODO: Remove later. HACK!
        # self.taxels[12].contactF = taxel_forces[13]
        # print ("Taxel forces: ", taxel_forces[12])

    def oscPoseControl(self, linkIdx, desPos, desOri, desVel, Kp, Kd, Ki, KpO, KdO, Vmax):
        def saturate(y):
            for i in range(len(y)):
                if np.abs(y[i]) >1:
                    y[i] = 1
            return y

        currentJointPos = self.current_joint_positions
        currentJointVel = self.current_joint_velocities

        Jee, Jr = self.ik_controller.calc_jacobian(linkIdx, [0, 0, 0], currentJointPos, currentJointVel, [0, 0, 0, 0, 0, 0, 0])
        J = np.concatenate((Jee, Jr), axis=0)
        damping_factor = adaptive_damping(J)
        print ("Pose damping factor: ", damping_factor)
        if damping_factor > 0.05:
            Kp = 2 * Kp
        # damping_factor_pos = damping_factor[0:3]
        # damping_factor_ori = damping_factor[3:6]
        currentEEPos, currPBOri = self.current_ee_pos, self.current_ee_ori
        currentEEVel = np.dot(Jee, currentJointVel)
        if currentEEVel.shape[0] == 1:
            currentEEVel = currentEEVel.reshape(-1, 1)

        lamda = Kp/Kd
        xtilda = currentEEPos - desPos
        # if np.linalg.norm(xtilda) < 0.002:
        #     xtilda = np.zeros_like(xtilda, dtype=np.float64)

        # using velocity saturation
        # des_x_dd = -Kd*(currentEEVel + saturate(Vmax/(lamda*np.abs(xtilda)))*lamda*xtilda)
        
        # without velocity saturation
        # TODO: Verify Kd sign
        des_x_dd = -Kp * (xtilda) - Kd * currentEEVel
        if des_x_dd.shape[0] == 1:
            des_x_dd = des_x_dd.reshape(-1, 1)
        
        if self.prevErr is None:
            self.prev_time = time.time()
            self.prevErr = -Kp * xtilda * 0.02
        
        # TODO: Use tracker cutoff value here
        if np.linalg.norm(xtilda) < 0.02:
            # use prevErr for integral term
            self.prevErr = (self.prevErr - Kp * xtilda) * 0.02
            # TODO: Move pose integral gain to function arguments
            iterm =  Ki * self.prevErr
            des_x_dd += iterm.reshape(-1, 1)
            # clip self.prevErr to 
        elif np.linalg.norm(xtilda) < 0.01:
            self.prevErr = np.zeros_like(xtilda)

        Mq = self.ik_controller.ros_calc_Mq(currentJointPos)
        Mq_inv = np.linalg.pinv(Mq)
        Jee_T = np.transpose(Jee)
        Mxee1 = np.linalg.pinv(np.matmul(Jee, np.matmul(Mq_inv, Jee_T)) + (damping_factor)**2 * np.eye(3)) #  + (0.01)**2 * np.eye(3)
        internal = np.matmul(Mxee1, des_x_dd)
        # Damped pseudo-inverse for Jee
        J_damped_inv = Jee_T @ np.linalg.pinv(Jee @ Jee_T + (damping_factor)**2 * np.eye(3))
        u_pos = np.matmul(Jee_T, internal)
        
        ###########
        currRotMat = np.array(currPBOri)
        desRotMat = desOri

        err = np.linalg.norm(Rotation.from_matrix(np.dot(currRotMat, desRotMat.T)).as_rotvec())

        currQuat = Rotation.from_matrix(currRotMat).as_quat()
        desQuat = Rotation.from_matrix(desRotMat).as_quat()
        currQuatConj = quaternion.conjugate(currQuat)
        q_r = quaternion.cross(desQuat, currQuatConj)
        q_r /= np.linalg.norm(q_r)
        ori_err = q_r[0:3] * np.sign(q_r[3])
        des_x_dd = KpO * ori_err
        
        Jr_T = np.transpose(Jr)
        Mxee2 = np.linalg.pinv(np.matmul(Jr, np.matmul(Mq_inv, Jr_T)) + (damping_factor)**2 * np.eye(3))
        internal = np.transpose(np.matmul(Mxee2, des_x_dd))
        u_ori = np.matmul(Jr_T, internal)
        ##########
        
        # J = Jee
        lambda_full_inv = np.dot(np.dot(J, np.linalg.inv(Mq)), J.transpose())
        lambda_full = np.linalg.pinv(lambda_full_inv)
        Jbar = np.dot(Mq_inv, J.transpose()).dot(lambda_full)
        filter = np.eye(J.shape[-1], J.shape[-1]) - np.dot(Jbar, J)
        
        u = u_pos + u_ori - np.dot(filter, KdO * currentJointVel).reshape(-1, 1)
        # u = Mq_inv @ u
        # u = np.clip(u, -10, 10)
        # print ("Filter: ", filter.shape)
        return np.array(u), np.array(filter)
    
    def oscContactControl(self, normalVec, Kp, Kd, Ki, desF, contactF, localPos, linkID, time_scale = 1.0):
        if contactF < 0.05:
            self.prevErr = None

        currentJointPos = self.current_joint_positions
        currentJointVel = self.current_joint_velocities
        currPBTrans, currPBOri = self.link_pos, self.link_ori

        # TODO: Fix this computation later
        world_to_link_tr = currPBOri[linkID]
        print ("Contact Kp: ", Kp)
        # normalVec = world_to_link_tr @ normalVec
        normalVec_ = np.dot(np.linalg.inv(world_to_link_tr), normalVec)
        # print("Contact Vector: ", contactF * normalVec)

        error = (desF - contactF)
        # if np.abs(error) < 1:
        #     error = 0.0
        # print ("Axiswise error: ", error * np.array(normalVec))
        des_ctrl = np.sign(error) * Kp * min(np.abs(error), 60) * np.array(normalVec_)
        print ("Des ctrl shape: ", des_ctrl.shape)
        
        # Calculate dt
        current_time = time.time()
        if self.prev_time is None:
            self.prev_time = current_time  # Initialize if it doesn't exist
        dt = 0.02 # current_time - self.prev_time
        self.prev_time = current_time

        # # Calculate
        # if np.abs(error) < 5.0 and np.abs(error) > 1:
        #     self.integral_err += error * np.array(normalVec) * dt #0.02
        #     integral_ctrl = Ki * self.integral_err
        #     des_ctrl += integral_ctrl
        # if np.abs(error) < 1:
        #     self.integral_err = np.zeros_like(self.integral_err, dtype=np.float64)

        if self.prevErr is None:
            self.prevErr = -error
        else:
            print ("Curr err: ", error, " prevErr: ", self.prevErr)
            diff = (error - self.prevErr)
            norm = np.linalg.norm(diff)
            diff = diff * np.min((norm, 1)) / norm #np.sign((error - self.prevErr)) * np.min([np.abs((error - self.prevErr)), 1])
            derivative = diff / dt
            des_ctrl += Kd * derivative * np.array(normalVec)
            print ("Kd osc: ", Kd)
            print ("Diff: ", (error - self.prevErr))
            self.prevErr = error

        Jt, Jr = self.ik_controller.ros_calc_contact_jacobian(linkID, localPos, currentJointPos, [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0])
        damping_factor = adaptive_damping(Jt)
        print ("Contact damping factor: ", damping_factor)
        damping_factor = 0.01
        Mq = self.ik_controller.ros_calc_Mq(currentJointPos)
        Mq_inv = np.linalg.pinv(Mq)
        Jt = np.concatenate((Jt, Jr), axis=0)
        Jt_T = np.transpose(Jt)
        # Jt_T = np.transpose(J)
        # Mxee = np.linalg.pinv(np.matmul(Jt, np.matmul(np.linalg.pinv(Mq), Jt_T)) + (damping_factor)**2 * np.eye(3))
        # Mxee_inv = np.matmul(Jt, np.matmul(np.linalg.pinv(Mq), Jt_T))
        des_ctrl = np.concatenate((des_ctrl, np.zeros(3,)))
        
        # internal = np.transpose(np.matmul(Mxee, des_ctrl.reshape(-1, 1)))
        # des_ctrl = internal
        # des_ctrl = np.matmul(Jt_T, internal.reshape(-1, 1))
        
        # K_d = np.array([1, 1,, 1]
        K_d = np.eye(3)
        np.fill_diagonal(K_d, Kd)
        
        # damping_term = K_d * Jt * np.array(currentJointVel).reshape(-1, 1)
        des_ctrl = des_ctrl.reshape(-1, 1)
        # des_ctrl += damping_term.reshape(-1, 1)
        
        u = np.matmul(Jt_T, des_ctrl)

        Mxee_null = np.linalg.pinv(np.matmul(Jt, np.matmul(Mq_inv, Jt.T)))
        Jpseudo = np.matmul(Mxee_null, np.matmul(Jt, np.linalg.pinv(Mq)))
        JJ = np.matmul(Jt.T, Jpseudo)
        n = JJ.shape[0]
        filter = np.identity(n) - JJ
        
        # u = Mq_inv @ u

        return u, filter
    
    def reset(self) -> None:
        self.ik_controller.reset()

# Inherit UnityRobot to create a RealRobot class with differen osc_ee_control and osc_contact_control functions
class RealRobot(UnityRobot):
    def __init__(
        self,
        env,
        robot_name: str,
        urdf_path: str = None,
        sensor_threshold: float = 1,
        base_pose: list = [0, 0, 0],
        base_orientation: list = [0, 0.0, 0.0, 1],
        **kwargs
    ):
        super().__init__(
            env=env,
            id=None,
            gripper_id=None,
            robot_name=robot_name,
            urdf_path=urdf_path,
            base_pose=base_pose,
            base_orientation=base_orientation,
        )
        # TODO: Add this to taxel config instead
        self.sensor_threshold = sensor_threshold # for real skin sensor
        # TODO: add this to runner/env/robot config
        taxel_config = load_yaml(kwargs.get("taxel_cfg", "real_taxel_data_v2.yaml"))
        for tid in taxel_config:
            taxel = taxel_config[tid]
            # TODO: Check this max force for each taxel. Safety!
            maxSensingForce = 25
            self.taxels.append(Taxel(tid - 1000, taxel['Link'], taxel['Position'], taxel['Normal'], maxSensingForce))
        # sort self.taxels by taxel id
        self.taxels = sorted(self.taxels, key=lambda x: x.taxel_id)

    def oscPoseControl(self, linkIdx, desPos, desOri, desVel, Kp, Kd, Ki, KpO, KdO, Vmax):
        def saturate(y):
            for i in range(len(y)):
                if np.abs(y[i]) >1:
                    y[i] = 1
            return y

        currentJointPos = self.current_joint_positions
        currentJointVel = self.current_joint_velocities
        Jee, Jr = self.ik_controller.calc_jacobian(linkIdx, [0, 0, 0], currentJointPos, currentJointVel, [0, 0, 0, 0, 0, 0, 0]) 
        currentEEPos, currentEEOri = self.current_ee_pos, self.current_ee_ori
        currentEEVel = np.dot(Jee, currentJointVel)
        # print ("Curr EE vel: ", currentEEVel)
        # print ("Kp: ", Kp, " Kd: ", Kd)

        # lamda = Kp/Kd
        xtilda = currentEEPos - desPos
        # cap xtilda magnitude to 0.03
        xtilda = (xtilda/np.linalg.norm(xtilda)) * min(np.linalg.norm(xtilda), 0.1)
        
        # using velocity saturation
        # des_x_dd = -Kd*(currentEEVel + saturate(Vmax/(lamda*np.abs(xtilda)))*lamda*xtilda)
        
        # without velocity saturation
        Kp = np.array(Kp).reshape(-1)
        Kd = np.array(Kd).reshape(-1)
        currentEEVel = np.array(currentEEVel).reshape(-1)
        des_x_dd = -Kp * (xtilda) - Kd * currentEEVel
        
        if self.prevErr is None:
            self.prev_time = time.time()
            self.prevErr = -Kp * xtilda * 0.001
        else:
            # use prevErr for integral term
            self.prevErr = self.prevErr - Kp * xtilda * 0.001
            # TODO: Move pose integral gain to function arguments
            des_x_dd += Ki * self.prevErr
            # clip self.prevErr to 
            if np.linalg.norm(xtilda) < 0.01:
                self.prevErr = np.zeros_like(xtilda)

        des_x_dd = des_x_dd.reshape(-1, 1)
        
        Mq = self.ik_controller.ros_calc_Mq(currentJointPos)
        Mq_inv = np.linalg.pinv(Mq)
        Jee_T = np.transpose(Jee)
        Mxee1 = np.linalg.pinv(np.matmul(Jee, np.matmul(Mq_inv, Jee_T)))
        internal = np.matmul(Mxee1, des_x_dd)
        u_pos = np.matmul(Jee_T, internal)
        
        #########
        currRotMat = np.array(currentEEOri)
        desRotMat = np.array(desOri)

        err = np.linalg.norm(Rotation.from_matrix(np.dot(currRotMat, desRotMat.T)).as_rotvec())
        
        currQuat = Rotation.from_matrix(currRotMat).as_quat()
        desQuat = Rotation.from_matrix(desRotMat).as_quat()
        currQuatConj = quaternion.conjugate(currQuat)
        q_r = quaternion.cross(desQuat, currQuatConj)
        q_r /= np.linalg.norm(q_r)
        ori_err = q_r[0:3] * np.sign(q_r[3])
        des_x_dd = KpO * ori_err
        
        Jr_T = np.transpose(Jr)
        Mxee2 = np.linalg.pinv(np.matmul(Jr, np.matmul(Mq_inv, Jr_T)))
        internal = np.transpose(np.matmul(Mxee2, des_x_dd))
        u_ori = np.matmul(Jr_T, internal)
        ##########

        
        J = np.concatenate((Jee, Jr), axis=0)
        lambda_full_inv = np.dot(np.dot(J, np.linalg.inv(Mq)), J.transpose())
        lambda_full = np.linalg.pinv(lambda_full_inv)
        Jbar = np.dot(Mq_inv, J.transpose()).dot(lambda_full)
        filter = np.eye(J.shape[-1], J.shape[-1]) - np.dot(Jbar, J)
        
        u = u_pos + u_ori - np.dot(filter, KdO * np.dot(Mq, currentJointVel)).reshape(-1, 1)
        # Mq_inv = np.eye(7)
        # # Mq_inv.diagonal = 1.0 / np.array([0.3, 0.3, 0.3, 0.3, 0.18, 0.18, 0.2])
        # np.fill_diagonal(Mq_inv, 1.0 / np.array([0.39, 0.39, 0.39, 0.39, 0.2, 0.2, 0.2]))
        u = Mq_inv @ u
        # print ("U shape: ", u.shape)
        return np.array(u), np.array(filter)

    def oscContactControl(self, normalVec, Kp, Kd, Ki, desF, contactF, localPos, linkID, time_scale = 1.0):
        if contactF < 0.05:
            self.prevErr = None

        currentJointPos = self.current_joint_positions
        currentJointVel = self.current_joint_velocities
        # TODO: Temp fix. Update Config.
        linkID += 1

        currPBTrans, currPBOri = self.link_pos, self.link_ori
        world_to_link_tr = currPBOri[linkID]
        normalVec = world_to_link_tr @ normalVec
        # print("Contact Vector: ", contactF * normalVec)

        error = (desF - contactF)
        # print ("Axiswise error: ", error * np.array(normalVec))
        des_ctrl = 1 * np.sign(error) * Kp * min(np.abs(error), 10) * np.array(normalVec)
        
        # Calculate dt
        current_time = time.time()
        if self.prev_time is None:
            self.prev_time = current_time  # Initialize if it doesn't exist
        dt = 0.02 # current_time - self.prev_time
        self.prev_time = current_time

        # Calculate
        if np.abs(error) < 5.0 and np.abs(error) > 1:
            self.integral_err += error * np.array(normalVec) * dt #0.02
            integral_ctrl = Ki * self.integral_err
            des_ctrl += integral_ctrl

        Jt, Jr = self.ik_controller.ros_calc_contact_jacobian(linkID, localPos, currentJointPos, currentJointVel, [0, 0, 0, 0, 0, 0, 0])
        Mq = self.ik_controller.ros_calc_Mq(currentJointPos)
        Mq_inv = np.linalg.pinv(Mq)
        # J = np.concatenate((Jt, Jr), axis=0)
        Jt_T = np.transpose(Jt)
        # Jt_T = np.transpose(J)
        Mxee = np.linalg.pinv(np.matmul(Jt, np.matmul(np.linalg.pinv(Mq), Jt_T)))
        # Mxee_inv = np.matmul(Jt, np.matmul(np.linalg.pinv(Mq), Jt_T))
        # des_ctrl = np.concatenate((des_ctrl, np.zeros(3,)))
        
        # internal = np.transpose(np.matmul(Mxee, des_ctrl.reshape(-1, 1)))
        # u = np.matmul(Jt_T, internal.reshape(-1, 1))
        
        # K_d = np.array([1, 1,, 1]
        K_d = np.eye(3)
        np.fill_diagonal(K_d, Kd)
        
        damping_term = K_d * Jt * np.array(currentJointVel).reshape(-1, 1)
        des_ctrl = des_ctrl.reshape(-1, 1)
        # des_ctrl += damping_term.reshape(-1, 1)
        
        u = np.matmul(Jt_T, des_ctrl)

        Jpseudo = np.matmul(Mxee, np.matmul(Jt, np.linalg.pinv(Mq)))
        JJ = np.matmul(Jt_T, Jpseudo)
        n = JJ.shape[0]
        filter = np.identity(n) - JJ
        
        u = Mq_inv @ u

        return u, filter