import pybullet as p
import pybullet_data
import numpy as np
from pyrcareworld.utils import PoseUtils
import math
from scipy.spatial.transform import Rotation as R

## copied over kinova_controller and modified

class PBAgent:
    """
    RFUniverseController is a class to generate robot arm joint states. In simulation environment, we mostly
    want to specify the 6DoF of a joint, then the robot arm will automatically move to that state. Thus, here
    we use pybullet.calculateInverseKinematics() to generate joint positions based on a given robot arm, a
    given end-effector joint and a target Cartesian position. The generated joint states will be passed to
    Unity by rfuniverse channels. Besides, this class will also provide functions to align coordinate in Unity
    and in pybullet.
    """

    def __init__(
        self,
        robot_urdf,
        base_pos=np.array([0, 0, 0]),
        base_orn=[0, 0.0, 0.0, 1],
        init_joint_positions=[0] * 7,
        render=False,
    ):

        if render:
            p.connect(p.GUI)  # For debug mode
        else:
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(0.02)

        self.bullet_client = p
        # TODO: check disabling inertia flag for real robot
        self.bullet_flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | self.bullet_client.URDF_USE_INERTIA_FROM_FILE | self.bullet_client.URDF_MAINTAIN_LINK_ORDER

        self.robot_name = "kinova_gen3_7dof_real"
        self.robot_urdf = robot_urdf
        self.end_effector_id = 9
        self.num_dof = 7
        self.init_joint_positions = self.get_pybullet_joint_pos_from_unity(
            init_joint_positions
        )
        self.bullet_base_pos = np.array(self.get_bullet_pos_from_unity(base_pos))
        self.bullet_base_orn = base_orn
        self.robot = self.bullet_client.loadURDF(
            self.robot_urdf,
            self.bullet_base_pos,
            base_orn,
            useFixedBase=True,
            flags=self.bullet_flags,
        )

        self.link_mass = []
        self.link_inertia = []
        self.center_of_mass = []
        self.revolute_joint_ids = []

        for j in range(9):
            self.bullet_client.changeDynamics(
                self.robot, j, linearDamping=0, angularDamping=0
            )
            info = self.bullet_client.getJointInfo(self.robot, j)

            jointName = info[12].decode('utf-8')
            jointType = info[2]
            if jointType == self.bullet_client.JOINT_REVOLUTE:
                self.revolute_joint_ids.append(j)
        
        print (self.revolute_joint_ids)
        self.move_radius = [0.92, 0.92, 0.92, 0.92, 0.7, 0.7, 0.7]
        self.reset()

    def get_bullet_pos_from_unity(self, unity_pos: list) -> list:
        return [unity_pos[2], -unity_pos[0], unity_pos[1]]

    def get_unity_pos_from_bullet(self, bullet_pos: list) -> list:
        return [-bullet_pos[1], bullet_pos[2], bullet_pos[0]]

    def get_unity_joint_pos_from_pybullet(self, pybullet_joint_pos: tuple) -> list:
        pybullet_joint_pos = list(pybullet_joint_pos)[: self.num_dof]
        for i, (joint_pos) in enumerate(pybullet_joint_pos):
            pybullet_joint_pos[i] = 180 * joint_pos / math.pi

        return pybullet_joint_pos

    def get_pybullet_joint_pos_from_unity(self, unity_joint_pos: list) -> list:
        unity_joint_pos = np.array(unity_joint_pos)[: self.num_dof]
        pybullet_joint_pos = unity_joint_pos * math.pi / 180 % (2 * math.pi)
        pybullet_joint_pos[pybullet_joint_pos > math.pi] -= 2 * math.pi
        pybullet_joint_pos[pybullet_joint_pos < -math.pi] += 2 * math.pi

        return pybullet_joint_pos.tolist()

    def calculate_ik(self, unity_eef_pos, eef_orn=None) -> list:
        if eef_orn is None:
            eef_orn = self.bullet_client.getQuaternionFromEuler(
                [math.pi / 2.0, 0.0, 0.0]
            )

        eef_pos = self.get_bullet_pos_from_unity(unity_eef_pos)

        joint_positions = self.bullet_client.calculateInverseKinematics(
            self.robot, self.end_effector_id, eef_pos, eef_orn, maxNumIterations=20
        )

        for i, (idx) in enumerate(self.revolute_joint_ids):
            self.bullet_client.resetJointState(self.robot, idx, joint_positions[i])

        return self.get_unity_joint_pos_from_pybullet(joint_positions)

    def calculate_ik_recursive(self, unity_eef_pos, eef_orn=None, currentPos=None) -> list:
        if eef_orn is None:
            eef_orn = self.bullet_client.getQuaternionFromEuler(
                [math.pi / 2.0, 0.0, 0.0]
            )
        if currentPos is not None:
            currentPos = self.get_pybullet_joint_pos_from_unity(currentPos)
        eef_orn = -np.array(self.get_bullet_pos_from_unity(np.radians(eef_orn)))
        eef_orn = self.bullet_client.getQuaternionFromEuler(eef_orn.tolist())
        eef_pos = self.get_bullet_pos_from_unity(unity_eef_pos)
        # eef_orn = [0,0,0,1]
        for i in range(20):
            joint_positions = self.bullet_client.calculateInverseKinematics(
                self.robot, self.end_effector_id, eef_pos, jointDamping=[0.1]*7, restPoses=currentPos, residualThreshold=0.01
            )

            for i, (idx) in enumerate(self.revolute_joint_ids):
                self.bullet_client.resetJointState(self.robot, idx, joint_positions[i])
        if np.random.rand() > 0.9:
            noise = np.random.rand(7) * 0.01
            joint_positions += noise
        return self.get_unity_joint_pos_from_pybullet(joint_positions)

    def ros_set_joint_positions(self, joint_positions):
        for i, (idx) in enumerate(self.revolute_joint_ids):
            self.bullet_client.resetJointState(
                self.robot, idx, joint_positions[i]
            )

    # def set_joint_positions(self, joint_positions):
    #     joint_positions = self.get_pybullet_joint_pos_from_unity(joint_positions)
    #     for i, (idx) in enumerate(self.revolute_joint_ids):
    #         self.bullet_client.resetJointState(
    #             self.robot, idx, joint_positions[i]
    #         )

    def get_link_state(self, link_idx):
        link_state = self.bullet_client.getLinkState(self.robot, link_idx)

        return link_state[0], link_state[1]

    # def get_link_state(self, link_idx):
    #     link_state = self.bullet_client.getLinkState(self.robot, link_idx)

    #     return self.get_unity_pos_from_bullet(link_state[0])
    
    def ros_get_link_states_pybullet(self, joint_positions):
        # for i, (idx) in enumerate(self.revolute_joint_ids):
        #     self.bullet_client.resetJointState(
        #         self.robot, idx, joint_positions[i]
        #     )
        link_tr, link_ori = [], []
        local_pos, local_ori = [], []
        for i in range(self.end_effector_id):
            state = self.bullet_client.getLinkState(self.robot, i, computeForwardKinematics=1)
            link_tr.append(state[0])
            # link_ori.append(np.array(p.getMatrixFromQuaternion(state[5])).reshape(3,3))
            link_ori.append(R.from_quat(state[1]).as_matrix())
            local_pos.append(state[2])
            local_ori.append(R.from_quat(state[3]).as_matrix())
        return link_tr, link_ori, local_pos, local_ori
    
    def get_link_states_pybullet(self, joint_positions):
        joint_positions = self.get_pybullet_joint_pos_from_unity(joint_positions)
        for i, (idx) in enumerate(self.revolute_joint_ids):
            self.bullet_client.resetJointState(
                self.robot, idx, joint_positions[i]
            )
        link_tr, link_ori = [], []
        for i in range(self.end_effector_id):
            state = self.bullet_client.getLinkState(self.robot, i)
            link_tr.append(state[0])
            link_ori.append(R.from_quat(state[1]).as_matrix())
        return link_tr, link_ori

    def calc_jacobian(self, link_idx, localPos, currentPos, currentVel, desObjAcc):
        # for i in range(self.num_dof):
        #     p.resetJointState(self.robot, i, currentPos[i])

        # result = p.getLinkState(self.robot, link_idx, computeForwardKinematics=1)
        # link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = result
        
        if not isinstance(currentPos, list):
            currentPos = currentPos.tolist()
        if not isinstance(currentVel, list):
            currentVel = currentVel.tolist()
        
        jac_t, jac_r = p.calculateJacobian(self.robot, link_idx, localPos, currentPos, currentVel, desObjAcc)
        tranJ = np.matrix(jac_t)
        rotJ = np.matrix(jac_r)
        return tranJ, rotJ

    # def calc_jacobian(self, link_idx, localPos, currentPos, currentVel, desObjAcc):
    #     currentPos = self.get_pybullet_joint_pos_from_unity(currentPos).tolist()
    #     for i in range(self.num_dof):
    #         p.resetJointState(self.robot, i, currentPos[i])
        
    #     result = p.getLinkState(self.robot, link_idx, computeForwardKinematics=1)
    #     link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = result
        
    #     jac_t, jac_r = p.calculateJacobian(self.robot, link_idx, [0, 0, 0], currentPos, currentVel, desObjAcc)
        
    #     tranJ = np.matrix(jac_t)
    #     rotJ = np.matrix(jac_r)
    #     return tranJ, rotJ

    def ros_calc_contact_jacobian(self, link_idx, localPos, currentPos, currentVel, desObjAcc):
        if not isinstance(currentPos, list):
            currentPos = currentPos.tolist()
        if not isinstance(currentVel, list):
            currentVel = currentVel.tolist()

        jac_t, jac_r = p.calculateJacobian(self.robot, link_idx, localPos, currentPos, currentVel, desObjAcc)

        tranJ = np.matrix(jac_t)
        rotJ = np.matrix(jac_r)
        return tranJ, rotJ

    def calc_contact_jacobian(self, link_idx, localPos, currentPos, currentVel, desObjAcc):
        currentPos = self.get_pybullet_joint_pos_from_unity(currentPos).tolist()
        jac_t, jac_r = p.calculateJacobian(self.robot, link_idx, localPos, currentPos, currentVel, desObjAcc)

        tranJ = np.matrix(jac_t)
        rotJ = np.matrix(jac_r)
        return tranJ, rotJ

    def ros_calc_Mq(self, currentPos):
        # for i in range(self.num_dof):
        #     p.resetJointState(self.robot, i, currentPos[i])
        # currentPos = self.get_pybullet_joint_pos_from_unity(currentPos).tolist()
        Mq = self.bullet_client.calculateMassMatrix(int(self.robot), currentPos)
        Mq = np.matrix(Mq).tolist()
        return Mq

    def calc_Mq(self, currentPos):
        currentPos = self.get_pybullet_joint_pos_from_unity(currentPos).tolist()
        Mq = self.bullet_client.calculateMassMatrix(int(self.robot), currentPos)
        Mq = np.matrix(Mq).tolist()
        return Mq
    
    def calc_gq(self, currentPos, currentVel, desObjAcc):
        zeros = np.zeros((7,))
        zeros = zeros.tolist()
        if not isinstance(currentPos, list):
            currentPos = currentPos.tolist()
        gq = self.bullet_client.calculateInverseDynamics(self.robot, currentPos, currentVel, zeros)        
        return gq
    
    def reset(self):
        for i, (idx) in enumerate(self.revolute_joint_ids):
            self.bullet_client.resetJointState(
                self.robot, idx, self.init_joint_positions[i]
            )
