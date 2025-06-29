from multipriority.agents import UnityRobot, RealRobot
from multipriority.utils import Taxel
from scipy.spatial.transform import Rotation as R
import numpy as np
from multipriority.utils import load_yaml

class ContactController:
    def __init__(self, taxel: Taxel) -> None:
        self.taxel = taxel
        self.skeleton_id = None
        self.robot = None
        self.Kp = None
        self.Kd = None
        self.Ki = None
        self.time_scale = None
        self.desF = None
        self.maxF = None
        self.contact_err = None
        self.contact_err_axes = None
        self.contactF = None
        self.time_scale = 1
    
    def set_desired_force(self, null=True):
        self.desF = self.taxel.skeleton_params['goalF']
        self.maxF = self.taxel.skeleton_params['max_force']
        # TODO: create a skeleton part class and attach the max force attributes etc. to it
        # Then everytime taxel makes contact, just update the skeleton_in_contact member variable for that taxel
        if null:
            self.Kp = np.array(self.taxel.skeleton_params['Kp_null'])
            self.Kd = np.array(self.taxel.skeleton_params['Kd_null']) #* time_scale
        else:
            self.Kp = np.array(self.taxel.skeleton_params['Kp'])
            self.Kd = np.array(self.taxel.skeleton_params['Kd']) #* time_scale
        self.Ki = np.array(self.taxel.skeleton_params['Ki']) #/time_scale
        print ("Kp: ", self.Kp)

    def update_contact_err(self):
        contactF, normalVec, localPos, linkID = self.taxel.get_contact_info()
        if self.desF == 0:
            self.contact_err = np.linalg.norm(contactF - self.desF)
        else:
            self.contact_err = np.abs(contactF - self.desF) # /self.desF
        self.contact_err_axes = (contactF  - self.desF) * np.array(self.taxel.normal)

    def update(self, controller_fn):
        contactF, normalVec, localPos, linkID = self.taxel.get_contact_info()
        self.contactF = contactF
        print ("Contact force: ", contactF)
        print ("Desired force: ", self.desF)
        print ("Error: ", self.desF - contactF)
            
        cmd, filter = controller_fn(normalVec, self.Kp, self.Kd, self.Ki, self.desF, contactF, localPos, linkID, self.time_scale)
        
        if self.desF == 0:
            self.contact_err = np.linalg.norm(contactF - self.desF)
        else:
            self.contact_err = np.linalg.norm(contactF - self.desF)
        self.contact_err_axes = (contactF  - self.desF) * np.array(self.taxel.normal)
        
        return cmd, filter

class TaskController:
    def __init__(self, robot, cfg_name) -> None:
        task_params = load_yaml(cfg_name)
        self.robot = robot
        self.goal_position = None
        self.goal_orientation = None
        self.goal_force = None
        self.Kp = np.array(task_params['Kp']) # [120, 120, 120])
        self.Kd = np.array(task_params['Kd']) # [10,10,10])
        self.Ki = np.array(task_params['Ki']) # [0, 0, 0])
        self.Kp_ori = np.array(task_params['Kp_ori']) # [400, 400, 400])
        self.Kd_ori = np.array(task_params['Kd_ori']) # [10, 10, 10])
        self.Vmax = None
        self.Kp_scheduled = None
        self.Kd_scheduled = None
        self.gain_sched_threshold = None

    def set_goal_pose(self, position, orientation):
        self.goal_position = position
        self.goal_orientation = orientation

    def set_controller_gains(self, Kp, Kd, Kp_ori=None, Kd_ori=None):
        self.Kp = Kp
        self.Kd = Kd
        self.Kp_ori = Kp_ori
        self.Kd_ori = Kd_ori
        
    def set_velocity_limit(self, Vmax):
        self.Vmax = Vmax

    def set_goal_force(self, goal_force):
        self.goal_force = goal_force

    def update(self, controller_fn):
        # compute task space command
        desVel = [0, 0, 0]
        cmd, filter = controller_fn(self.robot.ee_idx, self.goal_position, self.goal_orientation, desVel, self.Kp, self.Kd, self.Ki, self.Kp_ori, self.Kd_ori, self.Vmax)
        return cmd, filter

    def get_state_error(self):
        curr_pos_state = self.robot.current_ee_pos
        curr_ori_state = self.robot.current_ee_ori
        
        pos_err = np.linalg.norm(np.array(self.goal_position) - np.array(curr_pos_state))
        # relative_quaternion = R.from_quat(curr_quat_state).inv() * R.from_quat(self.goal_orientation)
        # rot_err = 2 * np.arccos(relative_quaternion.as_quat()[3])
        
        currRotMat = np.array(curr_ori_state)
        desRotMat = self.goal_orientation

        rot_err = np.linalg.norm(R.from_matrix(np.dot(currRotMat, desRotMat.T)).as_rotvec())
        
        if rot_err > np.pi:
            rot_err = 2 * np.pi - rot_err
        return [pos_err, rot_err]
    
    def get_state_error_axes(self):
        """A function to get the state error in the axes of the robot

        Args:
            obs (dict): observation dictionary

        """
        curr_pos_state = self.robot.curr_ee_pos
        
        err = np.array(self.goal_position) - np.array(curr_pos_state)
        
        x_err = err[0]
        y_err = err[1]
        z_err = err[2]
        
        return (x_err, y_err, z_err)
