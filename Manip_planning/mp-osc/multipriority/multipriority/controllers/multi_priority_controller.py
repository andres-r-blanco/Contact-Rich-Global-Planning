from multipriority.agents import UnityRobot, RealRobot
from multipriority.bandit import ContextualBandit
from .base_controllers import ContactController, TaskController
from multipriority.utils import *
import numpy as np

MAX_CONTACTS = 10

class MultiPriorityController:
    def __init__(self, contact_params, initial_priority, robot: UnityRobot, task_controller: TaskController) -> None:
        self.robot = robot
        self.contact_controllers = {}
        self.task_controller = task_controller
        self.priority = initial_priority
        self.active_controllers = [-1]
        self.last_priority = None
        self.max_contacts = MAX_CONTACTS
        self.buffer_len = 300
        self.buffer = dict(zip(range(self.buffer_len), [None]*self.buffer_len))
        self.retract_states = {}
        self.contact_params = contact_params
        self.logging_freq = 65
        self.contact_err = 0
        self.contact_err_axes = [0, 0, 0]
        self.contactF = 0
        self.use_ctrl = True

    def update(self, obs, add_gravity=False, test_contact=False):
        cmd = np.zeros((7, 1))
        cmd_out = np.zeros((7, 1))
        filter = np.eye(7)
        for controller in self.active_controllers:
            # if object type is ContactController then send osc_contact_control fn as argument
            if isinstance(controller, ContactController):
                print ("\n\nContact Controller running!\n\n")
                cmd, filter_new = controller.update(self.robot.oscContactControl)
            else:
                cmd, filter_new = controller.update(self.robot.oscPoseControl)
            # print ("OG command: ", cmd_out)
            # print ("Filtered command: ", filter @ cmd_out)
            # print ("Filter new: ", filter_new[0])
            # print ("Filter: ", filter[0])
            # print ("Filter: ", filter)
            # print ("Filtered command: ", filter @ cmd_out)
            # print ("Command: ", cmd)
            cmd_out = cmd + filter_new @ cmd_out

        if add_gravity:
            gq = self.robot.ik_controller.calc_gq(self.robot.current_joint_positions, self.robot.current_joint_velocities, [0, 0, 0, 0, 0, 0, 0])
            cmd_out = cmd_out + np.array(gq).reshape(-1, 1)

        return cmd_out

    def retract(self, taxel_feedback):
        self.robot.setJointPositionsDirectly(self.retract_states[taxel_feedback])
        return self.retract_states[taxel_feedback]
    
    def update_buffer(self, i, obs):
        if (len(self.buffer.keys()) >= self.buffer_len):
            dict_ids = np.array(list(self.buffer.keys()))
            if len(dict_ids) > 0:
                del self.buffer[np.min(dict_ids)]
            
        self.buffer[i] = obs['robot']['joint_positions']
        
    def update_retract_dict(self, i, taxel_ids):
        for taxel_id in taxel_ids:
            if taxel_id not in self.retract_states.keys():
                self.retract_states[int(taxel_id)] = self.buffer[np.min(np.array(list(self.buffer.keys())))]
    
