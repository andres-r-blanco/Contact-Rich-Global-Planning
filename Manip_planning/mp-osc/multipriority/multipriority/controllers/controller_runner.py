from multipriority.agents import UnityRobot, RealRobot
from multipriority.controllers import MultiPriorityController, TaskController, ContactController
from multipriority.envs.toy_env_sim import ToyOneEnv
from multipriority.utils import *
import copy
import numpy as np

class ControllerRunner:
    def __init__(self, priority, sim_obs, force_threshold, taxel_dict, 
                 env: ToyOneEnv, robot, task_controller: TaskController, 
                 multi_controller: MultiPriorityController, tracker: Tracker,
                 mode, soft_threshold, skeleton_contact_params, sorted_indices, trajectory, final_goal_ori, arm_num, goal, manager=None) -> None:
        self.robot = robot
        self.contact_controllers = {}
        self.task_controller = task_controller
        self.priority = priority
        self.obs = sim_obs
        self.multi_controller = multi_controller
        self.env = env
        self.FORCE_THRESHOLD = force_threshold
        self.taxel_dict = taxel_dict
        self.mode = mode
        self.soft_threshold = soft_threshold
        self.tracker = tracker
        self.skeleton_contact_params = skeleton_contact_params
        self.sorted_indices = sorted_indices
        self.bandit_marker = False
        self.trajectory = trajectory
        self.final_goal_ori = final_goal_ori
        self.final_goal = trajectory[-1][0]
        self.arm_num = arm_num
        self.goal = goal
        self.pose_err =  np.linalg.norm(np.array(self.obs['robot']['positions'][7]) - np.array(self.tracker.trajectory[-1][0]))

        if manager is not None:
            self.shared_dict = manager.dict({i: 0 for i in range(self.arm_num)})
        
        # Placeholder for logging outside of the ControllerRunner
        self.raw_force = None
        self.pose_error = np.linalg.norm(np.array(self.obs['robot']['positions'][7]) - np.array(self.goal))

        # Placeholder for old values:
        self.old_sorted_active_dict = None
        self.prev_taxel_ids = None
        
    def update_priority(self, updated_priority, mode):
        self.priority = updated_priority
        self.multi_controller.set_priority(updated_priority, mode)
    
    def get_priority_t(self, mode):
        """
        This function gets the priority based on mode
        #TODO: sorted_indices should be updated here
        """
        if mode == "A":
            prty = self.priority
        elif mode == "B":
            prty = heuristic_based_priority(self.old_sorted_active_dict, self.skeleton_contact_params)
        else: # TODO: Bandit is embeded here
            prty = self.priority
        
        return prty
    
    def process_active_dict(self, is_first=False):

        # TODO: Add a sorting algorithm here (sort the dictionary by index)
        skin_ids = self.obs['skin']['ids']
        skin_forces = self.obs['skin']['forces']
        skin_positions = self.obs['skin']['positions']
        
        sorted_indices = np.argsort(skin_ids)
        self.obs['skin']['ids'] = np.array(skin_ids)[sorted_indices]
        self.obs['skin']['forces'] = np.array(skin_forces)[sorted_indices]
        self.obs['skin']['positions'] = np.array(skin_positions)[sorted_indices]
        self.obs['skin']['skeleton_ids'] = np.array(self.obs['skin']['skeleton_ids'])[sorted_indices]
        print("Processing active dict taxel id: ", self.obs['skin']['ids'])
        forces, active_taxel_ids, positions = filter_active_taxels(self.obs['skin']['forces'], self.obs['skin']['ids'], self.obs['skin']['positions'], self.FORCE_THRESHOLD)
        print(f"Active taxel ids are {active_taxel_ids}")
        print(f"Active taxel forces are {forces}")
        print(f"Active taxel positions are {positions}")
        sorted_active_dict, self.taxel_dict, selected_taxel_ids = get_active_taxel_dict(ids = active_taxel_ids, 
                                                forces = forces, 
                                                positions = positions, 
                                                contact_params = self.multi_controller.contact_params,
                                                taxel_dict = self.taxel_dict,
                                                obs_skin= self.obs["skin"],
                                                is_us = True)
        print(f"Sorted active taxel ids are {selected_taxel_ids}")
        print(f"obs skin is {self.obs['skin']}")
        print(f"Sorted active dict is {sorted_active_dict}")
        # Log new max force based on raw taxel dict
        fc = []
        fc_lst = {}
        for key, i in zip(sorted_active_dict.keys(), range(self.arm_num)):
            if  sorted_active_dict[key].size == 0:
                # print(f"Taxel {key} is not active")
                fc += [0]
                fc_lst[i] = 0
            else:
                # print(f"Taxel {key} is active")
                fc += [sorted_active_dict[key][1, 0]]
                fc_lst[i] = sorted_active_dict[key][1, 0]
        
        for key in self.shared_dict.keys():
            self.shared_dict[key] = fc_lst[key]
            
        self.raw_force = fc
        
        sorted_active_dict = filter_active_taxels_soft(sorted_active_dict, self.soft_threshold)
        # print(f"Sorted_dict is {sorted_active_dict}")
        
        if is_first:
            # Update placeholder
            self.old_sorted_active_dict = sorted_active_dict
            self.prev_taxel_ids = np.copy(selected_taxel_ids)
        
        return active_taxel_ids, selected_taxel_ids, sorted_active_dict, forces, positions
    
    def get_force_dict(self):
        return self.shared_dict
        
    def run_controller(self, i, mode, feedback_t, is_first=False):
        ###############
        # At Step {t}
        ###############
        print ("Current tracker id: ", self.tracker.current_idx, " out of ", len(self.tracker.trajectory) - 1)

        self.obs['robot'] = self.env.robot.getRobotState()
        self.obs['skin'] = self.env.skin.getInfo()

        if is_first:
            self.process_active_dict(is_first = True)
            return
        
        prty = self.get_priority_t(mode)
        # print(f"Priority is {prty}")
        self.multi_controller.set_priority(prty, mode)

        goal_pos, goal_ori = self.tracker.get_next_waypoint(np.array(self.obs['robot']['positions'][7]))
        goal_ori = self.final_goal_ori

        self.pose_err =  np.linalg.norm(np.array(self.obs['robot']['positions'][7]) - np.array(self.tracker.trajectory[-1][0]))
        print(f"Pose error is {self.pose_err}")
        
        self.task_controller.set_goal_pose(goal_pos, goal_ori)

        action = self.multi_controller.update(self.obs, self.mode, add_gravity=True)
        self.env.step(action)

        ###############
        # At Step {t+1}
        ###############

        self.obs['robot'] = self.env.robot.getRobotState()
        self.obs['skin'] = self.env.skin.getInfo()
        
        active_taxel_ids, selected_taxel_ids, sorted_active_dict, forces, positions = self.process_active_dict(is_first = False)

        # Update active contact controllers
        self.multi_controller.update_controller_list(selected_taxel_ids)
        
        if (len(active_taxel_ids) > len(self.prev_taxel_ids)):
            # get new additions from prev taxel ids list
            new_taxel_ids = np.setdiff1d(active_taxel_ids, self.prev_taxel_ids)
            self.multi_controller.update_retract_dict(i, new_taxel_ids)
        
        # Process the dictionary so that only forces > soft threshold are kept
        self.multi_controller.update_buffer(i, self.obs)

        self.multi_controller.update_controller_list(selected_taxel_ids)
        
        self.old_sorted_actve_dict = sorted_active_dict
        self.prev_taxel_ids = np.copy(selected_taxel_ids)

        ###############################
        # MP Controllers Update Logic #
        ###############################

        # First, loop through each controller in the controller list for evaluation and update
        covered_vector = [0] * self.arm_num
        need = feedback_t
        flagged_dict = copy.deepcopy(sorted_active_dict)
        # print(f"flagged_dict is {flagged_dict}")

        # Update the controllers, and denote coverage info for further actions
        for key in self.multi_controller.contact_controllers:
            skeleton = self.taxel_dict[key].skeleton_id
            idx = np.where(selected_taxel_ids == key)
            self.taxel_dict[key].update(forces[idx][0], positions[idx])
            skeleton_contact_params = self.multi_controller.contact_params[f'b{skeleton}']
            force_threshold, Kp, Kd = skeleton_contact_params['force'], skeleton_contact_params['Kp'], skeleton_contact_params['Kd']
            self.multi_controller.contact_controllers[key].set_desired_force(force_threshold, Kp, Kd)
            
            covered_vector[skeleton] += 1
            if need != None:
                need[skeleton] = 3
            flagged_dict[f'b{skeleton}'] = clear(np.array(flagged_dict[f'b{skeleton}']), key)

        # if there are active taxels
        if selected_taxel_ids.size > 0:
            # Second, add the controller that has a higher priority but is not in the list yet:
            for skeleton in self.sorted_indices:
                if len(self.multi_controller.contact_controllers) <= self.multi_controller.max_contacts:
                    curr_link_active = flagged_dict[f'b{skeleton}']
                    if covered_vector[skeleton] == 0 and curr_link_active.size > 0: # If the controller list does not have a corresponding controller with higher priority, and there is active taxel
                        curr_id = int(curr_link_active[0, 0])
                        flagged_dict[f'b{skeleton}'] = clear(flagged_dict[f'b{skeleton}'], curr_id) # Remove it from the array
                        covered_vector[skeleton] += 1 # Update the dictated number
                        need[skeleton] = 3

                        idx = np.where(selected_taxel_ids == curr_id)
                        self.taxel_dict[curr_id].update(forces[idx][0], positions[idx])
                        # print(f"the multiprioirty priority is {self.multi_controller.priority}")
                        self.multi_controller.add_controller(curr_id, skeleton, self.taxel_dict)
            
            # Third, add the controller that experiences immediate feedback but is not in the list yet:
            for skeleton, val in enumerate(need):
                if len(self.multi_controller.contact_controllers) <= self.multi_controller.max_contacts:
                    curr_link_active = flagged_dict[f'b{skeleton}']
                    if val != 3:
                        curr_id = int(curr_link_active[0, 0])
                        flagged_dict[f'b{skeleton}'] = clear(flagged_dict[f'b{skeleton}'], curr_id) # Remove it from the array
                        covered_vector[skeleton] += 1 # Update the dictated number
                        need[skeleton] = 3

                        idx = np.where(selected_taxel_ids == curr_id)
                        self.taxel_dict[curr_id].update(forces[idx][0], positions[idx])
                        self.multi_controller.add_controller(curr_id, skeleton, self.taxel_dict)
