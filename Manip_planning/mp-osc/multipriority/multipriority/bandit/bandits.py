#!/usr/bin/env python3

from multipriority.agents import UnityRobot, RealRobot
from multipriority.utils import *
import numpy as np
import atexit

class ContextualBandit:

    def __init__(self, bandit_params, contact_params):
        """
        Parameters:
            bandit_params (dict): a dictionary read from the .yaml file
            contact_params (dict): a dictionary read fron the .yaml file
        """
        # Initialize params
        # TODO: Make variable names more descriptive
        self.w_user_feedback = bandit_params["w_user_feedback"]
        self.w_user_feedback_dec = bandit_params["w_user_feedback_dec"]
        self.lr = bandit_params["lr"]
        self.w_force_dev = bandit_params["w_force_dev"]
        self.w_pos = bandit_params["w_pos"]
        self.w_ori = bandit_params["w_ori"]
        self.gamma = bandit_params["gamma"]
        self.pos_error_threshold = bandit_params["pos_error_threshold"]
        self.ori_error_threshold = bandit_params["ori_error_threshold"]
        self.variance_threshold = bandit_params["variance_threshold"]
        self.num_arm = len(contact_params)
        self.w_soft_dev = bandit_params["w_soft_dev"]
        self.w_hard_dev_low = bandit_params["w_hard_dev_low"]
        self.w_hard_dev_high = bandit_params["w_hard_dev_high"]
        self.SITL_steps = bandit_params["SITL_steps"]

        # Initialize placeholders
        self.feedback_default = np.full(self.num_arm, 3)
        self.r_t = np.zeros(self.num_arm)
        self.priority = []
        for arm in contact_params.keys():
            self.priority += [contact_params[arm]["priority"]]
        # If task is enabled, add the task priority to the priority list
        if bandit_params["task_priority"] != -1:
            self.priority += [bandit_params["task_priority"]]
        total = sum(self.priority)
        self.weights = [x / total for x in self.priority]
        self.P_t = None
        self.P_t_history = []
        self.sigma = np.zeros(self.num_arm)
        self.epsilon = bandit_params["epsilon"]

        # Logging Data
        self.feedback_history = []
        self.reward_history = []
        self.value_history = []
        self.weights_history = []
        self.probability_history = []

        # Check for convergence
        self.is_converged = False
        self.variance_window_size = bandit_params["window"]
        
        convergence_checkers_dict = {
            "var": self.variance_check,
            "task": self.task_check,
            "reward": self.reward_check,
        }
        # TODO: load constant values needed to perform convergence checks here

        self.convergence_checkers = [convergence_checkers_dict[checker] for checker in bandit_params["convergence_checkers"]]

    def softmax(self, v):
        """
        Compute the softmax of a vector.

        Parameters:
        v (np.array): A numpy array representing the input vector.

        Returns:
        np.array: A numpy array where the softmax function has been applied to the input.
        """
        # Improved softmax for numerical stability
        v = v - np.max(v)
        exp_v = np.exp(v)
        return exp_v / np.sum(exp_v)
    
    def temporal_learning(self, expectation, gradient, sigma):

        V_t = self.value_function(sigma, self.weights)
        delta = expectation + self.gamma * V_t - self.value_function(self.sigma, self.weights) 
        print("delta: ", delta)
        positive_feedback_mask = np.sign(sigma) > 0

        # Only apply self.w_user_feedback_dec where the sign of difference between default and current feedback is positive
        additional_update = self.w_user_feedback_dec * positive_feedback_mask.astype(int)

        # Update the weights with the modified rule
        print("weight term 2:", self.lr * delta * gradient)
        self.value_history += [V_t.tolist()]
        print("weight term 3:", additional_update)
        self.weights = self.weights + self.lr * delta * gradient + additional_update
        self.weights_history += [self.weights.tolist()]
        print("Current weights: ", self.weights)

    def value_function(self, s, w):
        """
        Linear value function. (Naive version for now)

        Parameters:
        s (np.array): The state vector.
        w (np.array): The weight vector.

        Returns:
        float: The value of the state.
        """
        return np.dot(w, (np.abs(s)))
    
    def reward_function(self, sigma, state_error, tactile_data, hard_threshold, soft_threshold):
        """
        Reward function: R_t = -w_user_feedback * |sigma| - w_{pos, ori} * ||epsilon|| + w_user_feedback_dec * sign(sigma)

        Parameters:
        sigma (np.array): The deviation vector between the standard feedback and the current feedback vector
        state_error (np.array): The state error of the end-effector
        tactile_data (np.array): #TODO: to be determined
        threshold (float): The condition threshold of state error

        Returns:
        r_t: The reward vector for each arm of time stamp t
        """
        r_t = -self.w_user_feedback * np.abs(sigma)
        
        # Apply the penalty if the state error is too large:
        # TODO: Add the orientation error
        if state_error[0] >= self.pos_error_threshold:
            r_t -= self.w_pos * state_error[0]
        # if state_error[1] >= self.ori_error_threshold:
        #     r_t -= self.w_ori * state_error[1]
        
        # Calculate the force reward term:
        dev = 0
        for i, reading in enumerate(tactile_data):
                force_lst = tactile_data[reading][1, :]
                avg_deviation = 0
                if len(force_lst) != 0:
                    for force in force_lst:
                        if force <= soft_threshold[i]:
                            avg_deviation += self.w_soft_dev * (soft_threshold[i] - force) / soft_threshold[i]
                        elif force < hard_threshold[i]:
                            avg_deviation -= self.w_hard_dev_low * (hard_threshold[i] - force) / hard_threshold[i]
                        elif force >= hard_threshold[i]:
                            avg_deviation -= self.w_hard_dev_high * (-hard_threshold[i] + force) / hard_threshold[i]
                    avg_deviation = avg_deviation/len(force_lst)
                dev += avg_deviation ** 2

        r_c1_F = -self.w_force_dev * dev

        r_t += r_c1_F
        
        self.reward_history += [r_t.tolist()]

        return r_t
    
    def get_priority(self):
        """
        This function returns the priority of each body part

        Returns:
            priority: The rank of each body part
        """
        unique_vals = np.unique(self.P_t)[::-1]

        rank_dict = {value: rank for rank, value in enumerate(unique_vals)}
        ranks = [rank_dict[value] for value in self.P_t]
        self.priority = ranks
        return self.priority
    
    def variance_check(self):
        """
        This function compares the variance of the probability for a consecutive steps to check for convergence
        """
        convergence = False
        if len(self.P_t_history) >= self.variance_window_size:
            recent_P_ts = np.array(self.P_t_history[-self.variance_window_size:])
            variance = np.var(recent_P_ts, axis=0)
            if np.all(variance < self.variance_threshold):
                convergence = True
                self.probability_history += [self.P_t_history]
                self.P_t_history = []
                print("Convergence achieved based on variance threshold.")
        return convergence

    def task_check(self):
        """
        This function compares the task error to check for convergence
        """
        return True

    def reward_check(self):
        """
        This function compares the reward to check for convergence
        """
        return True    

    def main(self, sorted_active_dict, feedback_t, state_error, contact_params):
        """
        Contextual Multi-Armed Bandit Algorithm for Action Ranking with TD Learning under Epsilon-Greedy Algorithm

        Parameters:
        sorted_active_dict (np.array): The sorted tactile sensor data at time stamp t
        feedback_t (np.array): The current feedback vector received fron ROS service
        state_error (float): Current end-effector state error
        contact_params (dict): current contact params

        Returns:
        updated_contact_params (dict): the updated contact params
        """

        if np.random.rand() < self.epsilon:
            random_probs = np.random.rand(self.num_arm)
            self.P_t = random_probs / np.sum(random_probs)
        else:
            sigma = self.feedback_default - feedback_t

            # Calculate the reward term
            force_threshold = [contact_params[key]["force"] for key in contact_params]
            soft_threshold = [contact_params[key]["soft_factor"] for key in contact_params]
            r_t = self.reward_function(sigma = sigma, 
                                    state_error = state_error, 
                                    tactile_data = sorted_active_dict,
                                    hard_threshold = force_threshold,
                                    soft_threshold = soft_threshold)
            
            # Calculate the expectation of the reward function
            E_r = np.sum(self.softmax(self.weights) * r_t)

            # Compute TD error
            self.temporal_learning(expectation=E_r, 
                                    gradient=abs(sigma), 
                                    sigma = sigma)
            
            # Update the probability for each arm
            self.P_t = self.softmax(self.weights)

        self.P_t_history += [self.P_t.tolist()]
        print("Current possibility: ", self.P_t)

        self.priority = self.get_priority()
        self.is_converged = np.all([checker() for checker in self.convergence_checkers])
        print("Convergence Boolean: ", self.is_converged)

    def calc_priority(self, sorted_active_dict, feedback_t, state_error, contact_params):
        epsilon = 0.2
        self.main(sorted_active_dict = sorted_active_dict, 
                    feedback_t = feedback_t, 
                    state_error =  state_error, 
                    contact_params = contact_params)

        sorted_indices = np.argsort(self.priority)
        return sorted_indices