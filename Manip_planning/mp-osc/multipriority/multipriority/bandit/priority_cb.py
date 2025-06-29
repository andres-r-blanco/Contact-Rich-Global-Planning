import numpy as np
import matplotlib.pyplot as plt
from multipriority.utils import load_yaml
import math
from itertools import permutations
import random

random.seed(42)
np.random.seed(42)

from .utils_contextual import LinearRewardHistory, linucb_rank_policy, linucb_rank_policy_data, LinearContextualBandit, greedy_policy

rng = np.random.default_rng(42)

class PriorityLinUCB(LinearContextualBandit):
    def __init__(self, priority_params, contact_params, rng=None, sigma=1):
        self.state_size_dict = {
            'f_max': len(contact_params),
            'position': 3,
            'position_err': 3,
            'position_err_norm': 1,
            'f_d': len(contact_params),
            'f_th': len(contact_params),
            'u_fb': len(contact_params),
        }
        obs_len = 0
        obs_dict = {}
        for s in priority_params["state"]:
            # add start index of each state
            obs_dict[s] = obs_len
            obs_len += self.state_size_dict[s]
        self.obs_dict = obs_dict

        reward_dict = {}
        for r in priority_params["reward"]:
            reward_dict[r] = 1
        self.reward_dict = reward_dict

        self.state_space_size = obs_len
        self.state_tmpl = np.zeros(self.state_space_size)
        self.action_space_size = len(contact_params)
        self.priority = []
        for skel_id in contact_params.keys():
            self.priority += [contact_params[skel_id]["priority"]]

        if priority_params["task_priority"] != -1:
            self.priority += [priority_params["task_priority"]]
            self.action_space_size += 1

        super().__init__(self.action_space_size, rng, self.state_space_size, sigma)

        self.w_user_feedback = priority_params["w_user_feedback"]
        self.w_user_feedback_dec = priority_params["w_user_feedback_dec"]
        self.w_force_dev = priority_params["w_force_dev"]
        self.w_pos = priority_params["w_pos"]
        self.w_ori = priority_params["w_ori"]
        self.pos_error_threshold = priority_params["pos_error_threshold"]
        self.ori_error_threshold = priority_params["ori_error_threshold"]
        self.variance_threshold = priority_params["variance_threshold"]
        self.w_hard_dev_high = priority_params["w_hard_dev_high"]
        self.SITL_steps = priority_params["SITL_steps"]
        # self.K = len(contact_params)

        # Initialize placeholders
        self.feedback_default = np.full(len(contact_params), 3)
        self.r_t = np.zeros(self.action_space_size)
        
        # Logging Data
        self.feedback_history = []
        self.reward_history = []
        self.value_history = []
        self.weights_history = []
        self.probability_history = []
        self.bonus_history = []

        # Check for convergence
        self.is_converged = False
        self.variance_window_size = priority_params["window"]
        self.means = []
        self.bonuses = []
        
        # convergence_checkers_dict = {
        #     "var": self.variance_check,
        #     "task": self.task_check,
        #     "reward": self.reward_check,
        # }
        # # TODO: load constant values needed to perform convergence checks here

        # self.convergence_checkers = [convergence_checkers_dict[checker] for checker in bandit_params["convergence_checkers"]]

        self.hist = LinearRewardHistory(np.arange(self.action_space_size), self.state_space_size)
        self.policy = linucb_rank_policy_data
        self.rng = np.random.default_rng(42)
        self.best_arm = None
        self.reward_running_mean = 0
        self.reward_running_std = 0
        self.num_selected = np.zeros(self.action_space_size)
        self.cum_regret = 0

    def compute_regret(self):
        regret = 0
        for i in range(self.action_space_size):
            regret += self.num_selected[i] * (self.r_t[i] - self.r_t[self.best_arm])
        self.cum_regret += regret
        
    def pull_gaussian_reward(self, arm, context):
        return np.dot(self.mus[arm], context) + self.noise()

    def pull(self, sigma, state_error, tactile_data, hard_threshold, goal_force):
        """
        Reward function: R_t = -w_user_feedback * |sigma| - w_{pos, ori} * ||epsilon|| + w_user_feedback_dec * sign(sigma)

        Parameters:
        sigma (np.array): The deviation vector between the standard feedback and the current feedback vector
        state_error (np.array): The state error of the end-effector
        threshold (float): The condition threshold of state error

        Returns:
        reward: The reward vector for each arm of time stamp t
        """
        reward = 0
        non_zero = 1
        if 'f_d' in self.reward_dict:
            r_fd = 0
            for i, skel_id in enumerate(tactile_data):
                force_lst = tactile_data[skel_id][1, :]
                if len(force_lst) > 0:
                    f_reg = 0
                    f_reg = -self.w_force_dev * ((goal_force[i] - force_lst[0]))**2
                    non_zero += 1
                    # if np.abs(force_lst[0] - goal_force[i]) > 1:
                    #     f_reg = -self.w_force_dev #* (goal_force[i] - force_lst[0])**2
                    avg_deviation = f_reg # + f_thresh_err + f_des_err 
                else:
                    avg_deviation = 0. # can try playing with this value
                r_fd += avg_deviation
            reward = reward + r_fd / non_zero
        
        if 'f_th' in self.reward_dict:
            r_ft = 0
            non_zero = 1
            for i, skel_id in enumerate(tactile_data):
                force_lst = tactile_data[skel_id][1, :]
                if len(force_lst) > 0:
                    if force_lst[0] > hard_threshold[i]:
                        r_ft -= self.w_hard_dev_high * ((hard_threshold[i] - force_lst[0])) **2 # since this is objectively bad, no need to penalize if it's below since f_d should take care of that bit
                        non_zero += 1
            reward = reward + r_ft / non_zero

        if 'u_fb' in self.reward_dict:
            reward -= self.w_user_feedback * np.abs(sigma) # scales reward based on severity of deviation from default feedback

        if 'position' in self.reward_dict:
            reward = reward - min(1, self.w_pos * np.linalg.norm(state_error)) # penalizes a given s,a pair based on the state error

        self.reward_history += [[reward]]
        # filter out 2,4,6,9 from self.priority
        priority = [i for i in self.priority if i in [2,4,6,9]]
        # self.reward_running_mean = np.mean(self.reward_history)
        # self.reward_running_std = np.std(self.reward_history)
        # reward = (reward - self.reward_running_mean) / (self.reward_running_std + 0.01)
        
        print ("Priority: ", priority, " Reward: ", reward)
        
        return reward

    def get_random_context(self):
        return np.sqrt(10)*self.rng.standard_normal(self.d)
    
    def get_context(self, sorted_active_dict, feedback_t, pos, pos_error, contact_params):
        """
        This function constructs the features vector for each step

        Parameters:
        sorted_active_dict (np.array): The sorted tactile sensor data at time stamp t
        state_error (float): Current end-effector state error
        contact_params (dict): current contact params
        """
        s_t = np.zeros(self.state_space_size)
        if 'f_max' in self.obs_dict:
            s_t[self.obs_dict['f_max']:self.obs_dict['f_max']+self.state_size_dict['f_max']] = [np.max(sorted_active_dict[key][1], initial=0) for key in sorted_active_dict.keys()]
            s_t[self.obs_dict['f_max']:self.obs_dict['f_max']+self.state_size_dict['f_max']] += 0.01 # to avoid numerical issues
        if 'position' in self.obs_dict:
            s_t[self.obs_dict['position']:self.obs_dict['position']+self.state_size_dict['position']] = pos
        if 'position_err' in self.obs_dict:
            s_t[self.obs_dict['position_err']:self.obs_dict['position_err']+self.state_size_dict['position_err']] = pos_error
        if 'position_err_norm' in self.obs_dict:
            s_t[self.obs_dict['position_err_norm']] = np.linalg.norm(pos_error)
        if 'f_d' in self.obs_dict:
            s_t[self.obs_dict['f_d']:self.obs_dict['f_d']+self.state_size_dict['f_d']] = [np.max(sorted_active_dict[key][1], initial=0) - contact_params[key]["goalF"] for key in sorted_active_dict.keys()]
        if 'f_th' in self.obs_dict:
            s_t[self.obs_dict['f_th']:self.obs_dict['f_th']+self.state_size_dict['f_th']] = [np.max(sorted_active_dict[key][1], initial=0) - contact_params[key]["force"] for key in sorted_active_dict.keys()]
        if 'u_fb' in self.obs_dict:
            s_t[self.obs_dict['u_fb']:self.obs_dict['u_fb']+self.state_size_dict['u_fb']] = feedback_t - self.feedback_default + 0.01 # to avoid numerical issues

        soft_threshold = []
        hard_threshold = []
        return s_t, soft_threshold, hard_threshold

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
        self.sigma = self.feedback_default - feedback_t

        # Calculate the reward term
        force_threshold = [contact_params[key]["force"] for key in contact_params]
        soft_threshold = [contact_params[key]["soft_factor"] for key in contact_params]
        
        self.context, st, ht = self.get_context(sorted_active_dict, feedback_t, state_error, contact_params)
        
        best, priority, means, bonuses = self.policy(self.hist, self.context, self.rng)
        print ("Best Arm: ", best, " Priority: ", priority)
        self.priority = {f'b{value}': i for i, value in enumerate(priority)}

        # done = False

        # self.is_converged = np.all([checker() for checker in self.convergence_checkers])
        # print("Convergence Boolean: ", self.is_converged)

    def var_checker(self):
        """
        Checks if the mean of the UCBs has changed significantly
        """
        print ("=====================================")
        print ("Variance: ", np.var(self.value_history), " Bonus Variance: ", np.var(self.bonus_history))
        print ("=====================================")
        unexplored = self.hist.get_unexplored_actions(threshold=1)
        if len(unexplored) > 0 or len(self.value_history) == 1 or len(self.bonus_history) == 0:
            return False
        if len(self.value_history) > self.variance_window_size:
            self.value_history = self.value_history[1:]
        if len(self.bonus_history) > self.variance_window_size:
            self.bonus_history = self.bonus_history[1:]
        var1 = np.var(self.value_history) < self.variance_threshold
        var2 = np.var(self.bonus_history) < self.variance_threshold
        if var1 and var2:
            return True
        return False
        
    def get_action(self, context, alpha=1):
        self.context = context
        self.best_arm, self.priority, self.means, self.bonuses = self.policy(self.hist, context, self.rng, alpha=alpha)
        self.num_selected[self.best_arm] += 1
        # self.priority = (np.array(self.priority) + 1) * 2
        priority = [i for i in self.priority if i in [2,4,6,9]]
        print ("Priority: ", priority)
        self.value_history += [np.mean(self.means)]
        self.bonus_history += [np.mean(self.bonuses)]
        # print ("priority: ", self.priority)
        
        # priority = {f'b{value}': i for i, value in enumerate(self.priority.tolist())}
        # print ("priority dict: ", priority)
        return self.priority
        # return priority

    def get_action_2(self, context, alpha=1):
        self.context = context
        self.best_arm, self.priority = self.policy(self.hist, context, self.rng, alpha=alpha)
        return self.best_arm

    def update(self, sorted_active_dict, feedback_t, pos, pos_error, contact_params, done=False):
        sigma = feedback_t - self.feedback_default
        # Calculate the reward term
        force_threshold = [contact_params[key]["force"] for key in contact_params]
        goal_force = [contact_params[key]["goalF"] for key in contact_params]
        reward = self.pull(sigma = self.sigma, 
                        state_error = pos_error, 
                        tactile_data = sorted_active_dict,
                        hard_threshold = force_threshold,
                        goal_force = goal_force)
        self.hist.record(self.best_arm, self.context, reward)

    def test_main(self, alpha):
        context = self.get_random_context()
        self.best_arm, self.priority = self.policy(self.hist, context, self.rng)
        self.reward = self.pull_gaussian_reward(self.best_arm, context)
        # print ("reward: ", reward)
        self.hist.record(self.best_arm, context, self.reward)

if __name__ == "__main__":
    skeleton_config_name = 'test_cube_params.yaml'
    skeleton_contact_params = load_yaml(skeleton_config_name)
    bandit_config_name = 'bandit_params.yaml'
    bandit_params = load_yaml(bandit_config_name)
    
    cumulative_reward = 0
    cr = []
    
    
    def simulate_user_click(article_features, user_preference):
        """ Simulate whether a user clicks on an article based on their preference. """
        true_reward = article_features.dot(user_preference)
        click_probability = 1 / (1 + np.exp(-true_reward))
        return np.random.rand() < click_probability

    def simulate_user_click2(article_features, user_preference):
        """ Simulate whether a user clicks on an article based on their preference. """
        # true_reward = article_features.dot(user_preference)
        # click_probability = 1 / (1 + np.exp(-true_reward))
        click_probability = [0.2, 0.4, 0.8, 1.0, 0.3]
        return click_probability

    def simulate_user_click3(chosen_article, user_preference):
        """ Simulate whether a user clicks on an article based on their preference. """
        true_reward = article_features.dot(user_preference)
        click_prob = np.array([0.2, 0.4, 0.8, 1.0, 0.3])
        return np.random.rand() < click_prob[int(chosen_article)]
        
    # Simulation parameters
    n_articles = 5
    n_features = 3
    n_iterations = 1000

    # Randomly generated article features and user preferences
    article_features = np.random.rand(n_articles, n_features)
    user_preference = np.random.rand(n_features)

    # Initialize the bandit
    bandit = PriorityLinUCB(bandit_params, skeleton_contact_params, K=n_articles, d=n_features)

    # Run the simulation
    for _ in range(n_iterations):
        # Choose an article for the user
        chosen_article = bandit.get_action_2(user_preference, alpha=1.0)

        # Simulate whether the user clicks on the article
        reward = simulate_user_click3(chosen_article, user_preference)

        # Update the bandit with the observed reward
        bandit.hist.record(chosen_article, user_preference, reward)
    
    # prob = []
    # for i in range(n_articles):
        # prob.append(simulate_user_click2(article_features[i], user_preference))
    
    
    chosen_article = bandit.get_action_2(user_preference, alpha=0.01)
    # print ("prob: ", prob)
    print ("Use Preference: ", user_preference)
    print ("Reward prob based ranking: ", np.argsort(simulate_user_click2(chosen_article, user_preference))[::-1])
    print ("Best Arm: ", bandit.best_arm, " Priority: ", bandit.priority)
    # import matplotlib.pyplot as plt
    # for i in range(1000):
    #     priority_linucb.test_main(alpha=1)
    #     print ("Iteration: ", i, " Best Arm: ", priority_linucb.best_arm, " Priority: ", priority_linucb.priority)
    #     print ("-----------------------------------")
    #     cumulative_reward += priority_linucb.reward
    #     cr += [cumulative_reward]
    
    # plt.plot(cr, label="alpha1")
    
    # priority_linucb = PriorityLinUCB(bandit_params, skeleton_contact_params, K=len(skeleton_contact_params))
    
    # cumulative_reward = 0
    # cr = []
    
    # import matplotlib.pyplot as plt
    # for i in range(1000):
    #     priority_linucb.test_main(alpha=0.5)
    #     print ("Iteration: ", i, " Best Arm: ", priority_linucb.best_arm, " Priority: ", priority_linucb.priority)
    #     print ("-----------------------------------")
    #     cumulative_reward += priority_linucb.reward
    #     cr += [cumulative_reward]
    
    # plt.plot(cr, label="alpha2")
    # plt.legend()
    # plt.show()