## Author: Rohan Banerjee + Sarah Dean

import numpy as np
import scipy.stats
from ipywidgets import Button, HBox
import itertools
from plot_ellipse import plot_ellipse


class LinearContextualBandit():
    """
    Linear contextual bandit with Gaussian rewards.

    Default number of arms is K=5, default dimension of context is d=2.

    Context distribution is N(0, sigma^2*I_d).
    """
    def __init__(self, K=5, rng=None, d=2, sigma=1):
        self.rng = np.random if rng is None else rng
        self.K = K
        self.d = d

        # Generate d dim vectors
        self.mus = [np.sqrt(10)*self.rng.standard_normal(d) for _ in range(K)]
        # Observation noise is clipped Gaussian
        self.sigma = sigma
        self.noise = lambda: np.clip(self.sigma*self.rng.standard_normal(),-10,10)
    
    def pull(self, arm, context):
        return np.dot(self.mus[arm], context) + self.noise()
    
    def get_context(self):
        return np.sqrt(10)*self.rng.standard_normal(self.d)

class PriorityLinUCB(LinearContextualBandit):
    """
    Linear contextual bandit with Gaussian rewards.

    Default number of arms is K=5, default dimension of context is d=2.

    Context distribution is N(0, sigma^2*I_d).
    """
    def __init__(self, K=5, rng=None, d=2, sigma=1):
        super().__init__(K, rng, d, sigma)

        # Generate d dim vectors
        self.mus = [np.sqrt(10)*self.rng.standard_normal(d) for _ in range(K)]
        # Observation noise is clipped Gaussian
        self.sigma = sigma
        self.noise = lambda: np.clip(self.sigma*self.rng.standard_normal(),-10,10)
    
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
        if 'f_d' in self.reward_dict:
            r_fd = 0
            for i, skel_id in enumerate(tactile_data):
                force_lst = tactile_data[skel_id][1, :]
                if len(force_lst) > 0:
                    f_reg = -self.w_force_dev * (goal_force[i] - force_lst[0])**2
                    avg_deviation = f_reg # + f_thresh_err + f_des_err 
                else:
                    avg_deviation = 0. # can try playing with this value
                r_fd += avg_deviation
            reward += r_fd
        
        if 'f_th' in self.reward_dict:
            r_ft = 0
            for i, skel_id in enumerate(tactile_data):
                force_lst = tactile_data[skel_id][1, :]
                if len(force_lst) > 0:
                    if force_lst[0] > hard_threshold[i]:
                        r_ft -= self.w_hard_dev_high * (hard_threshold[i] - force_lst[0])**2 # since this is objectively bad, no need to penalize if it's below since f_d should take care of that bit
            reward += r_ft

        if 'u_fb' in self.reward_dict:
            reward -= self.w_user_feedback * np.abs(sigma) # scales reward based on severity of deviation from default feedback

        if 'position' in self.reward_dict:
            reward -= self.w_pos * np.linalg.norm(state_error) # penalizes a given s,a pair based on the state error

        self.reward_history += [[reward]]
        # filter out 2,4,6,9 from self.priority
        priority = [i for i in self.priority if i in [2,4,6,9]]
        print ("Priority: ", priority, " Reward: ", reward)
        return reward
    
    def get_context(self):
        return np.sqrt(10)*self.rng.standard_normal(self.d)
    
class LinearRewardHistory():
    """
    History of rewards for a linear contextual bandit.

    Similar to utils.RewardHistory, but with a context vector.

    self.ci: dictionary mapping action to (mean, s, v) where
        mean: estimated mean parameter vector for the action
        s: singular values of data matrix for the action (Sigma matrix)
        v: right singular vectors of the data matrix for the action (V^T matrix)
    """

    def __init__(self, actions, d):
        self.actions = actions
        self.action_dim=1
        self.reward_dim=1
        self.d = d
        self.history = np.empty([0,self.action_dim+d+self.reward_dim])  # TODO: refactor into a pandas df
        self.ucb_history = np.empty([0,len(self.actions)])            # optionally used to record UCB values
        self.ucb_mean_history = np.empty([0,len(self.actions)])       # optionally used to record UCB mean values
        self.ucb_bonus_history = np.empty([0,len(self.actions)])      # optionally used to record UCB bonus values
        self.ucb_alpha_bonus_history = np.empty([0,len(self.actions)])      # optionally used to record UCB alpha*bonus values
        self.per_action_history = {action: [] for action in actions}
        self.ci = {action: (None, None, None) for action in actions} # mean, Sigma
        self.T = 0
        # For each action, we store a state variable that indicates whether the CI requires an update.
        #      this will allow us to only recompute the CI for a particular action when we need to.
        #      (this will be useful for the LinUCB policy, where we only need to recompute the CI for
        #      the action with the highest UCB value)
        #
        # State dynamics:
        #   1. At initialization, this flag should be True (to force initial CI computation).
        #   2. When we record/add data for an arm, the flag should be set to True.
        #   3. When we re-compute the CI for an arm, the flag should be set to False.
        self.update_CI_required = {action: True for action in actions}

    def record(self, action, context, reward, pretrain=False):
        """
        Record a reward for a given action and context.

        Args:
            action (int): the action taken
            context (np.array): the context vector
            reward (float): the reward
            pretrain (bool): whether this a pre-training sample (default False)
        """
        self.history = np.append(self.history, [np.hstack([action, context, reward])], axis=0)
        self.per_action_history[action].append(np.hstack([context, reward]))
        if not pretrain:
            self.T += 1
        self.update_CI_required[action] = True   # make a note to update CI for this action.

    def record_batch(self, actions, contexts, rewards, pretrain=False):
        """
        Record a batch of rewards for given actions and contexts.

        Args:
            actions (np.array, shape=(T,)): the actions taken
            contexts (np.array, shape=(T,d)): the context vectors
            rewards (np.array, shape=(T,)): the rewards
            pretrain (bool): whether this a pre-training batch (default False)
        """
        self.history = np.append(self.history, np.hstack([actions.reshape(-1,1), contexts, rewards.reshape(-1,1)]), axis=0)
        for action in self.actions:
            if np.any(actions==action):
                per_action_data = np.hstack([contexts[actions==action], rewards[actions==action].reshape(-1,1)])
                self.per_action_history[action].extend(list(per_action_data))
                self.update_CI_required[action] = True   # make a note to update CI for this action.
        if not pretrain:
            self.T += len(actions)

    def record_ucb(self, ucb_list, mean_list, bonus_list, alpha=1):
        """
        Record a list of UCB values for each action.
        """
        # Here we need to check to see if self.T has incremented by more than 1
        # since we last added data. If it has, then we need to add a row of NaNs first.
        ucb_size = len(self.ucb_history)
        if ucb_size < self.T:
            nan_array = np.empty([self.T-ucb_size, len(self.actions)])
            nan_array[:] = np.nan
            self.ucb_history = np.append(self.ucb_history, nan_array, axis=0)
            self.ucb_mean_history = np.append(self.ucb_mean_history, nan_array, axis=0)
            self.ucb_bonus_history = np.append(self.ucb_bonus_history, nan_array, axis=0)
            self.ucb_alpha_bonus_history = np.append(self.ucb_alpha_bonus_history, nan_array, axis=0)
        self.ucb_history = np.append(self.ucb_history, [ucb_list], axis=0)
        self.ucb_mean_history = np.append(self.ucb_mean_history, [mean_list], axis=0)
        self.ucb_bonus_history = np.append(self.ucb_bonus_history, [bonus_list], axis=0)
        self.ucb_alpha_bonus_history = np.append(self.ucb_alpha_bonus_history, [alpha*bonus_list], axis=0)

    def compute_ci(self, svd_condition="strict"):
        """
        Computes the least-squares estimate of the mean parameter vector for each action,
        along with the singular values and right singular vectors of the data matrix for that action.

        svd_condition: "strict" or "loose" (if strict, only compute SVD
        if we have at least d samples; if loose, compute SVD as long as we have
        a single sample).
        - If the svd-condition is not met, then the s and v matrices are set to None.
        """
        svd_check = lambda N: N >= 1*self.d if svd_condition == "strict" else N >= 1
        # Only recompute CIs for actions that explicitly require it.
        actions_to_update = [a for a in self.actions if self.update_CI_required[a]]
        for action in actions_to_update:
            N = len(self.per_action_history[action])
            if N >= 1:
                matrix = np.array(self.per_action_history[action])
                A = matrix[:,:-1]
                b = matrix[:,-1]
                mean, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            else:
                mean = None
            if svd_check(N):
                _,s,v = np.linalg.svd(A)
            else:
                s, v = None, None  
            self.ci[action] = (mean, s, v)
            self.update_CI_required[action] = False # reset flag

    def get_unexplored_actions(self, threshold=1):
        """
        Returns the actions that have been explored less than (N+1)*d times.
        (bug-fixed from the original code)

        Previously, thresholds were of the form (N+1)*self.d, for user-specified N.

        Arg:
            threshold (int): the number of samples required to be considered explored
        """
        ret = [a for a in self.actions if len(self.per_action_history[a])<threshold]
        return ret

    def get_smallest_N(self):
        min_N = min([len(self.per_action_history[a]) for a in self.actions])
        ret = [a for a in self.actions if len(self.per_action_history[a]) <= min_N][0]
        return ret

    def get_means(self, context=None, svd_condition="strict", recompute_ci=True):
        if recompute_ci:
            self.compute_ci(svd_condition=svd_condition)
        if context is None:
            ret = np.array([self.ci[a][0] for a in self.actions])
        else:
            ret = np.array([np.dot(self.ci[a][0], context) for a in self.actions])
        return ret

    def get_highest_mean(self, context):
        self.compute_ci()
        max_r = max([np.dot(self.ci[a][0], context) for a in self.actions])
        ret = [a for a in self.actions if np.dot(self.ci[a][0], context) >= max_r][0]
        return ret

    def get_exploration_bonuses(self, context, svd_condition="strict", recompute_ci=True, lam=0):
        """
        Returns the exploration bonus for each arm.
        Assumes that each action has at least one sample.

        Args
            lam (float): the regularization parameter, to ensure that rank(sigma_matrix) = N,
                         where N is the number of observed samples for the action.
        """
        # ROHAN: bug fix to make UCB calculation correct
        # Is there some intuitive way to verify that this is correct?
        # (e.g. how are the singular values affected as a function of the number of data samples N)
        if recompute_ci:
            self.compute_ci(svd_condition=svd_condition)
        array = []
        for a in self.actions:
            num_pts = len(self.ci[a][1])
            sigma_inv = np.diag(1.0 / (self.ci[a][1] + lam))
            if num_pts < self.d:
                # Special handling where number of data points is less than context dimension:
                # 0-pad to make the sigma-inverse matrix have shape (num_pts,d)
                sigma_inv = np.hstack([sigma_inv, np.zeros([num_pts, self.d-num_pts])])
            array.append(np.linalg.norm(sigma_inv @ self.ci[a][2] @ context))
        ret = np.array(array)
        return ret

    def get_ucb(self, context, alpha, svd_condition="strict", lam=0.1):
        """
        Returns the following statistics:
            - ucb_list: the UCB value for each action
            - means: the estimated mean expected reward for each action
            - bonuses: the exploration bonus for each action
        """
        means = self.get_means(context, svd_condition=svd_condition, recompute_ci=True)
        bonuses = self.get_exploration_bonuses(context, svd_condition=svd_condition, recompute_ci=False, lam=lam)
        ucb_list = means + alpha*bonuses
        return ucb_list, means, bonuses

def update_plot(ax, hist, K, context, colors):
    """
    Updates the plot with information about the observed reward history.

    For each arm (with a different color):
    - plots the observed contexts (with crosses)
    - plots the estimated mean context (with a solid line from the origin to a solid dot).
    - show the CI ellipse around the mean context.

    The current input context is shown with a black cross.

    Args:
        ax (matplotlib.axes.Axes): the axes to plot on
        hist (LinearRewardHistory): the reward history
        K (int): the number of arms
        context (np.ndarray): the current context
        colors (list(color): len=K): a list of colors to use for each arm
    """
    print("TEST")
    ax.clear()
    
    plotted_arms = 0
    xmin, xmax = (-5,5)
    ymin, ymax = (-5,5)
    for arm in range(K):
        # current context
        ax.scatter(context[0], context[1], color='black', marker='x')
        
        # observed contexts
        if len(hist.per_action_history[arm]) > 0:
            contexts = np.array(hist.per_action_history[arm])[:,:-1]
            ax.scatter(contexts[:,0], contexts[:,1], color=colors[arm], marker='x', alpha=0.5)
            xmin = min(xmin, np.min(contexts[:,0]))
            xmax = max(xmax, np.max(contexts[:,0]))
            ymin = min(ymin, np.min(contexts[:,1]))
            ymax = max(ymax, np.max(contexts[:,1]))
        
        # estimated parameters
        mean, s, u = hist.ci[arm]
        if mean is not None:
            # mean
            ax.scatter(mean[0], mean[1], color=colors[arm], label=arm)
            ax.plot([0, mean[0]], [0, mean[1]], color=colors[arm])
            xmin = min(xmin, mean[0])
            xmax = max(xmax, mean[0])
            ymin = min(ymin, mean[1])
            ymax = max(ymax, mean[1])
        
            # confidence ellipse
            if s is not None:
                plot_ellipse(ax, cov=u.T@np.diag(1/s)@u, x_cent=mean[0], y_cent=mean[1], plot_kwargs={'alpha':0}, fill=True,
                fill_kwargs={'color':colors[arm],'alpha':0.1})
            plotted_arms += 1

    ax.set_title("contexts and estimated parameters")
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    if plotted_arms>0: ax.legend(title='arm')
    ax.grid()
    
class InteractivePlot():
    def __init__(self, mab, hist, axs):
        self.mab = mab
        self.hist = hist
        arm_buttons = [Button(description=str(arm)) for arm in np.arange(mab.K)]
        reveal_button = Button(description='Reveal')
        policy_bottons = [Button(description='ArgMax'), Button(description='LinUCB'), Button(description='Optimal')]
        self.combined = HBox([items for items in arm_buttons] + [reveal_button] + policy_bottons)
        
        self.ax = axs[0]
        self.ax2 = axs[1]
        self.colors = ['r', 'm', 'b', 'c', 'g', 'y']
        
        for n in range(mab.K):
            arm_buttons[n].on_click(self.upon_clicked)
        reveal_button.on_click(self.upon_reveal)
        for b in policy_bottons:
            b.on_click(self.upon_policy)
        
        self.context = self.mab.get_context()
        update_plot(self.ax, self.hist, self.mab.K, self.context, self.colors) 
        
    def upon_clicked(self, btn):
        arm = int(btn.description)
        reward = self.mab.pull(arm, self.context)
        self.hist.record(arm, self.context, reward)
        self.hist.compute_ci()
        self.context = self.mab.get_context()
        update_plot(self.ax, self.hist, self.mab.K, self.context, self.colors) 
    
    def upon_reveal(self, b):
        xs = [mu[0] for mu in self.mab.mus]
        ys = [mu[1] for mu in self.mab.mus]
        self.ax.scatter(xs, ys, marker="*", c=self.colors[0:self.mab.K])
        
    def upon_policy(self, b):
        if b.description == 'Optimal':
            plot_policy(self.ax2, self.hist, self.colors, self.context, mab=self.mab)
        else:
            plot_policy(self.ax2, self.hist, self.colors, self.context, ucb=(b.description == 'LinUCB'))

def plot_policy(ax, hist, colors, context, ucb=False, mab=None):
    """
    Plots a visual representation of different policies, represented
    by a color map showing which regions of context space correspond to particular actions.

    (If mab is provided, this plots the optimal policy. Otherwise, this plots
    the policy inferred from the history.)

    NOTE: Assumes a 2D context space.

    Args:
        ax (matplotlib.axes.Axes): the axes to plot on
        hist (LinearRewardHistory): the reward history
        colors (list(color): len=K): a list of colors to use for each arm
        context (np.ndarray): the current context
        ucb (bool): whether to use LinUCB or ArgMax policy
        mab (LinearContextualBandit): the ground-truth bandit
    """
    x = np.linspace(-10,10,100+1)
    y = np.linspace(-10,10,100+1)
    zz = get_policy(x, y, hist, ucb=ucb, mab=mab)
    ax.clear()
    ax.contourf(x, y, zz, colors=colors, levels=[-0.5,0.5,1.5,2.5], alpha=0.5)
    ax.scatter(context[0], context[1], color='black', marker='x')
    ax.set_title('policy')
    
def get_policy(xs, ys, hist, ucb=False, mab=None):
    """
    Assumes a 2D context space.
    """
    zz = np.zeros([len(xs), len(ys)])
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            context = np.array([x,y])
            zz[i,j] = get_argmax(context, hist, ucb=ucb, mab=mab)
    return zz.T


def get_argmax(context, hist, ucb=False, mab=None):
    ests = []
    for arm in hist.actions:
        if mab is None:
            mean, s, u = hist.ci[arm]
            if mean is not None and not ucb:
                ests.append(np.dot(mean, context))
            elif ucb and s is not None:
                pass
                est = np.dot(mean, context) + np.linalg.norm(np.diag(1/s) @ u @ context)
                ests.append(est)
            else:
                ests.append(-np.inf)
        else:
            ests.append(np.dot(mab.mus[arm], context))
    if np.max(ests) == np.inf:
        return None
    else:
        return np.argmax(ests)

# ROHAN: Modified to work with contexts
def greedy_policy(hist, context, rng):
    """
    A greedy policy that chooses the highest mean action, breaking ties randomly.
    [Note: fully deterministic]

    If there are unexplored actions (i.e. actions that have not been pulled at least once),
    then defaults to pulling arms in sequence.
    """
    unexplored = hist.get_unexplored_actions(threshold=1)
    if len(unexplored) > 0:
        print("Initial exploration: pulling all arms at least once...")
        return unexplored[0]
    return hist.get_highest_mean(context)

def linucb_policy(hist, context, rng, alpha=1, lam=0.1):
    """
    Policy that uses the LinUCB algorithm to select actions.
    [Note: fully deterministic]
    Also, records UCB data to history object.

    Args:
        alpha (float): the exploration parameter
        lam (float): the regularization parameter, used to ensure that the data matrix is invertible
    """
    # Initial exploration phase guarantees that every arm has at least 1 sample.
    unexplored = hist.get_unexplored_actions(threshold=1)
    if len(unexplored) > 0:
        print("Initial exploration: pulling all arms at least once...")
        return unexplored[0]
    ucb_list, means, bonuses = hist.get_ucb(context, alpha=alpha, svd_condition="loose", lam=lam)
    hist.record_ucb(ucb_list, means, bonuses, alpha=alpha)
    ret = np.argmax(ucb_list)
    return ret

def linucb_rank_policy(hist, context, rng, alpha=1, lam=0.1):
    """
    Policy that uses the LinUCB algorithm to select actions.
    [Note: fully deterministic]
    Also, records UCB data to history object.

    Args:
        alpha (float): the exploration parameter
        lam (float): the regularization parameter, used to ensure that the data matrix is invertible
    """
    # Initial exploration phase guarantees that every arm has at least 1 sample.
    unexplored = hist.get_unexplored_actions(threshold=1)
    if len(unexplored) > 0:
        print("Initial exploration: pulling all arms at least once...")
        return unexplored[0], np.arange(10)
    ucb_list, means, bonuses = hist.get_ucb(context, alpha=alpha, svd_condition="loose", lam=lam)
    hist.record_ucb(ucb_list, means, bonuses, alpha=alpha)
    ret1 = np.argmax(ucb_list)
    ret2 = np.argsort(ucb_list)[::-1]
    return ret1, ret2

def linucb_rank_policy_data(hist, context, rng, alpha=1, lam=0.1):
    """
    Policy that uses the LinUCB algorithm to select actions.
    [Note: fully deterministic]
    Also, records UCB data to history object.

    Args:
        alpha (float): the exploration parameter
        lam (float): the regularization parameter, used to ensure that the data matrix is invertible
    """
    # Initial exploration phase guarantees that every arm has at least 1 sample.
    unexplored = hist.get_unexplored_actions(threshold=1)
    if len(unexplored) > 0:
        print("Initial exploration: pulling all arms at least once...")
        return unexplored[0], np.arange(10), np.random.rand(10), np.random.rand(10)
    ucb_list, means, bonuses = hist.get_ucb(context, alpha=alpha, svd_condition="loose", lam=lam)
    hist.record_ucb(ucb_list, means, bonuses, alpha=alpha)
    ret1 = np.argmax(ucb_list)
    ret2 = np.argsort(ucb_list)[::-1]
    return ret1, ret2, means, bonuses