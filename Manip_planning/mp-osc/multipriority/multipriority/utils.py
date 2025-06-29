import os
import multipriority
import math
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import yaml
import time
import threading
import multiprocessing
import pickle
import random
import queue
from scipy.spatial.transform import Rotation as R
from pyrcareworld.utils.skeleton_visualizer import SkeletonVisualizer
import atexit
import json

MAX_CONTACTS = 5

class Taxel:
    def __init__(self, taxel_id, link_id, local_pos, normal, max_sensing_force):
        self.taxel_id = taxel_id
        self.link_id = link_id
        self.local_pos = local_pos
        self.normal = normal
        self.force = 0
        self.max_force = max_sensing_force
        self.skeleton_id = 0
        self.pos = None # global pos
        self.contactF = None
        # TODO: create a skeleton part class and attach the max force attributes etc. to it
        # Then everytime taxel makes contact, just update the skeleton_in_contact member variable for that taxel
        self.skeleton_in_contact = None
        self.unity_to_pb_transform = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
        self.pb_local_transform = np.array([[0,0,1], [0, -1, 0], [1, 0, 0]])
    
    def get_contact_info(self):
        return self.contactF, self.normal, self.local_pos, self.link_id
    
    def update(self, current_force, taxel_pos):
        # update contact info
        self.contactF = current_force
        self.pos = taxel_pos

    def calculate_local_position(self, world_to_link_trans_pb):
        # calculate local position of taxel in end effector frame
        taxel_pos_pb = np.dot(self.unity_to_pb_transform, np.array(self.pos))
        self.local_pos = np.dot(np.linalg.inv(world_to_link_trans_pb), np.array([taxel_pos_pb[0], taxel_pos_pb[1], taxel_pos_pb[2], 1]))[:3]
        return self.local_pos

class Tracker:
    def __init__(self, trajectory, lookahead=0.1):
        self.trajectory = trajectory
        self.lookahead = lookahead
        self.cutoff = lookahead * 0.8
        self.current_idx = 0
    
    def get_next_waypoint(self, current_pose):
        # find the next waypoint in the trajectory
        # that is at least lookahead distance away
        while True:
            dist = np.linalg.norm(np.array(current_pose) - np.array(self.trajectory[self.current_idx][0]))
            print ("Distance: ", dist)
            if dist > self.cutoff:
                next_wp = current_pose + ((self.trajectory[self.current_idx][0] - current_pose) / dist) * self.lookahead
                break
            else:
                next_wp = self.trajectory[self.current_idx][0]
                self.current_idx = min(self.current_idx + 1, len(self.trajectory) - 1)
                # print("Almost reached the last waypoint!")
                break
        print ("Next waypoint: ", next_wp)
        # print ("Next waypoint: ", next_wp)
        return np.array(next_wp), np.array(self.trajectory[self.current_idx][1])

class Plotter:
    def __init__(self, labels, plots):
        """
        Initialize the plots. Each plot in 'plots' is a dictionary with keys corresponding to data labels,
        and values corresponding to the initial data points.

        Parameters:
        labels (lst of str): a list of titles that are shown on y-axis
        plots (lst of dict): a list of plots dictionary
        """
        self.plots = plots
        self.labels = labels
        self.plot_command_queue = multiprocessing.Queue()
        self.is_running = multiprocessing.Value('b', False)

    def start(self):
        """
        Start the plotting process in a separate process.
        """
        if not self.is_running.value:
            self.plot_process = multiprocessing.Process(target=self.run)
            self.is_running.value = True
            self.plot_process.start()

    def run(self):
        """
        The plotting logic to be run in a separate process.
        """
        plt.ion()
        num_plots = len(self.plots)
        self.fig, self.axes = plt.subplots(num_plots, 1, figsize=(10, 6))
        self.axes = np.atleast_1d(self.axes)

        self.lines = []
        for ax, plot, label in zip(self.axes, self.plots, self.labels):
            print(plot)
            ax.set_title(label)
            if len(plot) == 1:
                for label, data in plot.items():
                    line, = ax.plot([0], [data], label=label)
                    ax.set_ylabel(label)
                    self.lines.append([line])
                ax.axhline(y=0, color='k',linestyle='--')
                self.symmetrize_y_axis(ax)
            else:
                bars = ax.bar(plot.keys(), plot.values())
                for bar in bars:
                    bar.set_label(bar.get_x())
                self.lines.append(bars)
                ax.legend()
        
        while self.is_running.value:
            try:
                command, args = self.plot_command_queue.get(timeout=1)
                getattr(self, command)(*args)
                plt.draw()
                plt.pause(0.1)
            except queue.Empty:
                pass
            except EOFError:
                break

        plt.close(self.fig)

    def update_plot(self, plot_index, data_dict):
        """
        Add a command to the queue to update the plot with new data.
        """
        self.plot_command_queue.put(('internal_update_plot', (plot_index, data_dict)))

    def symmetrize_y_axis(self, axes):
        y_max = np.abs(axes.get_ylim()).max()
        axes.set_ylim(ymin=-y_max, ymax=y_max)

    def internal_update_plot(self, plot_index, data_dict):
        """
        Internal method to update the plot. This method is executed by the plot process.
        """
        if plot_index < 0 or plot_index >= len(self.lines):
            raise IndexError("Plot index out of range.")

        ax = self.axes[plot_index]
        plot_type = len(self.plots[plot_index])

        if plot_type == 1:
            line = self.lines[plot_index][0]
            x_data, y_data = line.get_xdata(), line.get_ydata()
            x_data = np.append(x_data, len(x_data))
            y_data = np.append(y_data, next(iter(data_dict.values())))

            line.set_xdata(x_data)
            line.set_ydata(y_data)
            max_height = max(data_dict.values()) * 1.5
            ax.set_ylim(0, max_height)
            self.symmetrize_y_axis(ax)
            ax.relim()
            ax.autoscale_view()
        else:
            bars = self.lines[plot_index]
            for bar, key in zip(bars, data_dict.keys()):
                bar.set_height(data_dict[key])

            ax.legend()
            # Recalculate and set y-axis limits for bar plot
            max_height = max(data_dict.values()) * 1.5
            ax.set_ylim(0, max_height)

        self.axes[plot_index].relim()
        self.axes[plot_index].autoscale_view()

    def stop(self):
        """
        Stop the plotting process.
        """
        self.is_running.value = False
        self.plot_command_queue.close()
        self.plot_command_queue.join_thread()
        if self.plot_process and self.plot_process.is_alive():
            self.plot_process.join()

    def show(self):
        plt.show()


class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())

def filter_active_taxels(forces, ids, positions, threshold=0.0):
    """
    This function filters out the inactive taxels and sorte the taxels by force value
    """
    # filter out taxels with force magnitude below threshold
    forces = np.array(forces)
    # print("forces in filter_active_taxels: ", forces)
    sorted_idx = np.argsort(forces)[::-1]
    # print("sorted_idx in filter_active_taxels: ", sorted_idx)
    forces = forces[sorted_idx]
    ids = np.array(ids)
    ids = ids[sorted_idx]
    positions = np.array(positions)
    positions = positions[sorted_idx]
    mask = np.where(forces > threshold)
    # print("mask in filter_active_taxels: ", mask)
    return forces[mask], ids[mask], positions[mask]

def filter_active_taxels_soft(sorted_active_dict, soft_threshold):
    """This function returns a filtered dictionary with soft threshold

    Args:
        sorted_active_dict (dict): The dictionary returned from get_active_taxel_dict()
        soft_threshold (lst): a list of soft threshold for each body part
    """
    filtered_dict = {}
    for index, (key, value) in enumerate(sorted_active_dict.items()):
        column_index = None
        
        if value.size == 0:
            filtered_dict[key] = value
            continue
        
        for i in range(value.shape[1]):
            if value[1, i] < soft_threshold[index]:
                column_index = i
                break
        if column_index is not None:
            filtered_dict[key] = value[:, :column_index]
        else:
            filtered_dict[key] = value
    return filtered_dict

def get_skeleton_id(skeleton_pose, taxel_pose, threshold=0.2):
    min_distance = 100000
    min_id = None
    # print ("Skel pses: ", skeleton_pose)
    for i, pose in enumerate(skeleton_pose):
        dist = np.linalg.norm(pose - taxel_pose)
        if dist < min_distance:
            min_id = i
            min_distance = dist
    return min_id

def load_yaml(filename):
    config_path = Path(__file__).parents[1] / f'configs/{filename}'
    yaml_dict = yaml.load(open(config_path), Loader=yaml.FullLoader)
    return yaml_dict

def save_yaml(filename, data):
    config_path = Path(__file__).parents[1] / f'configs/{filename}'
    with open(config_path, 'w') as f:
        yaml.dump(data, f)

def load_taxel_dict(filename, obs):
    taxel_yaml = load_yaml(filename)
    taxel_dict = {}
    for taxel_id in taxel_yaml:
        taxel_info = taxel_yaml[taxel_id]
        taxel_obj = Taxel(taxel_id, taxel_info['Link'], taxel_info['Position'], taxel_info['Normal'])
        taxel_dict[taxel_id] = taxel_obj
    for i, tid in enumerate(obs['skin']['ids']):
        taxel_dict[tid].skeleton_id = i
    return taxel_dict

def load_trajectory(filename):
    data = np.load(Path(__file__).parents[1] / f'data/{filename}', allow_pickle=True)
    tmp = data['arr_0']
    traj = []
    for i in range(len(tmp)):
        traj.append([tmp[i]['ee_pos'], R.from_matrix(tmp[i]['ee_ori']).as_quat()])
    return traj

def get_active_taxel_dict(ids, forces, positions, contact_params, taxel_dict, obs_skin, is_us, skeleton_pos = None, not_raw = True):
    """
    This function takes the input from filter_active_taxel and preprocess an active taxel dictionary

    Parameters:
        ids: filtered sorted ids from filter_active_taxel()
        forces: filtered sorted forces from filter_active_taxel()
        contact_params: contact_params dictionary
        skeleton_pos: the position of each skeleton
        is_us (bool): if this is a user study

    Returns:
        active_taxel_dict: a processed active taxel dictionary
        id_to_skeleton: a dictionary that maps the taxel id to its contact skeleton id
    """
    active_taxel_dict = {key: np.array([[],[]]) for key in contact_params.keys()}
    selected_taxel_ids = []
    raw_active_taxel_dict = {} # store raw force values in this dictionary for viz
    real_id_dict = {0:0, 1:1, 2:1, 3:2, 4:3, 5:4, 6:5}
    # print("obs_skin is ", obs_skin)
    # print("obs_skin is ", obs_skin["skeleton_ids"])
    for id, force, position in zip(ids, forces, positions):
        if is_us:
            skt = obs_skin["skeleton_ids"][id-1000]
            if skt == -1:
                continue
            skt = real_id_dict[skt]
            print("Skt: ", skt)
        else:
            skt = get_skeleton_id(skeleton_pose=skeleton_pos, taxel_pose=position)

        if skt is None:
            skt = 0
        if force > contact_params[f'b{skt}']['force'] * contact_params[f'b{skt}']['soft_factor'] and not_raw:
            selected_taxel_ids.append(id)
            taxel_dict[id].skeleton_id = skt
            taxel_dict[id].contactF = force
            taxel_dict[id].pos = position
            active_taxel_dict[f'b{skt}'] = np.hstack((active_taxel_dict[f'b{skt}'], np.array([[id], [force]])))
            
        elif not not_raw:
            selected_taxel_ids.append(id)
            taxel_dict[id].skeleton_id = skt
            taxel_dict[id].contactF = force
            taxel_dict[id].pos = position
            active_taxel_dict[f'b{skt}'] = np.hstack((active_taxel_dict[f'b{skt}'], np.array([[id], [force]])))
    
    return active_taxel_dict, taxel_dict, np.array(selected_taxel_ids)

def update_threshold(sorted_active_dict, contact_params, feedback_t, dec_factor_1, dec_factor_2):
    """
    This function updates the force threshold for each body part

    Parameters:
        sorted_active_dict: a dictionary that maps active taxels to each controller
            expected_format: {"b1": [ids;
                                    forces],
                            "b2":...}
        contact_params: params for each controller
        feedback_t: the feedback received from the user
        dec_factor_1: the adjustment percentage when force is too much
        dec_factor_2: the adjustment percentage when force is a little bit large
    """

    for i, key in enumerate(sorted_active_dict.keys()):
        if feedback_t[i] < 3:
            if len(sorted_active_dict[key]) != 0:
                taxels = sorted_active_dict[key]
                if taxels.shape[1] == 0:
                    continue
                effective_force = taxels[1, 0]
                prev_force = contact_params[key]["force"]
                if feedback_t[i] == 1:
                    contact_params[key]["force"] = prev_force - abs(effective_force - prev_force) * dec_factor_1
                elif feedback_t[i] == 2:
                    contact_params[key]["force"] = prev_force - abs(effective_force - prev_force) * dec_factor_2
    return contact_params

def log_data(filename, names, *args):
        """
        A log function that saves the data automatically when the program stops execution

        Parameters:
            filename (str): the path and name of the file
            names (list): a list of string that contains the names of dictionary key
            *args (lst): lists of to-be-saved data

        Requires:
            len(names) == # *args
        """
        assert len(args) == len(names)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'log/{filename}_{timestamp}.pkl'
        data = {}
        for name, arg in zip(names, *args):
            data[name] = arg
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"{filename} successfully saved to pickle file.")

def clear(array, id):
    """
    This function looks for the corresponding id in the first row of the input and remove the column

    Parameters:
    array (2d np.array): an array whose 1st row is id, and 2nd row is corresponding force
    id (str): the id that needs to be clear
    """
    matching_columns = np.where(array[0] == id)[0]
    array = np.delete(array, matching_columns, axis=1)
    return array

def in_range(force, soft_threshold, hard_threshold):
    """
    This function evaluates the force value in the range of soft_threshold and hard_threshold

    Parameters:
    force: The force level that is sensed by the taxel
    soft_threshold: The soft threshold corresponding to the contact part
    hard_threshold: The hard threshold corresponding to the contact part

    Returns:
    """
    if force <= soft_threshold:
        return True, soft_threshold - force
    else:
        return False, hard_threshold - force

def heuristic_based_priority(sorted_active_dict, skeleton_contact_params):
    """
    This function calculates the priority of each body part based on the heuristic function
    that the priority is inversely proportional to the force difference between current force and hard threshold.
    For negative force difference it is directly proportional to the force difference magnitude.
    
    Parameters:
        sorted_active_dict: a dictionary that maps active taxels to each controller
            expected_format: {"b1": [ids;
                                    forces],
                            "b2":...}
        skeleton_contact_params: a dictionary that maps the body part to its contact params
            expected_format: {"b1": {"force": force_threshold,
                                    "Kp": Kp,
                                    "Kd": Kd},
                            "b2":...}
    Returns:
        priority: a list that maps the body part to its priority
    """
    priority = np.empty(len(sorted_active_dict.keys()))
    for key in sorted_active_dict:
        if len(sorted_active_dict[key][1]) == 0:
            priority[int(key[1])] = 100000.0
            continue
        taxels = sorted_active_dict[key]
        force = taxels[1, 0]
        #print ("Force: ", force)
        force_diff = skeleton_contact_params[key]["goalF"] - force
        if force_diff < 0:
            priority[int(key[1])] = 1/force_diff
        else:
            priority[int(key[1])] = force_diff
    
    priority = np.argsort(priority)
    return priority

def update_plot_dict(plt_dict, vals):
    """
    This function updates the information in plt_dict with each entry in the vals
    """
    return {key:vals[i] for i, key in enumerate(plt_dict.keys())}

def generate_priority(size):
    """
    This function generates a shuffled priority list with given size
    """
    lst = np.array(list(range(size)))
    np.random.shuffle(lst)
    lst.tolist()
    return lst

def generate_user_with_unique_priority():
    """
    Generates a user profile with unique priorities for each body part.
    """
    body_parts = ['Head', 'Left Arm', 'Right Arm', 'Left Leg', 'Right Leg', 'Torso']
    body_parts = [f'b{i}' for i in range(len(body_parts))]
    personality = {}
    priority_values = list(range(1, len(body_parts) + 1))
    random.shuffle(priority_values)

    for part, priority in zip(body_parts, priority_values):
        desired_force = random.randint(1, 5)
        sensitivity = random.uniform(0.5, 2.0)
        personality[part] = {
            'desired_force': desired_force, 
            'sensitivity': sensitivity,
            'priority': priority
        }
    
    return personality

def simulate_feedback(sorted_active_dict, user_profile):
    """
    Simulates feedback given user profile and current forces.
    """
    most_critical_feedback = None
    max_importance = 0

    for key in sorted_active_dict:
        if len(sorted_active_dict[key]) == 0:
            continue
        taxels = sorted_active_dict[key]
        if len(taxels) == 0:
            continue
        force = taxels[1]
        desired_force = user_profile[key]['desired_force']
        sensitivity = user_profile[key]['sensitivity']
        priority = user_profile[key]['priority']

        # Calculate deviation and importance
        deviation = abs(force - desired_force)
        importance = deviation * sensitivity / priority
        
        if not most_critical_feedback or importance > max_importance:
            max_importance = importance

            # Determine feedback rating
            if force < desired_force:
                rating = 4 if force >= desired_force - sensitivity else 5
            elif force > desired_force:
                rating = 2 if force <= desired_force + sensitivity else 1
            else:
                rating = 3

            most_critical_feedback = (key, rating)

    return most_critical_feedback

def final_print(interval, time_stam, stucked = None):
    print(f"Interval is {interval}")
    print(f"It stops at {time_stam}")
    print(f'Robot starts to get stucked at {stucked}')

def final_log(tp, names, datas, purpose, vis_process, start_process):
    for name, data in zip(names, datas):
        path = os.path.abspath(__file__)
        print("-----------------------")
        print(f"/multipriority/{purpose}/{name} is saved to {path}")
        np.savez_compressed(f"multipriority/user_study_data/{tp}_{name}", np.array(data))
    # atexit._run_exitfuncs()
    # vis_process.terminate()
    # start_process.terminate()
        
def start_vis(forces_queue):
    skeleton_visualizer = SkeletonVisualizer()
    skeleton_visualizer.show()
    while True:
        skeleton_visualizer.update(forces_queue)

def start(forces_queue, shared_dict, stop_event):
    while not stop_event.is_set():
        time.sleep(.018)
        forces = shared_dict.copy()
        print("Forces being put: ", forces)
        forces_queue.put(forces)

def final_plot(data_list):
    """
    Plots a line graph from a list of numerical values.
    
    :param data_list: List of numerical values to plot.
    """
    if not isinstance(data_list, list):
        raise ValueError("Input must be a list of numerical values.")
    
    if not all(isinstance(item, (int, float)) for item in data_list):
        raise ValueError("All items in the list must be numerical values.")

    plt.figure(figsize=(10, 6)) 
    plt.plot(data_list, marker='o', linestyle='-', color='b') 
    plt.title('Line Plot of the Input List') 
    plt.xlabel('Index') 
    plt.ylabel('Value') 
    plt.grid(True)  
    plt.show()

def compute_position_err(obs, final_pos, axiswise=False):
    """
    Computes the position error between the current position and the final position.
    """
    curr_pos = np.array(obs['robot']['positions'][7])
    pos_err = np.abs(curr_pos - final_pos)
    if axiswise:
        return pos_err
    return np.linalg.norm(pos_err)

def load_yaml_grid(filename):
    yaml_dict = yaml.load(open(filename), Loader=yaml.FullLoader)
    return yaml_dict

def log_eval_data(log_folder_name, summary_dict, eval_ep_forces, eval_ep_position_err, eval_ep_priority, priority_params, contact_params):
    """
    Logs the evaluation data.
    """
    eval_ep_forces = np.array(eval_ep_forces)
    eval_ep_position_err = np.array(eval_ep_position_err)
    folder_path = Path(__file__).parents[1] / f'logs/{log_folder_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # save summary dict as json
    with open(folder_path / 'summary.json', 'w') as f:
        json.dump(summary_dict, f)

    # save priority params as yaml
    with open(folder_path / 'priority_params.yaml', 'w') as f:
        yaml.dump(priority_params, f)
    
    # save contact params as yaml
    with open(folder_path / 'contact_params.yaml', 'w') as f:
        yaml.dump(contact_params, f)
    
    # eval_ep_forces is timesteps x num_body_parts
    # generate a plot with num_body_parts subplots
    # make sure each plot is large enough to see
    num_body_parts = eval_ep_forces.shape[1]
    fig, axes = plt.subplots(num_body_parts, 1, figsize=(10, 6 * num_body_parts))
    for i in range(num_body_parts):
        axes[i].plot(eval_ep_forces[:, i])
        axes[i].set_ylabel(f'Body Part {i} Force')
        axes[i].set_xlabel('Timesteps')
    fig.suptitle('Force vs. Timesteps')
    fig.savefig(folder_path / 'forces.png')
    plt.close(fig)
    
    # eval_ep_position_err is timesteps x 1
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    axes.plot(eval_ep_position_err)
    axes.set_ylabel('Position Error')
    axes.set_xlabel('Timesteps')
    fig.suptitle('Position Error vs. Timesteps')
    fig.savefig(folder_path / 'position_err.png')
    plt.close(fig)
    
    # eval_ep_priority is timesteps x num_body_parts
    # generate animation showing priority over time 
    # Setting up the figure for the bar plot animation
    # Setting up the figure for the bar plot animation
    data = np.array(eval_ep_priority)
    new_data = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        if len(data[i]) < 6:
            print ("Data issues")
            raise ValueError
        for j in range(data.shape[1]):
            new_data[i, int(data[i, j])] = 1 / (j + 1)
    data = new_data
    print ("Data shape: ", data.shape)
    # the value at index i is the body part number and the index is the priority
    # we plot the priority corresponding to the body part number
    
    entities = [2,4,6,9]# data.shape[1]
    labels = np.array([f"Body Part {i}" for i in entities])
    labels[-1] = "Task"
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Using the same sample data as before
    timesteps = data.shape[0]

    # Plotting each entity's data as a line
    for i, val in enumerate(entities):
        ax.plot(range(timesteps), data[:, val], marker='o', label=labels[i])

    # Adding details
    # ax.set_xticks(range(timesteps))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Priority")
    ax.set_title("Priority Over Time")
    ax.legend()
    ax.grid(True)
    # save the plot
    fig.savefig(folder_path / 'priority_static.png')
    plt.close(fig)

    # # Setting up the figure for the updated bar plot animation
    # fig, ax = plt.subplots(figsize=(10, 6))
    # bars = ax.bar(labels, data[0, [2,4,6,9]], color=plt.cm.viridis(np.linspace(0, 1, len(entities))))

    # # Setting the axes limits and labels
    # ax.set_ylim(0, 1)
    # ax.set_xlabel("Body Parts / Task")
    # ax.set_ylabel("Priority")
    # ax.set_title("Priority change over time")
    # ax.tick_params(axis='x', labelrotation=30)

    # # Initialization function for the animation
    # def init():
    #     for bar in bars:
    #         bar.set_height(0)
    #     return bars

    # # Animation function: called sequentially
    # def animate(i):
    #     for bar, height in zip(bars, data[i, [2,4,6,9]]):
    #         bar.set_height(height)
    #     return bars

    # Creating the updated bar plot animation
    # ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(data), interval=15, blit=True)

    # Saving the animation as an MP4 file
    # mp4_filename_updated_bar = folder_path / 'priority.mp4'
    # ani.save(mp4_filename_updated_bar, writer='ffmpeg', dpi=80)
    
    # Also save all the arrays as npy files
    np.save(folder_path / 'forces.npy', eval_ep_forces)
    np.save(folder_path / 'position_err.npy', eval_ep_position_err)
    np.save(folder_path / 'priority.npy', eval_ep_priority)