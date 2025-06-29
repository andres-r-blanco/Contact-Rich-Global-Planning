"""
This module provides dataset classes for different types of state representations.
"""

import os
import torch
import numpy as np
import pickle5 as pickle
import random

from cfg.train_cfg import TrainConfig

import time
import h5py

class ToyEnvDatasetAE(torch.utils.data.Dataset):
    """
    Dataset loader that parses ToyEnv dataset. The dataset consists of npz arrays with
    parent key `arr_0` containing the state-action pairs. The observations at each timestep consist of
    the following features:
        -Joint position ("joint_pos")
        -Joint velocity ("joint_vel")
        -Joint force ("joint_force")
        -Skin force ("skin")
        -Skin IDs that are needed to sort the data correctly ("skin_ids")
    The joint velocity at current timestep is used as the action for the next timestep.
    """
    def __init__(self, cfg: TrainConfig):
        x_trajectories_list = []
        y_trajectories_list = []

        count = 0
        files = os.listdir(os.path.join(cfg.root_dir, cfg.data_dir))
        # files.sort()
        for file in files:
            count += 1
            if count > 1:
                break
            if file.endswith(".npz"):
                print(f"Loading data from {file}")
                with np.load(os.path.join(cfg.root_dir, cfg.data_dir, file), allow_pickle=True) as data:
                    data = data["arr_0"]

                    states = np.array([np.concatenate([data[i][key] for key in cfg.obs_types]) for i in range(len(data))])
                    actions = np.array([data[i]['action'] for i in range(len(data))])
                    states[:, 14:] = np.where(states[:, 14:] < 0.05, 0, 1)
                    print (states.shape, actions.shape)
                    states = torch.tensor(states, dtype=torch.float32)
                    actions = torch.tensor(actions, dtype=torch.float32)
                    # Construct the input and output tensors
                    x, y = construct_input_output_v3(cfg, states, actions)

                    # Append the trajectories to the list
                    x_trajectories_list.append(x)
                    y_trajectories_list.append(y)

        # Concatenate the trajectories
        self.x_trajectories = torch.cat(x_trajectories_list, dim=0)
        self.y_trajectories = torch.cat(y_trajectories_list, dim=0)

        # Normalize the data to the range [-1, 1]
        if cfg.normalization:
            self.x_trajectories, self.x_min_vals, self.x_max_vals = normalize(
                self.x_trajectories
            )
            self.y_trajectories, self.y_min_vals, self.y_max_vals = normalize(
                self.y_trajectories
            )

        # Get the number of trajectories
        self.num_trajectories = self.x_trajectories.shape[0]
        print(f"Number of trajectories: {self.num_trajectories}")

        # Set the input and output sizes for the network
        cfg.network_config.input_size = self.x_trajectories.shape[-1]
        cfg.network_config.output_size = self.y_trajectories.shape[-1]

    def __len__(self):
        return self.num_trajectories
    
    def __getitem__(self, idx):
        return self.x_trajectories[idx], self.y_trajectories[idx]
    
    def get_min_max_vals(self):
        """
        Get the minimum and maximum values of the data for denormalization.
        """
        if hasattr(self, "x_min_vals"):
            return {
                "x_min_vals": self.x_min_vals,
                "x_max_vals": self.x_max_vals,
                "y_min_vals": self.y_min_vals,
                "y_max_vals": self.y_max_vals,
            }
        else:
            return None

class ToyEnvDataset(torch.utils.data.Dataset):
    """
    Dataset loader that parses ToyEnv dataset. The dataset consists of npz arrays with
    parent key `arr_0` containing the state-action pairs. The observations at each timestep consist of
    the following features:
        -Joint position ("joint_pos")
        -Joint velocity ("joint_vel")
        -Joint force ("joint_force")
        -Skin force ("skin")
        -Skin IDs that are needed to sort the data correctly ("skin_ids")
    The joint velocity at current timestep is used as the action for the next timestep.
    """
    def __init__(self, cfg: TrainConfig):
        x_trajectories_list = []
        y_trajectories_list = []

        count = 0
        files = os.listdir(os.path.join(cfg.root_dir, cfg.data_dir))
        # files.sort()
        for file in files:
            count += 1
            # if count > 100:
            #     break
            if file.endswith(".npz"):
                print(f"Loading data from {file}")
                with np.load(os.path.join(cfg.root_dir, cfg.data_dir, file), allow_pickle=True) as data:
                    data = data["arr_0"]

                    states = np.array([np.concatenate([data[i][key] for key in cfg.obs_types]) for i in range(len(data))])
                    actions = np.array([data[i]['action'] for i in range(len(data))])
                    states[:, 14:] = np.where(states[:, 14:] < 0.05, 0, 1)
                    print (states.shape, actions.shape)
                    states = torch.tensor(states, dtype=torch.float32)
                    actions = torch.tensor(actions, dtype=torch.float32)
                    # Construct the input and output tensors
                    x, y = construct_input_output_v3(cfg, states, actions)

                    # Append the trajectories to the list
                    x_trajectories_list.append(x)
                    y_trajectories_list.append(y)

        # Concatenate the trajectories
        self.x_trajectories = torch.cat(x_trajectories_list, dim=0)
        self.y_trajectories = torch.cat(y_trajectories_list, dim=0)

        # Normalize the data to the range [-1, 1]
        if cfg.normalization:
            self.x_trajectories, self.x_min_vals, self.x_max_vals = normalize(
                self.x_trajectories
            )
            self.y_trajectories, self.y_min_vals, self.y_max_vals = normalize(
                self.y_trajectories
            )

        # Get the number of trajectories
        self.num_trajectories = self.x_trajectories.shape[0]
        print(f"Number of trajectories: {self.num_trajectories}")

        # Set the input and output sizes for the network
        cfg.network_config.input_size = self.x_trajectories.shape[-1]
        cfg.network_config.output_size = self.y_trajectories.shape[-1]

    def __len__(self):
        return self.num_trajectories
    
    def __getitem__(self, idx):
        return self.x_trajectories[idx], self.y_trajectories[idx]
    
    def get_min_max_vals(self):
        """
        Get the minimum and maximum values of the data for denormalization.
        """
        if hasattr(self, "x_min_vals"):
            return {
                "x_min_vals": self.x_min_vals,
                "x_max_vals": self.x_max_vals,
                "y_min_vals": self.y_min_vals,
                "y_max_vals": self.y_max_vals,
            }
        else:
            return None
        
class ToyEnvDataset_wm(torch.utils.data.Dataset):
    """
    Dataset loader that parses ToyEnv dataset. The dataset consists of npz arrays with
    parent key `arr_0` containing the state-action pairs. The observations at each timestep consist of
    the following features:
        -Joint position ("joint_pos")
        -Joint velocity ("joint_vel")
        -Joint force ("joint_force")
        -Skin force ("skin")
        -Skin IDs that are needed to sort the data correctly ("skin_ids")
    The joint velocity at current timestep is used as the action for the next timestep.
    """
    def __init__(self, cfg: TrainConfig):
        x_trajectories_list = []
        y_trajectories_list = []
        a_trajectories_list = []
        x_trajectories_list_test = []
        y_trajectories_list_test = []
        a_trajectories_list_test = []

        random.seed(cfg.seed)

        files_all = os.listdir(os.path.join(cfg.root_dir, cfg.data_dir))
        random.shuffle(files_all)
        split_index = int(len(files_all) * cfg.train_split)
        train_files = files_all[:split_index]
        test_files = files_all[split_index:]
        # random.seed(cfg.seed)
        # h5_files = [file for file in files_all if file.endswith(".h5")]
        # fixed_size = cfg.data_size
        # files = random.sample(h5_files, fixed_size)

        start = time.time()
        for file in train_files:
            data = h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file), 'r')
            for key in list(data.keys()):
                hdf5_data = data[key]
                states = np.concatenate([hdf5_data[key] for key in cfg.obs_types], axis=1)
                actions = np.array(hdf5_data['action'])
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                # Construct the input and output tensors
                x, y, a = construct_input_output_v5(cfg, states, actions)
                # Append the trajectories to the list
                x_trajectories_list.append(x)
                y_trajectories_list.append(y)
                a_trajectories_list.append(a)

        for file in test_files:
            data = h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file), 'r')
            for key in list(data.keys()):
                hdf5_data = data[key]
                states = np.concatenate([hdf5_data[key] for key in cfg.obs_types], axis=1)
                actions = np.array(hdf5_data['action'])
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                # Construct the input and output tensors
                x, y, a = construct_input_output_v5(cfg, states, actions)
                # Append the trajectories to the list
                x_trajectories_list_test.append(x)
                y_trajectories_list_test.append(y)
                a_trajectories_list_test.append(a)
        # for file in files_all:
        #     data = h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file), 'r')
        #     for key in list(data.keys()):
        #         hdf5_data = data[key]
        #         states = np.concatenate([hdf5_data[key] for key in cfg.obs_types], axis=1)
        #         actions = np.array(hdf5_data['action'])
        #         states = torch.tensor(states, dtype=torch.float32)
        #         actions = torch.tensor(actions, dtype=torch.float32)
        #         # # Construct the input and output tensors
        #         x, y, a = construct_input_output_v5(cfg, states, actions)
        #         # Append the trajectories to the list
        #         x_trajectories_list.append(x)
        #         y_trajectories_list.append(y)
        #         a_trajectories_list.append(a)

        end = time.time()
        print(f"Time taken to load data: {end - start}")
        # exit(0)

        # Concatenate the trajectories
        self.x_trajectories = torch.cat(x_trajectories_list, dim=0)
        self.y_trajectories = torch.cat(y_trajectories_list, dim=0)
        self.a_trajectories = torch.cat(a_trajectories_list, dim=0)

        self.x_trajectories_test = torch.cat(x_trajectories_list_test, dim=0)
        self.y_trajectories_test = torch.cat(y_trajectories_list_test, dim=0)
        self.a_trajectories_test = torch.cat(a_trajectories_list_test, dim=0)

        # Normalize the data to the range [-1, 1]
        if cfg.normalization:
            self.x_trajectories, self.x_min_vals, self.x_max_vals = normalize(
                self.x_trajectories
            )
            self.y_trajectories, self.y_min_vals, self.y_max_vals = normalize(
                self.y_trajectories
            )
            self.a_trajectories, self.a_min_vals, self.a_max_vals = normalize(
                self.a_trajectories
            )
            self.x_trajectories_test, _, _ = normalize(self.x_trajectories_test)
            self.y_trajectories_test, _, _ = normalize(self.y_trajectories_test)
            self.a_trajectories_test, _, _ = normalize(self.a_trajectories_test)

            # self.x_trajectories_test = normalize(self.x_trajectories_test, self.x_min_vals, self.x_max_vals)
            # self.y_trajectories_test = normalize(self.y_trajectories_test, self.y_min_vals, self.y_max_vals)
            # self.a_trajectories_test = normalize(self.a_trajectories_test, self.a_min_vals, self.a_max_vals)

        # Get the number of trajectories
        self.num_trajectories = self.x_trajectories.shape[0]
        print(f"Number of training trajectories: {self.num_trajectories}")
        self.num_trajectories_test = self.x_trajectories_test.shape[0]  
        print(f"Number of testing trajectories: {self.num_trajectories_test}")

        # Set the input and output sizes for the network
        cfg.network_config.input_size = self.x_trajectories.shape[-1]
        cfg.network_config.output_size = self.y_trajectories.shape[-1]
        cfg.network_config.action_size = self.a_trajectories.shape[-1]
        cfg.network_config.sequence_length = self.x_trajectories.shape[1]

        self.cfg = cfg

    def __len__(self):
        return self.num_trajectories
    
    def __getitem__(self, idx):
        return self.x_trajectories[idx], self.y_trajectories[idx], self.a_trajectories[idx]
    
    def get_min_max_vals(self):
        """
        Get the minimum and maximum values of the data for denormalization.
        """
        if hasattr(self, "x_min_vals"):

            return {
                "x_min_vals": self.x_min_vals,
                "x_max_vals": self.x_max_vals,
                "y_min_vals": self.y_min_vals,
                "y_max_vals": self.y_max_vals,
                "a_min_vals": self.a_min_vals,
                "a_max_vals": self.a_max_vals,
            }
        else:
            return None
class Dataset(torch.utils.data.Dataset):
    """
    Dataset loader that parses ToyEnv dataset. The dataset consists of npz arrays with
    parent key `arr_0` containing the state-action pairs. The observations at each timestep consist of
    the following features:
        -Joint position ("joint_pos")
        -Joint velocity ("joint_vel")
        -Joint force ("joint_force")
        -Skin force ("skin")
        -Skin IDs that are needed to sort the data correctly ("skin_ids")
    The joint velocity at current timestep is used as the action for the next timestep.
    """
    def __init__(self, x , y, a, num_trajectories):
        self.x_trajectories = x
        self.y_trajectories = y
        self.a_trajectories = a
        self.num_trajectories = num_trajectories
    
    def __getitem__(self, idx):
        return self.x_trajectories[idx], self.y_trajectories[idx], self.a_trajectories[idx]
    
    def __len__(self):
        return self.num_trajectories

def construct_input_output(cfg: TrainConfig, states, actions):
    """
    Construct the input and output tensors for training from the raw state
    and action tensors. This includes splitting the data into sequences of
    length `sequence_length`.

    Args:
        cfg (TrainConfig): The training configuration.
        states (torch.Tensor) [Trajectory, State]: The state tensor.

    Returns:
        x (torch.Tensor) [Batch, Sequence, Input]: The input tensor.
        y (torch.Tensor) [Batch, Sequence, Output]: The output tensor.

    """

    # Concatenate the state and action tensors and remove the last state
    if cfg.training_objective == "forward":
        # Calculate the number of elements that can be divided by cfg.sequence_length
        window_size = cfg.sequence_length

        # padd x by window size with the first value of x
        states_ = torch.zeros((states.shape[0] + window_size, states.shape[1]))
        states_[:window_size] = states[0]
        states_[window_size:] = states

        states_seq = torch.zeros((states.shape[0], states.shape[1] * window_size))
        for i in range(states.shape[0]):
            states_seq[i, :] = states_[i : i + window_size, :].flatten()
        # breakpoint()

        # Concatenate the state and action tensors and remove the last state
        x = torch.cat((states_seq[:-1], actions[:-1]), dim=-1)

        # Concatenate the state tensors and remove the first state
        if cfg.predict_delta:
            # Predict the change in state
            y = states[1:] - states[:-1]
        else:
            # Predict the absolute state
            y = states[1:]
    
    elif cfg.training_objective == "forward_v2":
        # Calculate the number of elements that can be divided by cfg.sequence_length
        # Generate x and y based on prediction horizon. If horizon > 1, predict state_{t+H} from state_{t} and action_{t:t+H-1}
        window_size = cfg.sequence_length

        # padd x by window size with the first value of x
        states_ = torch.zeros((states.shape[0] + window_size, states.shape[1]))
        states_[:window_size] = states[0]
        states_[window_size:] = states

        states_seq = torch.zeros((states.shape[0], states.shape[1] * window_size))
        for i in range(states.shape[0]):
            states_seq[i, :] = states_[i : i + window_size, :].flatten()
        # breakpoint()

        horizon = cfg.horizon
        action_seq = torch.zeros((states.shape[0], actions.shape[1] * horizon))
        for i in range(states.shape[0]):
            action_seq[i, :] = actions[i : i + horizon, :].flatten()
        # breakpoint()
            
        # Concatenate the state and action tensors and remove the last state
        x = torch.cat((states_seq[:-horizon], action_seq), dim=-1)
        # Concatenate the state tensors and remove the first state
        if cfg.predict_delta:
            # Predict the change in state
            y = states[horizon:] - states[:-horizon]
        else:
            # Predict the absolute state
            y = states[horizon:]

    return x, y



class PunyoDataset(torch.utils.data.Dataset):
    """
    Dataset loader that parses Punyo experiment data. The dataset consists of either
    (state-action) - (next state) pairs or (state-next state) - (action) pairs,
    depending on the training objective.

    The state is constructed from optional observation types and actions, where actions
    form the end of the state vector.

    The currently used pkl files contain the following observation types:
        -Joint positions and velocities ("state")
        -Joint torques ("torque")
        -Tactile information ("distal_arm_pressures")
    """

    def __init__(self, cfg: TrainConfig):
        # Load the data from pickle files and convert to torch tensors
        x_trajectories_list = []
        y_trajectories_list = []

        for file in os.listdir(os.path.join(cfg.root_dir, cfg.data_dir)):
            if file.endswith(".pkl"):
                print(f"Loading data from {file}")
                with open(os.path.join(cfg.root_dir, cfg.data_dir, file), "rb") as fh:
                    df = pickle.load(fh)

                obs_types_list = []
                for type in cfg.obs_types:
                    obs_types_list.append(
                        torch.tensor(
                            np.stack(df[("observations", type)].values),
                            dtype=torch.float32,
                        )
                    )
                states = torch.cat(obs_types_list, dim=-1)
                actions = torch.tensor(
                    np.stack(df[("policy_action", "action")].values),
                    dtype=torch.float32,
                )
                # breakpoint()
                # Release memory
                del df

                # Construct the input and output tensors
                # x, y = construct_input_output(cfg, states, actions)
                x, y = construct_input_output_v2(cfg, states, actions)

                # Append the trajectories to the list
                x_trajectories_list.append(x)
                y_trajectories_list.append(y)
                print(x.shape, y.shape)
        # breakpoint()
        # Concatenate the trajectories
        self.x_trajectories = torch.cat(x_trajectories_list, dim=0)
        self.y_trajectories = torch.cat(y_trajectories_list, dim=0)
        # breakpoint()

        # Normalize the data to the range [-1, 1]
        if cfg.normalization:
            self.x_trajectories, self.x_min_vals, self.x_max_vals = normalize(
                self.x_trajectories
            )
            self.y_trajectories, self.y_min_vals, self.y_max_vals = normalize(
                self.y_trajectories
            )

        # Get the number of trajectories
        self.num_trajectories = self.x_trajectories.shape[0]
        print(f"Number of trajectories: {self.num_trajectories}")

        # Set the input and output sizes for the network
        cfg.network_config.input_size = self.x_trajectories.shape[-1]
        cfg.network_config.output_size = self.y_trajectories.shape[-1]

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, idx):
        return self.x_trajectories[idx], self.y_trajectories[idx]

    def get_min_max_vals(self):
        """
        Get the minimum and maximum values of the data for denormalization.
        """
        if hasattr(self, "x_min_vals"):
            return {
                "x_min_vals": self.x_min_vals,
                "x_max_vals": self.x_max_vals,
                "y_min_vals": self.y_min_vals,
                "y_max_vals": self.y_max_vals,
            }
        else:
            return None


def construct_input_output_v2(cfg: TrainConfig, states, actions):
    """
    Construct the input and output tensors for training from the raw state
    and action tensors. This includes splitting the data into sequences of
    length `sequence_length`.

    Args:
        cfg (TrainConfig): The training configuration.
        states (torch.Tensor) [Trajectory, State]: The state tensor.
        actions (torch.Tensor) [Trajectory, Action]: The action tensor.

    Returns:
        x (torch.Tensor) [Batch, Sequence, Input]: The input tensor.
        y (torch.Tensor) [Batch, Sequence, Output]: The output tensor.

    """

    if cfg.training_objective == "forward":
        # Calculate the number of elements that can be divided by cfg.sequence_length
        window_size = cfg.sequence_length

        # padd x by window size with the first value of x
        states_ = torch.zeros((states.shape[0] + window_size, states.shape[1]))
        states_[:window_size] = states[0]
        states_[window_size:] = states

        states_seq = torch.zeros((states.shape[0], states.shape[1] * window_size))
        for i in range(states.shape[0]):
            states_seq[i, :] = states_[i : i + window_size, :].flatten()
        # breakpoint()

        # Concatenate the state and action tensors and remove the last state
        x = torch.cat((states_seq[:-1], actions[:-1]), dim=-1)

        # Concatenate the state tensors and remove the first state
        if cfg.predict_delta:
            # Predict the change in state
            y = states[1:] - states[:-1]
        else:
            # Predict the absolute state
            y = states[1:]

    elif cfg.training_objective == "forward_v2":
        # Calculate the number of elements that can be divided by cfg.sequence_length
        # Generate x and y based on prediction horizon. If horizon > 1, predict state_{t+H} from state_{t} and action_{t:t+H-1}
        window_size = cfg.sequence_length

        # padd x by window size with the first value of x
        states_ = torch.zeros((states.shape[0] + window_size, states.shape[1]))
        states_[:window_size] = states[0]
        states_[window_size:] = states

        states_seq = torch.zeros((states.shape[0], states.shape[1] * window_size))
        for i in range(states.shape[0]):
            states_seq[i, :] = states_[i : i + window_size, :].flatten()
        # breakpoint()

        horizon = cfg.horizon
        action_seq = torch.zeros((states.shape[0] - horizon, actions.shape[1] * horizon))
        for i in range(states.shape[0] - horizon):
                action_seq[i, :] = actions[i : i + horizon, :].flatten()

        # Concatenate the state and action tensors and remove the last state
        x = torch.cat((states_seq[:-horizon], action_seq), dim=-1)

        # Concatenate the state tensors and remove the first state
        if cfg.predict_delta:
            # Predict the change in state
            y = states[horizon:] - states[:-horizon]
        else:
            # Predict the absolute state
            y = states[horizon:]
        
        # Output is arranged as joint_pos (7) + joint_vel (7) + skin (56)
        # Depending on remove_out, remove the corresponding elements from y
        if "joint_vel" in cfg.remove_out and "skin" in cfg.remove_out:
            y = y[:, :-63]
        elif "joint_vel" in cfg.remove_out:
            y = torch.cat((y[:, :7], y[:, 14:]), dim=1)
        elif "skin" in cfg.remove_out:
            y = y[:, :-56]
        print (y.shape)
    return x, y


def construct_input_output_v3(cfg: TrainConfig, states, actions):
    """
    Construct the input and output tensors for training from the raw state
    and action tensors. This includes splitting the data into sequences of
    length `sequence_length`.

    Args:
        cfg (TrainConfig): The training configuration.
        states (torch.Tensor) [Trajectory, State]: The state tensor.
        actions (torch.Tensor) [Trajectory, Action]: The action tensor.

    Returns:
        x (torch.Tensor) [Batch, Sequence, Input]: The input tensor.
        y (torch.Tensor) [Batch, Sequence, Output]: The output tensor.

    """

    if cfg.training_objective == "forward":
        # Concatenate the state and action tensors and remove the last state
        x = torch.cat((states[:-1], actions[:-1]), dim=-1)

        # Concatenate the state tensors and remove the first state
        if cfg.predict_delta:
            # Predict the change in state
            y = states[1:] - states[:-1]
        else:
            # Predict the absolute state
            y = states[1:]

    elif cfg.training_objective == "inverse":
        # Concatenate the state with the next state
        x = torch.cat((states[:-1], states[1:]), dim=-1)
        if cfg.predict_delta:
            raise ValueError("Delta prediction is not supported for inverse dynamics.")
        else:
            # Predict the absolute action
            y = actions[:-1]
    
    elif cfg.training_objective == "forward_v2":
        # Calculate the number of elements that can be divided by cfg.sequence_length
        # Generate x and y based on prediction horizon. If horizon > 1, predict state_{t+H} from state_{t} and action_{t:t+H-1}
        window_size = cfg.sequence_length

        # padd x by window size with the first value of x
        states_ = torch.zeros((states.shape[0] + window_size, states.shape[1]))
        states_[:window_size] = states[0]
        states_[window_size:] = states

        states_seq = torch.zeros((states.shape[0], states.shape[1] * window_size))
        for i in range(states.shape[0]):
            states_seq[i, :] = states_[i : i + window_size, :].flatten()
        # breakpoint()

        horizon = cfg.horizon
        action_seq = torch.zeros((states.shape[0] - horizon, actions.shape[1] * horizon))
        for i in range(states.shape[0] - horizon):
                action_seq[i, :] = actions[i : i + horizon, :].flatten()

        # Concatenate the state and action tensors and remove the last state
        x = torch.cat((states_seq[:-horizon], action_seq), dim=-1)

        # Concatenate the state tensors and remove the first state
        if cfg.predict_delta:
            # Predict the change in state
            y = states[horizon:] - states[:-horizon]
        else:
            # Predict the absolute state
            y = states[horizon:]

        # y[:, 14:] = torch.where(y[:, 14:] < 0.3, torch.zeros_like(y[:, 14:]), torch.ones_like(y[:, 14:]))

        if "joint_vel" in cfg.remove_out and "skin" in cfg.remove_out:
            y = y[:, :-63]
        elif "joint_vel" in cfg.remove_out:
            y = torch.cat((y[:, :7], y[:, 14:]), dim=1)
        elif "skin" in cfg.remove_out:
            y = y[:, :-56]

    # Calculate the number of elements that can be divided by cfg.sequence_length
    num_elements = (x.shape[0] // cfg.sequence_length) * cfg.sequence_length

    # Truncate the tensors to make its total size divisible by cfg.sequence_length
    x = x[:num_elements]
    y = y[:num_elements]

    # Reshape the tensors
    x = x.view(-1, cfg.sequence_length, x.shape[-1])
    y = y.view(-1, cfg.sequence_length, y.shape[-1])
    
    print ("Shapes: ", x.shape, y.shape)

    return x, y

def construct_input_output_v4(cfg: TrainConfig, states, actions):
    """
    Construct the input and output tensors for training from the raw state
    and action tensors. This includes splitting the data into sequences of
    length `sequence_length`.

    Args:
        cfg (TrainConfig): The training configuration.
        states (torch.Tensor) [Trajectory, State]: The state tensor.
        actions (torch.Tensor) [Trajectory, Action]: The action tensor.

    Returns:
        x (torch.Tensor) [Batch, Sequence, Input]: The input tensor.
        y (torch.Tensor) [Batch, Sequence, Output]: The output tensor.

    """

    if cfg.training_objective == "forward":
        # Concatenate the state and action tensors and remove the last state
        x = torch.cat((states[:-1], actions[:-1]), dim=-1)

        # Concatenate the state tensors and remove the first state
        if cfg.predict_delta:
            # Predict the change in state
            y = states[1:] - states[:-1]
        else:
            # Predict the absolute state
            y = states[1:]

    elif cfg.training_objective == "inverse":
        # Concatenate the state with the next state
        x = torch.cat((states[:-1], states[1:]), dim=-1)
        if cfg.predict_delta:
            raise ValueError("Delta prediction is not supported for inverse dynamics.")
        else:
            # Predict the absolute action
            y = actions[:-1]
    
    elif cfg.training_objective == "forward_v2":
        # Calculate the number of elements that can be divided by cfg.sequence_length
        # Generate x and y based on prediction horizon. If horizon > 1, predict state_{t+H} from state_{t} and action_{t:t+H-1}
        window_size = cfg.sequence_length

        # padd x by window size with the first value of x
        states_ = torch.zeros((states.shape[0] + window_size, states.shape[1]))
        states_[:window_size] = states[0]
        states_[window_size:] = states

        states_seq = torch.zeros((states.shape[0], states.shape[1] * window_size))
        for i in range(states.shape[0]):
            states_seq[i, :] = states_[i : i + window_size, :].flatten()
        # breakpoint()

        horizon = cfg.horizon
        action_seq = torch.zeros((states.shape[0] - horizon, actions.shape[1] * horizon))
        for i in range(states.shape[0] - horizon):
                action_seq[i, :] = actions[i : i + horizon, :].flatten()

        # Concatenate the state and action tensors and remove the last state
        x = torch.cat((states_seq[:-horizon], action_seq), dim=-1)

        # # Concatenate the state tensors and remove the first state
        # if cfg.predict_delta:
        #     # Predict the change in state
        #     y = states[horizon:] - states[:-horizon]
        # else:
        #     # Predict the absolute state
        #     y = states[horizon:]

        # # y[:, 14:] = torch.where(y[:, 14:] < 0.3, torch.zeros_like(y[:, 14:]), torch.ones_like(y[:, 14:]))

        # if "joint_vel" in cfg.remove_out and "skin" in cfg.remove_out:
        #     y = y[:, :-63]
        # elif "joint_vel" in cfg.remove_out:
        #     y = torch.cat((y[:, :7], y[:, 14:]), dim=1)
        # elif "skin" in cfg.remove_out:
        #     y = y[:, :-56]

    # Calculate the number of elements that can be divided by cfg.sequence_length
    num_elements = (x.shape[0] // cfg.sequence_length) * cfg.sequence_length

    # Truncate the tensors to make its total size divisible by cfg.sequence_length
    x = x[:num_elements]

    # Reshape the tensors
    x = x.view(-1, cfg.sequence_length, x.shape[-1])
    
    print ("Shapes: ", x.shape, y.shape)

    return x, x

def normalize(tensor, min_vals=None, max_vals=None):
    """
    Normalize a trajectory tensor to the range [-1, 1] along the last dimension.
    If min_vals and max_vals are not provided, they are calculated from the tensor.

    Args:
        tensor (torch.Tensor) [Batch, Sequence, State]: The trajectory tensor.
        min_vals (torch.Tensor) [State]: The minimum values of the state variables, default=None.
        max_vals (torch.Tensor) [State]: The maximum values of the state variables, default=None.

    Returns:
        normalized_tensor (torch.Tensor) [Batch, Sequence, State]: The normalized trajectory tensor.
        min_vals (torch.Tensor) [State]: The minimum values of the state variables.
                                         Only returned if not provided as input.
        max_vals (torch.Tensor) [State]: The maximum values of the state variables.
                                         Only returned if not provided as input.

    """

    # Calculate the minimum and maximum values of the state variables
    if min_vals is None and max_vals is None:
        return_min_max = True
        min_vals = tensor.min(dim=0, keepdim=True)[0]
        max_vals = tensor.max(dim=0, keepdim=True)[0]

        # min_vals = tensor.min(dim=0)[0].min(dim=0)[0]
        # max_vals = tensor.max(dim=0)[0].max(dim=0)[0]
    elif min_vals is not None and max_vals is not None:
        return_min_max = False
    else:
        raise ValueError("min_vals and max_vals must be provided together.")

    # Normalize the tensor
    normalized_tensor = 2 * ((tensor - min_vals) / (max_vals - min_vals + 1e-6)) - 1
    # breakpoint()
    # Return the normalized tensor and the min and max values if needed
    if return_min_max:
        return normalized_tensor, min_vals, max_vals
    else:
        return normalized_tensor


def denormalize(tensor, min_vals, max_vals):
    """
    Denormalize a trajectory tensor from the range [-1, 1] to the original range.

    Args:
        tensor (torch.Tensor) [Batch, Sequence, State]: The normalized trajectory tensor.
        min_vals (torch.Tensor) [State]: The minimum values of the state variables.
        max_vals (torch.Tensor) [State]: The maximum values of the state variables.

    Returns:
        denormalized_tensor (torch.Tensor) [Batch, Sequence, State]: The denormalized trajectory tensor.

    """

    # Denormalize the tensor
    return 0.5 * (tensor + 1) * (max_vals - min_vals + 1e-6) + min_vals

def construct_input_output_v5(cfg: TrainConfig, states, actions):
    """
    Construct the input and output tensors for training from the raw state
    and action tensors. This includes splitting the data into sequences of
    length `sequence_length`.

    Args:
        cfg (TrainConfig): The training configuration.
        states (torch.Tensor) [Trajectory, State]: The state tensor.
        actions (torch.Tensor) [Trajectory, Action]: The action tensor.

    Returns:
        x (torch.Tensor) [Batch, Sequence, Input]: The input tensor.
        y (torch.Tensor) [Batch, Sequence, Output]: The output tensor.
        a (torch.Tensor) [Batch, Sequence, Action]: The action tensor.

    """

    horizon = cfg.horizon
    input_window_size = cfg.sequence_length

    states_expand = torch.zeros((states.shape[0] + input_window_size - 1, states.shape[1]))
    states_expand[:input_window_size - 1] = states[0]
    states_expand[input_window_size - 1:] = states

    states_seq = torch.zeros((states.shape[0] - horizon, input_window_size, states.shape[1]))
    for i in range(states.shape[0] - horizon):
        for j in range(input_window_size):
            states_seq[i, j, :] = states_expand[i + j, :].flatten()

    action_seq = torch.zeros((states.shape[0] - horizon, actions.shape[1] * horizon))
    for i in range(states.shape[0] - horizon):
        action_seq[i, :] = actions[i : i + horizon, :].flatten()
        
    y_seq = states[horizon:]

    # if "joint_vel" in cfg.remove_out and "skin" in cfg.remove_out:
    #         y_seq = y_seq[:, :-63]
    # elif "joint_vel" in cfg.remove_out:
    #     y_seq = torch.cat((y_seq[:, :7], y_seq[:, 14:]), dim=1)
    # elif "skin" in cfg.remove_out:
    #     y_seq = y_seq[:, :-56]

    states_seq = states_seq.view(-1, input_window_size, states_seq.shape[-1])
    y_seq = y_seq.view(-1, 1, y_seq.shape[-1])
    action_seq = action_seq.view(-1, 1, action_seq.shape[-1])

    return states_seq, y_seq, action_seq
