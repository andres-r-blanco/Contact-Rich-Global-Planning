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
import cv2

from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from sklearn.model_selection import KFold
import gc
import json
from tqdm import tqdm

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

class ResNet50Encoder:
    def __init__(self, device='cuda'):
        self.device = device

        weights = ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = resnet18(weights=weights)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, imgs, batch_size=100):
        imgs = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2)
        
        features_list = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                batch_imgs = imgs[i:i + batch_size]
                batch_imgs = torch.stack([self.preprocess(img) for img in batch_imgs])
                batch_imgs = batch_imgs.to(self.device)
                
                with torch.cuda.amp.autocast():
                    batch_features = self.model(batch_imgs)
                batch_features = batch_features.view(batch_features.size(0), -1)
                features_list.append(batch_features.cpu())

                del batch_imgs, batch_features
                torch.cuda.empty_cache()

        features = torch.cat(features_list, dim=0)
        return features.numpy()

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
        forces_list = []

        # weights = ResNet50_Weights.DEFAULT
        # preprocess = weights.transforms()

        random.seed(cfg.seed)

        files_all = os.listdir(os.path.join(cfg.root_dir, cfg.data_dir))

        for file in files_all:
            data = h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file), 'r')
            for key in list(data.keys()):
                hdf5_data = data[key]
                forces = np.array(hdf5_data['skin'])

                # Append the trajectories to the list
                forces_list.append(forces)

        # Concatenate the trajectories
        self.forces = np.concatenate(forces_list, axis=0)

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
        
class ToyEnvDataset_cross_val(torch.utils.data.Dataset):
    """
    Dataset loader that parses ToyEnv dataset for cross-validation. The dataset consists of npz arrays with
    parent key `arr_0` containing the state-action pairs. The observations at each timestep consist of
    the following features:
        -Joint position ("joint_pos")
        -Joint velocity ("joint_vel")
        -Joint force ("joint_force")
        -Skin force ("skin")
        -Skin IDs that are needed to sort the data correctly ("skin_ids")
    The joint velocity at current timestep is used as the action for the next timestep.
    """
    def __init__(self, cfg: TrainConfig, n_splits: int = 10):
        self.cfg = cfg
        self.n_splits = n_splits
        # self.encoder = ResNet50Encoder()

        random.seed(cfg.seed)
        files_all = os.listdir(os.path.join(cfg.root_dir, cfg.data_dir))
        random.shuffle(files_all)

        self.files_all = files_all

        self.processed_data = []
        start = time.time()
        for file in files_all:
            data = h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file), 'r')
            for key in list(data.keys()):
                hdf5_data = data[key]
                states = np.concatenate([hdf5_data[key] for key in cfg.obs_types], axis=1)
                taxel_ids = np.array(hdf5_data['skin_ids']) - 1000
                states_joint = states[:, :14]
                state_taxel = states[:, 14:]
                if cfg.filter_all_zeros:
                    if np.all(state_taxel == 0):
                        continue
                sorted_indices = np.argsort(taxel_ids, axis=1)
                sorted_states_taxel = np.take_along_axis(state_taxel, sorted_indices, axis=1)
                states = np.hstack((states_joint, sorted_states_taxel))
                actions = np.array(hdf5_data['action'])
                # imgs = np.array(hdf5_data['RGB'])

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                # img_features = self.encoder.encode(imgs, batch_size=256)
                # img_features = torch.zeros((imgs.shape[0], 512))
                # img_features = torch.tensor(img_features, dtype=torch.float16)

                # Construct the input and output tensors
                x, y, a= construct_input_output_taxel(cfg, states, actions, cfg.link_name)

                mask = ~((x[:, :, 14:] == 0).all(dim=-1) & (y == 0).all(dim=-1))
                x = x[mask].unsqueeze(1)
                y = y[mask].unsqueeze(1)
                a = a[mask].unsqueeze(1)
                self.processed_data.append((x, y, a))

                torch.cuda.empty_cache()

        end = time.time()
        print(f"Time taken to load data: {end - start}")

        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        self.splits = list(self.kf.split(self.processed_data))

    def get_test_data(self, fold_idx: int):
        train_idx, test_idx = self.splits[fold_idx]
        test_data = [self.processed_data[i] for i in test_idx]
        x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test = zip(*test_data)
        return x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test

    def set_fold(self, fold_idx: int):
        train_idx, test_idx = self.splits[fold_idx]

        train_data = [self.processed_data[i] for i in train_idx]
        test_data = [self.processed_data[i] for i in test_idx]

        x_trajectories_list, y_trajectories_list, a_trajectories_list = zip(*train_data)
        x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test = zip(*test_data)

        # Concatenate the trajectories
        self.x_trajectories = torch.cat(x_trajectories_list, dim=0)
        self.y_trajectories = torch.cat(y_trajectories_list, dim=0)
        self.a_trajectories = torch.cat(a_trajectories_list, dim=0)

        self.x_trajectories_test = torch.cat(x_trajectories_list_test, dim=0)
        self.y_trajectories_test = torch.cat(y_trajectories_list_test, dim=0)
        self.a_trajectories_test = torch.cat(a_trajectories_list_test, dim=0)

        # Normalize the data to the range [-1, 1]
        if self.cfg.normalization:
            self.x_trajectories, self.x_min, self.x_max = normalize(
                self.x_trajectories
            )
            self.a_trajectories, self.a_min, self.a_max = normalize(
                self.a_trajectories
            )
            self.x_trajectories_test= normalize(
                self.x_trajectories_test, self.x_min, self.x_max
            )
            self.a_trajectories_test = normalize(
                self.a_trajectories_test, self.a_min, self.a_max
            )

        # Get the number of trajectories
        self.num_trajectories = self.x_trajectories.shape[0]
        print(f"Number of training trajectories: {self.num_trajectories}")
        self.num_trajectories_test = self.x_trajectories_test.shape[0]
        print(f"Number of testing trajectories: {self.num_trajectories_test}")

        # Set the input and output sizes for the network
        self.cfg.network_config.input_size = self.x_trajectories.shape[-1]
        self.cfg.network_config.output_size = self.y_trajectories.shape[-1]
        self.cfg.network_config.action_size = self.a_trajectories.shape[-1]
        self.cfg.network_config.sequence_length = self.x_trajectories.shape[1]

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
        
class ToyEnvDataset_cross_val_case_3(torch.utils.data.Dataset):
    """
    Dataset loader that parses ToyEnv dataset for cross-validation. The dataset consists of npz arrays with
    parent key `arr_0` containing the state-action pairs. The observations at each timestep consist of
    the following features:
        -Joint position ("joint_pos")
        -Joint velocity ("joint_vel")
        -Joint force ("joint_force")
        -Skin force ("skin")
        -Skin IDs that are needed to sort the data correctly ("skin_ids")
    The joint velocity at current timestep is used as the action for the next timestep.
    """
    def __init__(self, cfg: TrainConfig, n_splits: int = 10):
        self.cfg = cfg
        self.n_splits = n_splits
        # self.encoder = ResNet50Encoder()

        random.seed(cfg.seed)
        files_all = os.listdir(os.path.join(cfg.root_dir, cfg.data_dir))
        random.shuffle(files_all)

        self.files_all = files_all

        self.processed_data = []
        start = time.time()
        for file in files_all:
            data = h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file), 'r')
            for key in list(data.keys()):
                hdf5_data = data[key]
                states = np.concatenate([hdf5_data[key] for key in cfg.obs_types], axis=1)
                taxel_ids = np.array(hdf5_data['skin_ids']) - 1000
                states_joint = states[:, :14]
                state_taxel = states[:, 14:]
                if cfg.filter_all_zeros:
                    if np.all(state_taxel == 0):
                        continue
                sorted_indices = np.argsort(taxel_ids, axis=1)
                sorted_states_taxel = np.take_along_axis(state_taxel, sorted_indices, axis=1)
                states = np.hstack((states_joint, sorted_states_taxel))
                actions = np.array(hdf5_data['action'])
                # imgs = np.array(hdf5_data['RGB'])

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                # img_features = self.encoder.encode(imgs, batch_size=256)
                # img_features = torch.zeros((imgs.shape[0], 512))
                # img_features = torch.tensor(img_features, dtype=torch.float16)

                # Construct the input and output tensors
                x, y, a= construct_input_output_taxel_case_3(cfg, states, actions, cfg.link_name)

                mask = ~((x[:, :, 14:] == 0).all(dim=-1) & (y == 0).all(dim=-1))
                x = x[mask].unsqueeze(1)
                y = y[mask].unsqueeze(1)
                a = a[mask].unsqueeze(1)

                self.processed_data.append((x, y, a))

                torch.cuda.empty_cache()

        end = time.time()
        print(f"Time taken to load data: {end - start}")

        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        self.splits = list(self.kf.split(self.processed_data))

    def get_test_data(self, fold_idx: int):
        train_idx, test_idx = self.splits[fold_idx]
        test_data = [self.processed_data[i] for i in test_idx]
        x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test = zip(*test_data)
        return x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test

    def set_fold(self, fold_idx: int):
        train_idx, test_idx = self.splits[fold_idx]

        train_data = [self.processed_data[i] for i in train_idx]
        test_data = [self.processed_data[i] for i in test_idx]

        x_trajectories_list, y_trajectories_list, a_trajectories_list = zip(*train_data)
        x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test = zip(*test_data)

        # Concatenate the trajectories
        self.x_trajectories = torch.cat(x_trajectories_list, dim=0)
        self.y_trajectories = torch.cat(y_trajectories_list, dim=0)
        self.a_trajectories = torch.cat(a_trajectories_list, dim=0)

        self.x_trajectories_test = torch.cat(x_trajectories_list_test, dim=0)
        self.y_trajectories_test = torch.cat(y_trajectories_list_test, dim=0)
        self.a_trajectories_test = torch.cat(a_trajectories_list_test, dim=0)

        # Normalize the data to the range [-1, 1]
        if self.cfg.normalization:
            self.x_trajectories, self.x_min, self.x_max = normalize(
                self.x_trajectories
            )
            self.a_trajectories, self.a_min, self.a_max = normalize(
                self.a_trajectories
            )
            self.x_trajectories_test= normalize(
                self.x_trajectories_test, self.x_min, self.x_max
            )
            self.a_trajectories_test = normalize(
                self.a_trajectories_test, self.a_min, self.a_max
            )

        # Get the number of trajectories
        self.num_trajectories = self.x_trajectories.shape[0]
        print(f"Number of training trajectories: {self.num_trajectories}")
        self.num_trajectories_test = self.x_trajectories_test.shape[0]
        print(f"Number of testing trajectories: {self.num_trajectories_test}")

        # Set the input and output sizes for the network
        self.cfg.network_config.input_size = self.x_trajectories.shape[-1]
        self.cfg.network_config.output_size = self.y_trajectories.shape[-1]
        self.cfg.network_config.action_size = self.a_trajectories.shape[-1]
        self.cfg.network_config.sequence_length = self.x_trajectories.shape[1]

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
        
class ToyEnvDataset_cross_val_case_2(torch.utils.data.Dataset):
    """
    Dataset loader that parses ToyEnv dataset for cross-validation. The dataset consists of npz arrays with
    parent key `arr_0` containing the state-action pairs. The observations at each timestep consist of
    the following features:
        -Joint position ("joint_pos")
        -Joint velocity ("joint_vel")
        -Joint force ("joint_force")
        -Skin force ("skin")
        -Skin IDs that are needed to sort the data correctly ("skin_ids")
    The joint velocity at current timestep is used as the action for the next timestep.
    """
    def __init__(self, cfg: TrainConfig, n_splits: int = 10):
        self.cfg = cfg
        self.n_splits = n_splits
        # self.encoder = ResNet50Encoder()

        random.seed(cfg.seed)
        files_all = os.listdir(os.path.join(cfg.root_dir, cfg.data_dir))
        random.shuffle(files_all)

        self.files_all = files_all

        self.processed_data = []
        start = time.time()
        for file in files_all:
            data = h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file), 'r')
            for key in list(data.keys()):
                hdf5_data = data[key]
                states = np.concatenate([hdf5_data[key] for key in cfg.obs_types], axis=1)
                taxel_ids = np.array(hdf5_data['skin_ids']) - 1000
                states_joint = states[:, :14]
                state_taxel = states[:, 14:]
                if cfg.filter_all_zeros:
                    if np.all(state_taxel == 0):
                        continue
                sorted_indices = np.argsort(taxel_ids, axis=1)
                sorted_states_taxel = np.take_along_axis(state_taxel, sorted_indices, axis=1)
                states = np.hstack((states_joint, sorted_states_taxel))
                actions = np.array(hdf5_data['action'])
                # imgs = np.array(hdf5_data['RGB'])

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                # img_features = self.encoder.encode(imgs, batch_size=256)
                # img_features = torch.zeros((imgs.shape[0], 512))
                # img_features = torch.tensor(img_features, dtype=torch.float16)

                # Construct the input and output tensors
                x, y, a= construct_input_output_taxel_case_3(cfg, states, actions, cfg.link_name)

                mask = ~((x[:, :, 14:] == 0).all(dim=-1) & (y == 0).all(dim=-1))
                x = x[mask].unsqueeze(1)
                y = y[mask].unsqueeze(1)
                a = a[mask].unsqueeze(1)

                self.processed_data.append((x, y, a))

                torch.cuda.empty_cache()

        end = time.time()
        print(f"Time taken to load data: {end - start}")

        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        self.splits = list(self.kf.split(self.processed_data))

    def get_test_data(self, fold_idx: int):
        train_idx, test_idx = self.splits[fold_idx]
        test_data = [self.processed_data[i] for i in test_idx]
        x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test = zip(*test_data)
        return x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test

    def set_fold(self, fold_idx: int):
        train_idx, test_idx = self.splits[fold_idx]

        train_data = [self.processed_data[i] for i in train_idx]
        test_data = [self.processed_data[i] for i in test_idx]

        x_trajectories_list, y_trajectories_list, a_trajectories_list = zip(*train_data)
        x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test = zip(*test_data)

        # Concatenate the trajectories
        self.x_trajectories = torch.cat(x_trajectories_list, dim=0)
        self.y_trajectories = torch.cat(y_trajectories_list, dim=0)
        self.a_trajectories = torch.cat(a_trajectories_list, dim=0)

        self.x_trajectories_test = torch.cat(x_trajectories_list_test, dim=0)
        self.y_trajectories_test = torch.cat(y_trajectories_list_test, dim=0)
        self.a_trajectories_test = torch.cat(a_trajectories_list_test, dim=0)

        # Normalize the data to the range [-1, 1]
        if self.cfg.normalization:
            self.x_trajectories, self.x_min, self.x_max = normalize(
                self.x_trajectories
            )
            self.a_trajectories, self.a_min, self.a_max = normalize(
                self.a_trajectories
            )
            self.x_trajectories_test= normalize(
                self.x_trajectories_test, self.x_min, self.x_max
            )
            self.a_trajectories_test = normalize(
                self.a_trajectories_test, self.a_min, self.a_max
            )

        # Get the number of trajectories
        self.num_trajectories = self.x_trajectories.shape[0]
        print(f"Number of training trajectories: {self.num_trajectories}")
        self.num_trajectories_test = self.x_trajectories_test.shape[0]
        print(f"Number of testing trajectories: {self.num_trajectories_test}")

        # Set the input and output sizes for the network
        self.cfg.network_config.input_size = self.x_trajectories.shape[-1]
        self.cfg.network_config.output_size = self.y_trajectories.shape[-1]
        self.cfg.network_config.action_size = self.a_trajectories.shape[-1]
        self.cfg.network_config.sequence_length = self.x_trajectories.shape[1]

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
        
class ToyEnvDataset_cross_val_wm(torch.utils.data.Dataset):
    """
    Dataset loader that parses ToyEnv dataset for cross-validation. The dataset consists of npz arrays with
    parent key `arr_0` containing the state-action pairs. The observations at each timestep consist of
    the following features:
        -Joint position ("joint_pos")
        -Joint velocity ("joint_vel")
        -Joint force ("joint_force")
        -Skin force ("skin")
        -Skin IDs that are needed to sort the data correctly ("skin_ids")
    The joint velocity at current timestep is used as the action for the next timestep.
    """
    def __init__(self, cfg: TrainConfig, n_splits: int = 10):
        self.cfg = cfg
        self.n_splits = n_splits
        # self.encoder = ResNet50Encoder()

        random.seed(cfg.seed)
        files_all = os.listdir(os.path.join(cfg.root_dir, cfg.data_dir))
        random.shuffle(files_all)

        self.files_all = files_all

        self.processed_data = []
        start = time.time()
        for file in files_all:
            data = h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file), 'r')
            for key in list(data.keys()):
                hdf5_data = data[key]
                states = np.concatenate([hdf5_data[key] for key in cfg.obs_types], axis=1)
                taxel_ids = np.array(hdf5_data['skin_ids']) - 1000
                states_joint = states[:, :14]
                state_taxel = states[:, 14:]
                if cfg.filter_all_zeros:
                    if np.all(state_taxel == 0):
                        continue
                sorted_indices = np.argsort(taxel_ids, axis=1)
                sorted_states_taxel = np.take_along_axis(state_taxel, sorted_indices, axis=1)
                states = np.hstack((states_joint, sorted_states_taxel))
                actions = np.array(hdf5_data['action'])
                # imgs = np.array(hdf5_data['RGB'])

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                # img_features = self.encoder.encode(imgs, batch_size=256)
                # img_features = torch.zeros((imgs.shape[0], 512))
                # img_features = torch.tensor(img_features, dtype=torch.float16)

                # Construct the input and output tensors
                x, y, a= construct_input_output_taxel_wm(cfg, states, actions)

                # mask = ~((x[:, :, 14:] == 0).all(dim=-1) & (y == 0).all(dim=-1))
                # x = x[mask].unsqueeze(1)
                # y = y[mask].unsqueeze(1)
                # a = a[mask].unsqueeze(1)

                self.processed_data.append((x, y, a))

                torch.cuda.empty_cache()

        end = time.time()
        print(f"Time taken to load data: {end - start}")

        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        self.splits = list(self.kf.split(self.processed_data))

    def get_test_data(self, fold_idx: int):
        train_idx, test_idx = self.splits[fold_idx]
        test_data = [self.processed_data[i] for i in test_idx]
        x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test = zip(*test_data)
        return x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test

    def set_fold(self, fold_idx: int):
        train_idx, test_idx = self.splits[fold_idx]

        train_data = [self.processed_data[i] for i in train_idx]
        test_data = [self.processed_data[i] for i in test_idx]

        x_trajectories_list, y_trajectories_list, a_trajectories_list = zip(*train_data)
        x_trajectories_list_test, y_trajectories_list_test, a_trajectories_list_test = zip(*test_data)

        # Concatenate the trajectories
        self.x_trajectories = torch.cat(x_trajectories_list, dim=0)
        self.y_trajectories = torch.cat(y_trajectories_list, dim=0)
        self.a_trajectories = torch.cat(a_trajectories_list, dim=0)

        self.x_trajectories_test = torch.cat(x_trajectories_list_test, dim=0)
        self.y_trajectories_test = torch.cat(y_trajectories_list_test, dim=0)
        self.a_trajectories_test = torch.cat(a_trajectories_list_test, dim=0)

        # Normalize the data to the range [-1, 1]
        if self.cfg.normalization:
            self.x_trajectories[:, :, :7], self.pos_min, self.pos_max = normalize(
                self.x_trajectories[:, :, :7]
            )
            self.x_trajectories[:, :, 7:14], self.vel_min, self.vel_max = normalize(
                self.x_trajectories[:, :, 7:14]
            )
            # self.x_trajectories[:, :, 14:], self.force_min, self.force_max = normalize(
            #     self.x_trajectories[:, :, 14:]
            # )
            self.a_trajectories, self.a_min, self.a_max = normalize(
                self.a_trajectories
            )
            self.x_trajectories_test[:, :, :7] = normalize(
                self.x_trajectories_test[:, :, :7], self.pos_min, self.pos_max
            )
            self.x_trajectories_test[:, :, 7:14] = normalize(
                self.x_trajectories_test[:, :, 7:14], self.vel_min, self.vel_max
            )
            # self.x_trajectories_test[:, :, 14:] = normalize(
            #     self.x_trajectories_test[:, :, 14:], self.force_min, self.force_max
            # )
            self.a_trajectories_test = normalize(
                self.a_trajectories_test, self.a_min, self.a_max
            )

        # Get the number of trajectories
        self.num_trajectories = self.x_trajectories.shape[0]
        print(f"Number of training trajectories: {self.num_trajectories}")
        self.num_trajectories_test = self.x_trajectories_test.shape[0]
        print(f"Number of testing trajectories: {self.num_trajectories_test}")

        # Set the input and output sizes for the network
        self.cfg.network_config.input_size = self.x_trajectories.shape[-1]
        self.cfg.network_config.output_size = self.y_trajectories.shape[-1]
        self.cfg.network_config.action_size = self.a_trajectories.shape[-1]
        self.cfg.network_config.sequence_length = self.x_trajectories.shape[1]

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
        
def preprocess_h5_to_binary(cfg, mode='train'):
    random.seed(cfg.seed)
    files_all = os.listdir(os.path.join(cfg.root_dir, cfg.data_dir))
    random.shuffle(files_all)
    split_index = int(len(files_all) * cfg.train_split)
    train_files = files_all[:split_index]
    test_files = files_all[split_index:]
    files = train_files if mode == 'train' else test_files

    data_save_path = os.path.join(cfg.data_save_dir, f"{mode}_data.h5")
    index_save_path = os.path.join(cfg.data_save_dir, f"{mode}_index.json")

    if os.path.exists(data_save_path) and os.path.exists(index_save_path):
        print(f"Data and index files already exist at {data_save_path} and {index_save_path}. Skipping preprocessing.")
        return

    index = []  # 全局索引
    total_samples = 0

    # Calculate the total number of samples for the HDF5 file
    for file_name in files:
        with h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file_name), 'r') as data:
            for traj_idx in list(data.keys()):
                hdf5_data = data[traj_idx]
                state_taxel = np.array(hdf5_data['skin'])
                if cfg.filter_all_zeros and np.all(state_taxel == 0):
                    continue
                n_samples = state_taxel.shape[0]
                total_samples += max(0, n_samples - cfg.horizon)

    # Create HDF5 file to store data
    with h5py.File(data_save_path, 'w') as hf:
        # Create datasets to store all the required data
        states_dataset = hf.create_dataset('state', shape=(total_samples, 70), dtype='f4')
        y_dataset = hf.create_dataset('y', shape=(total_samples, cfg.horizon, 70), dtype='f4')
        actions_dataset = hf.create_dataset('action', shape=(total_samples, 70), dtype='f4')
        imgs_dataset = hf.create_dataset('img', shape=(total_samples, cfg.horizon + 1, 128, 128, 3), dtype='f4')
        depth_dataset = hf.create_dataset('depth', shape=(total_samples, cfg.horizon + 1, 128, 128), dtype='f4')
        # labels_dataset = hf.create_dataset('label', shape=(total_samples,), dtype='i4')

        sample_idx = 0
        files = files[:5]
        for file_idx, file_name in enumerate(tqdm(files, desc="Processing files")):
            with h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file_name), 'r') as data:
                for traj_idx in tqdm(list(data.keys()), desc=f"Processing trajectories in {file_name}", leave=False):
                    hdf5_data = data[traj_idx]
                    state_taxel = np.array(hdf5_data['skin'])
                    if cfg.filter_all_zeros and np.all(state_taxel == 0):
                        continue
                    n_samples = state_taxel.shape[0]
                    states = np.concatenate([hdf5_data[key] for key in cfg.obs_types], axis=1)
                    taxel_ids = np.array(hdf5_data['skin_ids']) - 1000
                    states_joint = states[:, :14]
                    state_taxel = states[:, 14:]
                    sorted_indices = np.argsort(taxel_ids, axis=1)
                    sorted_states_taxel = np.take_along_axis(state_taxel, sorted_indices, axis=1)
                    states = np.hstack((states_joint, sorted_states_taxel))
                    actions = np.array(hdf5_data['joint_vel'])
                    imgs = np.array(hdf5_data['RGB'])
                    depth = np.array(hdf5_data['Depth'])

                    # Calculate labels for all samples at once
                    sample = state_taxel[:-cfg.horizon]
                    sample_future = state_taxel[cfg.horizon:]
                    labels = np.where(np.all(sample == 0, axis=1) & np.all(sample_future == 0, axis=1), 0, 1)

                    # Store the samples in the HDF5 file
                    states_dataset[sample_idx:sample_idx + n_samples - cfg.horizon] = states[:-cfg.horizon]
                    y_dataset[sample_idx:sample_idx + n_samples - cfg.horizon] = np.array([states[i+1:i+1 + cfg.horizon] for i in range(n_samples - cfg.horizon)])
                    actions_dataset[sample_idx:sample_idx + n_samples - cfg.horizon] = np.array([actions[i:i + cfg.horizon].flatten() for i in range(n_samples - cfg.horizon)])
                    imgs_dataset[sample_idx:sample_idx + n_samples - cfg.horizon] = np.array([imgs[i:i+1 + cfg.horizon] for i in range(n_samples - cfg.horizon)])
                    depth_dataset[sample_idx:sample_idx + n_samples - cfg.horizon] = np.array([depth[i:i+1 + cfg.horizon] for i in range(n_samples - cfg.horizon)])

                    # Store the samples in the HDF5 file
                    # states_dataset[sample_idx:sample_idx + n_samples - cfg.horizon] = states[:-cfg.horizon]
                    # y_dataset[sample_idx:sample_idx + n_samples - cfg.horizon] = states[cfg.horizon:]
                    # actions_dataset[sample_idx:sample_idx + n_samples - cfg.horizon] = np.array([actions[i:i + cfg.horizon].flatten() for i in range(n_samples - cfg.horizon)])
                    # imgs_dataset[sample_idx:sample_idx + n_samples - cfg.horizon] = imgs[:-cfg.horizon]
                    # depth_dataset[sample_idx:sample_idx + n_samples - cfg.horizon] = depth[:-cfg.horizon]
                    # labels_dataset[sample_idx:sample_idx + n_samples - cfg.horizon] = labels

                    # Add index
                    for i in range(n_samples - cfg.horizon):
                        index.append({'offset': int(sample_idx + i), 'label': int(labels[i])})
                    sample_idx += n_samples - cfg.horizon

                    # for idx in range(n_samples - cfg.horizon):
                    #     sample = state_taxel[idx]
                    #     sample_future = state_taxel[idx + cfg.horizon]
                    #     label = 1 if not (np.all(sample == 0) and np.all(sample_future == 0)) else 0

                    #     # Store the sample in the HDF5 file
                    #     states_dataset[sample_idx] = states[idx]
                    #     y_dataset[sample_idx] = states[idx + cfg.horizon]
                    #     actions_dataset[sample_idx] = actions[idx:idx + cfg.horizon].flatten()
                    #     imgs_dataset[sample_idx] = imgs[idx]
                    #     labels_dataset[sample_idx] = label

                    #     # Add index
                    #     index.append({'offset': sample_idx, 'label': label})
                    #     sample_idx += 1

        # Save the global index
        with open(index_save_path, 'w') as f_index:
            json.dump(index, f_index)


def reorganize_data_by_label(cfg, mode='train'):
    random.seed(cfg.seed)
    
    # Paths to the original data and index files
    data_file_path = os.path.join(cfg.data_save_dir, f"{mode}_data.h5")
    index_file_path = os.path.join(cfg.data_save_dir, f"{mode}_index.json")
    
    # Check if the original data and index files exist
    if not os.path.exists(data_file_path) or not os.path.exists(index_file_path):
        print(f"Original data or index file not found: {data_file_path} or {index_file_path}")
        return

    # Load the index file containing offsets and labels
    with open(index_file_path, 'r') as f_index:
        index = json.load(f_index)
    
    # Separate the indices for label 0 and label 1
    label_0_indices = [entry for entry in index if entry['label'] == 0]
    label_1_indices = [entry for entry in index if entry['label'] == 1]
    
    # Shuffle both label groups
    random.shuffle(label_0_indices)
    random.shuffle(label_1_indices)
    
    # Balance the dataset by selecting the minimum number of samples from each label group
    min_samples = min(len(label_0_indices), len(label_1_indices)) // 5
    label_0_indices = label_0_indices[:min_samples]
    label_1_indices = label_1_indices[:min_samples]
    
    # Interleave label 0 and label 1 samples, starting with label 0
    interleaved_indices = []
    for i in range(min_samples):
        interleaved_indices.append(label_0_indices[i])
        interleaved_indices.append(label_1_indices[i])
    
    # Create new HDF5 files to store the separated and balanced data
    new_data_save_path = os.path.join(cfg.data_save_dir, f"{mode}_reorganized_data.h5")
    index_save_path = os.path.join(cfg.data_save_dir, f"{mode}_reorganized_index.json")

    if os.path.exists(new_data_save_path) or os.path.exists(index_save_path):
        print(f"Reorganized data or index file already exists: {new_data_save_path} or {index_save_path}")
        return

    # Open the original data file for reading
    with h5py.File(data_file_path, 'r') as hf:
        # Create a new HDF5 file for writing the reorganized data
        with h5py.File(new_data_save_path, 'w') as new_hf:
            # Create new datasets in the HDF5 file to store the separated label data
            states_dataset_new = new_hf.create_dataset('state', shape=(min_samples * 2, 70), dtype='f4')
            y_dataset_new = new_hf.create_dataset('y', shape=(min_samples * 2, cfg.horizon, 70), dtype='f4')
            actions_dataset_new = new_hf.create_dataset('action', shape=(min_samples * 2, 70), dtype='f4')
            imgs_dataset_new = new_hf.create_dataset('img', shape=(min_samples * 2, cfg.horizon + 1, 128, 128, 3), dtype='f4')
            depth_dataset_new = new_hf.create_dataset('depth', shape=(min_samples * 2, cfg.horizon + 1, 128, 128), dtype='f4')

            # Iterate over the interleaved indices and process each sample
            for sample_idx, entry in enumerate(interleaved_indices):
                print(f"Processing sample {sample_idx} / {len(interleaved_indices)}")
                offset = entry['offset']
                label = entry['label']
                
                # Read the sample data from the original HDF5 file using the offset
                state = hf['state'][offset]
                y = hf['y'][offset]
                actions = hf['action'][offset]
                imgs = hf['img'][offset]
                depth = hf['depth'][offset]

                # Write the sample data into the new HDF5 file
                states_dataset_new[sample_idx] = state
                y_dataset_new[sample_idx] = y
                actions_dataset_new[sample_idx] = actions
                imgs_dataset_new[sample_idx] = imgs
                depth_dataset_new[sample_idx] = depth

            # Create a new index for the reorganized data
            new_index = [{'offset': sample_idx, 'label': 1 if entry['label'] == 1 else 0} for sample_idx, entry in enumerate(interleaved_indices)]
        
        # Save the new index to a JSON file
        with open(index_save_path, 'w') as f_index:
            json.dump(new_index, f_index)

    print(f"Data reorganization complete. Saved to {new_data_save_path} and {index_save_path}.")
        
class OptimizedToyEnvDataset_each(torch.utils.data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode

        # HDF5 file path
        binary_file_path = os.path.join(cfg.data_save_dir, f"{mode}_data.h5")
        index_file_path = os.path.join(cfg.data_save_dir, f"{mode}_index.json")

        self.hf = h5py.File(binary_file_path, 'r')  # Open the HDF5 file
        with open(index_file_path, 'r') as f:
            self.index = json.load(f)

        # Extract label distribution
        self.labels = [item['label'] for item in self.index]

        self.cfg.network_config.input_size = 70
        self.cfg.network_config.output_size = 70
        self.cfg.network_config.action_size = 70
        self.cfg.network_config.sequence_length = 1

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sample_data = {
            'state': self.hf['state'][idx],
            'y': self.hf['y'][idx],
            'action': self.hf['action'][idx],
            'img': self.hf['img'][idx],
            'depth': self.hf['depth'][idx],
        }
        # state = torch.tensor(self.hf['state'][idx], dtype=torch.float32)
        # y = torch.tensor(self.hf['y'][idx], dtype=torch.float32)
        # action = torch.tensor(self.hf['action'][idx], dtype=torch.float32)
        # img = torch.tensor(self.hf['img'][idx], dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)

        state = torch.tensor(sample_data['state'], dtype=torch.float32)
        y = torch.tensor(sample_data['y'], dtype=torch.float32)
        action = torch.tensor(sample_data['action'], dtype=torch.float32)
        img = torch.tensor(sample_data['img'], dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        depth = torch.tensor(sample_data['depth'], dtype=torch.float32).unsqueeze(-1).permute(2, 0, 1) * 255 # Convert to (C, H, W)
        return state, y, action, img, depth
    

class ToyEnvDataset_each(torch.utils.data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode

        # HDF5 文件路径
        binary_file_path = os.path.join(cfg.data_save_dir, f"{mode}_reorganized_data.h5")
        index_file_path = os.path.join(cfg.data_save_dir, f"{mode}_reorganized_index.json")

        # 打开 HDF5 文件
        self.hf = h5py.File(binary_file_path, 'r')  
        
        # 读取索引文件
        with open(index_file_path, 'r') as f:
            self.index = json.load(f)


        # 配置网络参数
        self.cfg.network_config.input_size = 14
        self.cfg.network_config.output_size = 70
        self.cfg.network_config.action_size = 70
        self.cfg.network_config.sequence_length = 1

        # 初始化时加载批次数据
        self.batch_size = cfg.batch_size
        self.num_batches = len(self.index) // self.batch_size  # 计算可以生成的完整批次数量

    def __len__(self):
        # 返回完整批次数量，丢弃最后一个不完整的批次
        return self.num_batches

    def load_batch(self, batch_idx):
        """一次性读取一个连续的批次数据"""
        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size

        # 确保读取范围在数据集内
        if end_idx > len(self.index):
            end_idx = len(self.index)

        batch_data = {
            'state': self.hf['state'][start_idx:end_idx],
            'y': self.hf['y'][start_idx:end_idx],
            'action': self.hf['action'][start_idx:end_idx],
            'img': self.hf['img'][start_idx:end_idx],
            'depth': self.hf['depth'][start_idx:end_idx],
        }

        return batch_data

    def __getitem__(self, idx):
        """根据给定的索引读取并返回一个批次的数据"""
        batch_data = self.load_batch(idx)

        # 这里直接从已加载的批次数据中获取样本
        state = torch.tensor(batch_data['state'], dtype=torch.float32)
        y = torch.tensor(batch_data['y'], dtype=torch.float32)
        action = torch.tensor(batch_data['action'], dtype=torch.float32)
        img = torch.tensor(batch_data['img'], dtype=torch.float32).permute(0, 1, 4, 2, 3)  # 转换为 (B, C, H, W)
        depth = torch.tensor(batch_data['depth'], dtype=torch.float32).unsqueeze(-1).permute(0, 1, 4, 2, 3)  # 转换为 (B, C, H, W)
        
        # depth = torch.tensor(batch_data['depth'], dtype=torch.float32)

        # 获取深度图像的最小值和最大值
        min_depth = torch.min(depth)
        max_depth = torch.max(depth)

        # print(f"Min depth: {min_depth}, Max depth: {max_depth}")

        # 归一化到 [0, 1] 范围
        depth_normalized = (depth - min_depth) / (max_depth - min_depth)

        # 处理无效值 (例如 NaN 或 Inf)
        depth_normalized = torch.where(torch.isnan(depth_normalized), torch.zeros_like(depth_normalized), depth_normalized)


        return state, y, action, img, depth_normalized
    
class OptimizedToyEnvDataset_baseline(torch.utils.data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode

        # HDF5 文件路径
        binary_file_path = os.path.join(cfg.data_save_dir, f"{mode}_reorganized_data.h5")
        index_file_path = os.path.join(cfg.data_save_dir, f"{mode}_reorganized_index.json")

        # 打开 HDF5 文件
        self.hf = h5py.File(binary_file_path, 'r')  
        
        # 读取索引文件
        with open(index_file_path, 'r') as f:
            self.index = json.load(f)


        # 配置网络参数
        self.cfg.network_config.input_size = 14
        self.cfg.network_config.output_size = 70
        self.cfg.network_config.action_size = 70
        self.cfg.network_config.sequence_length = 1

        # 初始化时加载批次数据
        self.batch_size = cfg.batch_size
        self.num_batches = len(self.index) // self.batch_size  # 计算可以生成的完整批次数量

    def __len__(self):
        # 返回完整批次数量，丢弃最后一个不完整的批次
        return self.num_batches

    def load_batch(self, batch_idx):
        """一次性读取一个连续的批次数据"""
        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size

        # 确保读取范围在数据集内
        if end_idx > len(self.index):
            end_idx = len(self.index)

        batch_data = {
            'state': self.hf['state'][start_idx:end_idx],
            'y': self.hf['y'][start_idx:end_idx],
            'action': self.hf['action'][start_idx:end_idx],
        }

        return batch_data

    def __getitem__(self, idx):
        """根据给定的索引读取并返回一个批次的数据"""
        batch_data = self.load_batch(idx)

        # 这里直接从已加载的批次数据中获取样本
        state = torch.tensor(batch_data['state'], dtype=torch.float32)
        y = torch.tensor(batch_data['y'], dtype=torch.float32)
        action = torch.tensor(batch_data['action'], dtype=torch.float32)

        return state, y, action
        
class ToyEnvDataset_wm_each(torch.utils.data.Dataset):
    def __init__(self, cfg: TrainConfig, mode: str = 'train'):
        self.cfg = cfg
        self.mode = mode

        random.seed(cfg.seed)
        files_all = os.listdir(os.path.join(cfg.root_dir, cfg.data_dir))
        random.shuffle(files_all)
        split_index = int(len(files_all) * cfg.train_split)
        train_files = files_all[:split_index]
        test_files = files_all[split_index:]

        self.files = train_files if self.mode == 'train' else test_files
        self.file_handles = {file_name: h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file_name), 'r') 
                             for file_name in self.files}

        # Build global index and label distribution
        self.data_index = []
        self.labels = []
        for file_idx, file_name in enumerate(self.files):
            data = self.file_handles[file_name]
            for traj_idx in list(data.keys()):
                hdf5_data = data[traj_idx]
                state_taxel = np.array(hdf5_data['skin'])
                if self.cfg.filter_all_zeros:
                    if np.all(state_taxel == 0):
                        continue
                n_samples = state_taxel.shape[0]
                for sample_idx in range(n_samples - cfg.horizon):
                    sample = state_taxel[sample_idx]
                    sample_future = state_taxel[sample_idx + cfg.horizon]
                    label = 1 if not (np.all(sample == 0) and np.all(sample_future == 0)) else 0
                    self.data_index.append((file_idx, traj_idx, sample_idx))
                    self.labels.append(label)

        label_counts = np.bincount(self.labels)
        self.weights = [1 / label_counts[label] for label in self.labels]

        self.cfg.network_config.input_size = 70
        self.cfg.network_config.output_size = 70
        self.cfg.network_config.action_size = 70
        self.cfg.network_config.sequence_length = 1

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_idx, traj_idx, sample_idx = self.data_index[idx]
        file_name = self.files[file_idx]
        data = self.file_handles[file_name][traj_idx]

        states = np.concatenate([data[key] for key in self.cfg.obs_types], axis=1)
        taxel_ids = np.array(data['skin_ids']) - 1000
        states_joint = states[:, :14]
        state_taxel = states[:, 14:]
        sorted_indices = np.argsort(taxel_ids, axis=1)
        sorted_states_taxel = np.take_along_axis(state_taxel, sorted_indices, axis=1)
        states = np.hstack((states_joint, sorted_states_taxel))
        actions = np.array(data['action'])
        img = np.array(data['RGB'])[sample_idx]

        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        img = torch.from_numpy(img).float().permute(2, 0, 1)

        horizon = self.cfg.horizon
        state = states[sample_idx]
        y = states[sample_idx + horizon]
        action = actions[sample_idx:sample_idx + horizon].flatten()

        return state, y, action, img

    def __del__(self):
        # Ensure all files are closed when the object is deleted
        for handle in self.file_handles.values():
            handle.close()


class ToyEnvDataset_wm_baseline(torch.utils.data.Dataset):
    def __init__(self, cfg: TrainConfig, mode: str = 'train'):
        self.cfg = cfg
        self.mode = mode

        random.seed(cfg.seed)
        files_all = os.listdir(os.path.join(cfg.root_dir, cfg.data_dir))
        random.shuffle(files_all)
        split_index = int(len(files_all) * cfg.train_split)
        train_files = files_all[:split_index]
        test_files = files_all[split_index:]

        self.files = train_files if self.mode == 'train' else test_files
        self.file_handles = {file_name: h5py.File(os.path.join(cfg.root_dir, cfg.data_dir, file_name), 'r') 
                             for file_name in self.files}

        # Build global index and label distribution
        self.data_index = []
        self.labels = []
        for file_idx, file_name in enumerate(self.files):
            data = self.file_handles[file_name]
            for traj_idx in list(data.keys()):
                hdf5_data = data[traj_idx]
                state_taxel = np.array(hdf5_data['skin'])
                if self.cfg.filter_all_zeros:
                    if np.all(state_taxel == 0):
                        continue
                n_samples = state_taxel.shape[0]
                for sample_idx in range(n_samples - cfg.horizon):
                    sample = state_taxel[sample_idx]
                    sample_future = state_taxel[sample_idx + cfg.horizon]
                    label = 1 if not (np.all(sample == 0) and np.all(sample_future == 0)) else 0
                    self.data_index.append((file_idx, traj_idx, sample_idx))
                    self.labels.append(label)

        label_counts = np.bincount(self.labels)
        self.weights = [1 / label_counts[label] for label in self.labels]

        self.cfg.network_config.input_size = 70
        self.cfg.network_config.output_size = 70
        self.cfg.network_config.action_size = 70
        self.cfg.network_config.sequence_length = 1

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_idx, traj_idx, sample_idx = self.data_index[idx]
        file_name = self.files[file_idx]
        data = self.file_handles[file_name][traj_idx]

        states = np.concatenate([data[key] for key in self.cfg.obs_types], axis=1)
        taxel_ids = np.array(data['skin_ids']) - 1000
        states_joint = states[:, :14]
        state_taxel = states[:, 14:]
        sorted_indices = np.argsort(taxel_ids, axis=1)
        sorted_states_taxel = np.take_along_axis(state_taxel, sorted_indices, axis=1)
        states = np.hstack((states_joint, sorted_states_taxel))
        actions = np.array(data['action'])

        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()

        horizon = self.cfg.horizon
        state = states[sample_idx]
        y = states[sample_idx + horizon]
        action = actions[sample_idx:sample_idx + horizon].flatten()

        return state, y, action

    def __del__(self):
        # Ensure all files are closed when the object is deleted
        for handle in self.file_handles.values():
            handle.close()


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
    def __init__(self, x, y, a, num_trajectories):
        self.x_trajectories = x
        self.y_trajectories = y
        self.a_trajectories = a
        self.num_trajectories = num_trajectories
    
    def __getitem__(self, idx):
        return self.x_trajectories[idx], self.y_trajectories[idx], self.a_trajectories[idx]
    
    def __len__(self):
        return self.num_trajectories
    
class Dataset_case_3(torch.utils.data.Dataset):
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
    def __init__(self, x, y, a, i, num_trajectories):
        self.x_trajectories = x
        self.y_trajectories = y
        self.a_trajectories = a
        self.i_trajectories = i
        self.num_trajectories = num_trajectories
    
    def __getitem__(self, idx):
        return self.x_trajectories[idx], self.y_trajectories[idx], self.a_trajectories[idx], self.i_trajectories[idx]
    
    def __len__(self):
        return self.num_trajectories
    
class Dataset_case_2(torch.utils.data.Dataset):
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
    def __init__(self, x, y, a, num_trajectories):
        self.x_trajectories = x
        self.y_trajectories = y
        self.a_trajectories = a
        self.num_trajectories = num_trajectories
    
    def __getitem__(self, idx):
        return self.x_trajectories[idx], self.y_trajectories[idx], self.a_trajectories[idx]
    
    def __len__(self):
        return self.num_trajectories
    
class Dataset_case_wm_each(torch.utils.data.Dataset):
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
    def __init__(self, x, y, a, img, num_trajectories):
        self.x_trajectories = x
        self.y_trajectories = y
        self.a_trajectories = a
        self.img_trajectories = img
        self.num_trajectories = num_trajectories
    
    def __getitem__(self, idx):
        return self.x_trajectories[idx], self.y_trajectories[idx], self.a_trajectories[idx], self.img_trajectories[idx]
    
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

def construct_input_output_link(cfg: TrainConfig, states, actions):
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
        
    forces = states[:, -56:]

    forces_transformed = torch.zeros((forces.shape[0], 4))

    for i in range(forces.shape[0]):
        if torch.any(forces[i, 0:9] > 0):
            forces_transformed[i, 0] = 1
        if torch.any(forces[i, 9:18] > 0):
            forces_transformed[i, 1] = 1
        if torch.any(forces[i, 18:37] > 0):
            forces_transformed[i, 2] = 1
        if torch.any(forces[i, 37:56] > 0):
            forces_transformed[i, 3] = 1


    y_seq = forces_transformed[horizon:]

    # if "joint_vel" in cfg.remove_out and "skin" in cfg.remove_out:
    #         y_seq = y_seq[:, :-63]
    # elif "joint_vel" in cfg.remove_out:
    #     y_seq = torch.cat((y_seq[:, :7], y_seq[:, 14:]), dim=1)
    # elif "skin" in cfg.remove_out:
    #     y_seq = y_seq[:, :-56]

    # 0/1 input
    states_seq[:, :, -56:] = (states_seq[:, :, -56:] > 0).float()

    states_seq = states_seq.view(-1, input_window_size, states_seq.shape[-1])
    y_seq = y_seq.view(-1, 1, y_seq.shape[-1])
    action_seq = action_seq.view(-1, 1, action_seq.shape[-1])
    # imgs_seq = 0
    return states_seq, y_seq, action_seq

def construct_input_output_taxel(cfg: TrainConfig, states, actions, link_name):
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
        
    forces = states[:, -56:]

    # if link_name == 'Link_5':
    #     forces_transformed = forces[:, 0:9]
    # elif link_name == 'Link_4':
    #     forces_transformed = forces[:, 9:18]
    # elif link_name == 'Link_3':
    #     forces_transformed = forces[:, 18:37]
    # elif link_name == 'Link_2':
    #     forces_transformed = forces[:, 37:56]
    forces_transformed = forces

    # forces_transformed = torch.where(forces_transformed > 0, torch.tensor(1.0), torch.tensor(0.0))

    y_seq = forces_transformed[horizon:]

    # 0/1 input
    # states_seq[:, :, -56:] = (states_seq[:, :, -56:] > 0).float()
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

def construct_input_output_taxel_case_3(cfg: TrainConfig, states, actions, link_name):
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
        
    forces = states[:, -56:]

    # if link_name == 'Link_5':
    #     forces_transformed = forces[:, 0:9]
    # elif link_name == 'Link_4':
    #     forces_transformed = forces[:, 9:18]
    # elif link_name == 'Link_3':
    #     forces_transformed = forces[:, 18:37]
    # elif link_name == 'Link_2':
    #     forces_transformed = forces[:, 37:56]
    forces_transformed = forces

    # forces_transformed = torch.where(forces_transformed > 0, torch.tensor(1.0), torch.tensor(0.0))

    y_seq = forces_transformed[horizon:]

    # 0/1 input
    # states_seq[:, :, -56:] = (states_seq[:, :, -56:] > 0).float()
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

def construct_input_output_taxel_wm(cfg: TrainConfig, states, actions):
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
        
    # forces = states

    # if link_name == 'Link_5':
    #     forces_transformed = forces[:, 0:9]
    # elif link_name == 'Link_4':
    #     forces_transformed = forces[:, 9:18]
    # elif link_name == 'Link_3':
    #     forces_transformed = forces[:, 18:37]
    # elif link_name == 'Link_2':
    #     forces_transformed = forces[:, 37:56]
    # forces_transformed = forces

    # forces_transformed = torch.where(forces_transformed > 0, torch.tensor(1.0), torch.tensor(0.0))

    y_seq = states[horizon:]

    # 0/1 input
    # states_seq[:, :, -56:] = (states_seq[:, :, -56:] > 0).float()
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

def construct_input_output_taxel_wm_each(cfg: TrainConfig, states, actions, imgs, sample_idx):
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

    state = states[sample_idx]
    y = states[sample_idx + horizon]
    action = actions[sample_idx:sample_idx + horizon].flatten()
    img = imgs[sample_idx]
    img = img.permute(2, 0, 1)


    # state = state.view(-1, state.shape[-1])
    # y = y.view(-1, y.shape[-1])
    # action = action.view(-1, action.shape[-1])
    # img = img.view(-1, img.shape[-3], img.shape[-2], img.shape[-1])

    return state, y, action, img