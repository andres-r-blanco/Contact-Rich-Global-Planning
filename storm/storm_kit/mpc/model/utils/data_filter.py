import numpy as np
import torch
import os
import random
import sys

obs_types: tuple = (
        "joint_pos",
        "joint_vel",
        "skin",
    )

files = os.listdir("/home/angchen/xac/data/world_model/test_data")

for file in files:
    if file.endswith(".npz"):
        try:
            with np.load(os.path.join("/home/angchen/xac/data/world_model/test_data", file), allow_pickle=True, mmap_mode='c') as data:
                data = data["arr_0"]

                states = np.array([np.concatenate([data[i][key] for key in obs_types]) for i in range(len(data))])
                actions = np.array([data[i]['action'] for i in range(len(data))])
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                print("success")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            os.remove(os.path.join("/home/angchen/xac/data/world_model/test_data", file))
            print(f"{file} has been deleted.")
