import os
import sys
import torch
import wandb
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml
from dataclasses import asdict

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cfg.train_cfg import TrainConfig
from cfg.network_cfg import RNNConfig, MLPConfig, AEConfig, DYNConfig
from networks.mdn_rnn import MDNRNN
from networks.mlp import MLP, AEMLP, Dynamics, AE, VAE, WM_dynamics, WM_encoder, WM_predictor_force, WM_encoder_baseline, WM_predictor_force_baseline, WM_TransformerDynamics_baseline
from utils.data_set import ToyEnvDataset_wm, ToyEnvDatasetAE, Dataset, ToyEnvDataset_wm_each, Dataset_case_wm_each, ToyEnvDataset_wm_baseline
from utils.wandb_utils import init_wandb
from utils.logger import save_run
import datetime
import torch.nn.functional as F
import time
import numpy as np
from utils.data_set import normalize, denormalize
import cv2
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, classification_report
import json
import random

# Define paths to the test data and model
# test_name = '/home/angchen/xac/data/world_model/test_data_100/ee_trajectory20240814-212350.npz'
test_folder = '/home/angchen/xac/projects/world_model/multi_priority_control/multipriority/multipriority/learned_dynamics_reg_single/test_data'
# save_folder = '/home/angchen/xac/projects/world_model/multi_priority_control/multipriority/multipriority/learned_dynamics_reg_single/test_data_error'

# os.makedirs(save_folder, exist_ok=True)


# Main function to run the data processing and visualization
def main():
    # Get current date and time
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get the root directory
    root_dir = Path(__file__).resolve().parents[2]

    # Create the configuration for training
    cfg = TrainConfig(root_dir=root_dir)
    cfg.train_taxel = True
    cfg.network_config.input_size = 14
    cfg.network_config.output_size = 70
    cfg.network_config.action_size = 70
    cfg.network_config.sequence_length = 1

    # Initialize the models
    encoder = WM_encoder_baseline(cfg.network_config)
    encoder = encoder.to("cuda")
    predictor = WM_predictor_force_baseline(cfg.network_config)
    predictor = predictor.to("cuda")
    dynamics = WM_TransformerDynamics_baseline(cfg.network_config)
    dynamics = dynamics.to("cuda")

    # Load the checkpoint (best or final model)
    # checkpoint_path = os.path.join(cfg.root_dir, cfg.log_dir, 'best_model.pth')  # or 'final_model.pth'
    checkpoint_path = '/home/angchen/xac/projects/world_model/multi_priority_control/multipriority/multipriority/learned_dynamics_reg_single/logs/baseline/best_model.pth'
    checkpoint = torch.load(checkpoint_path)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    predictor.load_state_dict(checkpoint['predictor_state_dict'])
    dynamics.load_state_dict(checkpoint['dynamics_state_dict'])

    # Set the models to evaluation mode
    encoder.eval()
    predictor.eval()
    dynamics.eval()

    files_all = os.listdir(test_folder)
    random.shuffle(files_all)
    # 随机抽取 10% 的文件
    num_files = len(files_all)
    num_sample = max(1, int(num_files * 1))  # 至少抽取一个文件
    sampled_files = random.sample(files_all, num_sample)

    batch_size = cfg.batch_size

    for file in sampled_files:
        data = np.load(os.path.join(test_folder, file))
        x = torch.tensor(data['x'], dtype=torch.float32)
        # img = torch.tensor(data['img'], dtype=torch.float32).permute(0, 3, 1, 2).to('cuda')
        a = torch.tensor(data['a'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.float32)
        # y = y[:, 14:]

        indices_list = []

        # 分批处理
        for start in range(0, x.shape[0], batch_size):
            end = start + batch_size
            x_batch = x[start:end].to('cuda')
            # img_batch = img[start:end]
            a_batch = a[start:end].to('cuda')
            y_batch = y[start:end].to('cuda')

            hidden = encoder(x_batch)
            pred_hidden = dynamics(hidden, a_batch)
            forces_pred = torch.stack([predictor(pred_hidden[:, t]) for t in range(cfg.horizon)], dim=1)
            loss_force = F.mse_loss(forces_pred, y_batch[:, :, 14:70])
            print(f"loss_force: {loss_force}")

            # for i in range(forces_pred.shape[0]):
            #     abs_diff = torch.abs(forces_pred[i] - y_batch[i])
            #     if np.any(abs_diff.cpu().detach().numpy() > 5):
            #         indices_list.append(start + i)

        # if indices_list:
        #     new_data = {
        #         'x': x[indices_list].cpu().numpy(),
        #         'y': y[indices_list].cpu().numpy(),
        #         'a': a[indices_list].cpu().numpy(),
        #         'img': img[indices_list].cpu().numpy(),
        #     }
        #     save_path = os.path.join(save_folder, f"error_{file}")
        #     np.savez(save_path, **new_data)
        #     print(f"保存文件: {save_path}")


if __name__ == "__main__":
    main()
