"""
This module provides utility functions for logging data to Weights & Biases.
"""

import wandb
import yaml
import os
from dataclasses import asdict

from cfg.train_cfg import TrainConfig


def init_wandb(cfg: TrainConfig):
    """Initialize Weights & Biases."""

    # Initialize Weights & Biases
    if cfg.log_wandb:
        wandb.init(project="MBLHD", dir="/tmp", save_code=True)
    else:
        wandb.init(project="MBLHD", dir="/tmp", save_code=True, mode="disabled")

    # Update the configuration from the Weights & Biases sweep
    if wandb.run.sweep_id is not None:
        update_cfg_from_wandb_sweep(cfg)

    # Check if cfg is already a dictionary, if not convert it to a dictionary
    if isinstance(cfg, dict):
        wandb.config.update(cfg)
    else:
        wandb.config.update(asdict(cfg))

    # Use global step for logging
    wandb.define_metric("*", step_metric="step")


def update_cfg_from_wandb_sweep(cfg: TrainConfig):
    """Update the training configuration from a Weights & Biases sweep configuration."""
    for key, value in wandb.config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        elif hasattr(cfg.network_config, key):
            setattr(cfg.network_config, key, value)
        else:
            raise ValueError(
                f'Invalid key "{key}" in Weights & Biases sweep configuration.'
            )
