"""
This module provides a basic logging functionality.
"""

import os
import wandb
import torch
import pickle
import yaml
import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataclasses import asdict

from utils.data_set import PunyoDataset, normalize, denormalize
from cfg.train_cfg import TrainConfig


def save_run(network, cfg: TrainConfig, test_data_loader, min_max_vals):
    """
    Save the training run by exporting the training configuration and the model weights
    to a log directory. The function also exports a trajectory from the network for
    visualization purposes.
    """

    # Get current date and time
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.run.name = date_time

    # Build the log path
    log_path = os.path.join(cfg.root_dir, cfg.log_dir, date_time)
    os.makedirs(log_path)

    # Save the config to a yaml file
    with open(os.path.join(log_path, "config.yaml"), "w") as f:
        if isinstance(cfg, dict):
            cfg.root_dir = str(cfg.root_dir)
            yaml.dump(cfg, f, sort_keys=False)
        else:
            cfg.root_dir = str(cfg.root_dir)
            yaml.dump(asdict(cfg), f, sort_keys=False)

    # Export a trajectory from the network
    # export_trajectory(network, cfg, log_path, test_data_loader, min_max_vals)

    # Save the model and the normalization parameters
    # TODO: Save the model at multiple checkpoints and log the best one
    save_dict = {
        "model": network.state_dict(),
        "normalization": min_max_vals,
    }
    torch.save(save_dict, os.path.join(log_path, "model.pt"))
    wandb.save(os.path.join(log_path, "model.pt"), base_path=log_path)
    wandb.finish()


def export_trajectory(
    network, cfg: TrainConfig, log_path, test_data_loader, min_max_vals
):
    """
    Export a trajectory from the network for visualization purposes.

    This function outputs a pickle file containing ground truth trajectories from the
    dataset and predicted trajectories from the network. The pickle file can then be
    visualized in Anzu using:

    bazel run //tactile/punyo:punyo_trajectory_visualizer -- --punyo_v2
    --trajectory_file_path {path_to_pickle_file}
    """

    with torch.no_grad():
        for i in range(cfg.record_trajectories):
            # Get a sample trajectory
            x, y = next(iter(test_data_loader))
            action_size = (
                cfg.network_config.input_size
                - cfg.network_config.output_size * cfg.sequence_length
            )

            # Set the initial hidden state to zero for recurrent networks
            hidden = (
                torch.zeros(1, 1, network.hidden_size),
                torch.zeros(1, 1, network.hidden_size),
            )

            # Predict the trajectory
            y_pred = predict_trajectory(
                cfg, network, x, hidden, action_size, min_max_vals
            )

            # Denormalize the data with the original scaling values of the target
            if cfg.normalization:
                y = denormalize(
                    y,
                    min_max_vals["y_min_vals"],
                    min_max_vals["y_max_vals"],
                )
                y_pred = denormalize(
                    y_pred,
                    min_max_vals["y_min_vals"],
                    min_max_vals["y_max_vals"],
                )

            # Recover the absolute state from the delta
            if cfg.predict_delta:
                y = get_state_from_delta(cfg, x, y, action_size, min_max_vals)
                y_pred = get_state_from_delta(cfg, x, y_pred, action_size, min_max_vals)

            # Squeeze the tensors
            y = y.squeeze()
            y_pred = y_pred.squeeze()

            # Plot the first trajectory
            if i == 0 and cfg.plot_trajectory and wandb.run.sweep_id is None:
                plot_trajectory(cfg, y, y_pred)

            # Save the trajectory
            with open(os.path.join(log_path, f"trajectory_{i}.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "trajectory_target": y.numpy(),
                        "trajectory_pred": y_pred.numpy(),
                    },
                    f,
                )


def predict_trajectory(cfg: TrainConfig, network, x, hidden, action_size, min_max_vals):
    """
    Predict a trajectory using the trained network.

    Args:
        cfg (TrainConfig): The configuration object.
        network (nn.Module): The trained network.
        x (torch.Tensor): The input tensor.
        hidden (tuple): The hidden state of the recurrent network.
        action_size (int): The size of the action space.
        min_max_vals (dict): The normalization parameters.

    Returns:
        y_pred (torch.Tensor): The predicted trajectory.

    """
    breakpoint()
    y_pred = torch.zeros((x.shape[0], cfg.network_config.output_size))

    # Predict the trajectory
    if cfg.training_objective == "forward":
        # Predict the first time step
        y_pred, hidden = network.predict(
            x,
            hidden=hidden,
            temperature=cfg.temperature,
        )
        state = x[:, :28]  # Needed for delta prediction

        for j in range(y_pred.shape[1] - 1):
            if cfg.predict_from_ground_truth:
                # Predict the next time step from the ground truth (x)
                state = x[:, [j + 1], :-action_size]
            else:
                # Predict the next time step from the previous prediction (y_pred)
                if cfg.predict_delta:
                    if cfg.normalization:
                        state = add_delta_to_state(
                            state, y_pred[:, [j]], min_max_vals, action_size
                        )
                    else:
                        state += y_pred[:, [j]]
                else:
                    if cfg.normalization:
                        state = set_prediction_to_state(
                            y_pred[:, [j]], min_max_vals, action_size
                        )
                    else:
                        state = y_pred[:, [j]]

            # Concatenate the state and action
            action = x[:, [j + 1], -action_size:]
            input = torch.cat([state, action], dim=-1)

            # Predict
            y_pred[:, j + 1], hidden = network.predict(
                input,
                hidden=hidden,
                temperature=cfg.temperature,
            )

    # Predict the action from a state pair
    elif cfg.training_objective == "inverse":
        for j in range(y_pred.shape[1]):
            y_pred[:, j] = network(x[:, [j]])

    return y_pred


def plot_trajectory(cfg: TrainConfig, y, y_pred):
    """
    Plot the target and predicted trajectory.
    """

    if cfg.training_objective == "forward":
        fig, axs = plt.subplots(2, 2)
        idx = 0
        if "state" in cfg.obs_types:
            # Plot the position
            axs[0, 0].plot(y[:, idx : idx + 14])
            axs[0, 0].set_prop_cycle(None)  # reset color cycle
            axs[0, 0].plot(y_pred[:, idx : idx + 14], linestyle="--")
            axs[0, 0].set_title("Position [rad]")
            idx += 14

            # Plot the velocity
            axs[1, 0].plot(y[:, idx : idx + 14])
            axs[1, 0].set_prop_cycle(None)  # reset color cycle
            axs[1, 0].plot(y_pred[:, idx : idx + 14], linestyle="--")
            axs[1, 0].set_title("Velocity [rad/s]")
            idx += 14

        if "torque" in cfg.obs_types:
            # Plot the torque
            axs[0, 1].plot(y[:, idx : idx + 14])
            axs[0, 1].set_prop_cycle(None)  # reset color cycle
            axs[0, 1].plot(y_pred[:, idx : idx + 14], linestyle="--")
            axs[0, 1].set_title("Torque [Nm]")
            idx += 14

        if "distal_arm_pressures" in cfg.obs_types:
            # Plot the distal arm pressures
            axs[1, 1].plot(y[:, idx : idx + 26])
            axs[1, 1].set_prop_cycle(None)  # reset color cycle
            axs[1, 1].plot(y_pred[:, idx : idx + 26], linestyle="--")
            axs[1, 1].set_title("Pressure [hPa]")
            idx += 26

        # Set the legend
        lines = [
            Line2D([0], [0], lw=2),
            Line2D([0], [0], linestyle="--", lw=2),
        ]
        labels = ["Ground Truth", "Predicted"]
        for ax in axs.flat:
            ax.legend(lines, labels)

    elif cfg.training_objective == "inverse":
        fig, axs = plt.subplots(1, 1)
        axs.plot(y)
        axs.set_prop_cycle(None)  # reset color cycle
        axs.plot(y_pred, linestyle="--")
        axs.set_title("Inverse Dynamics")
        lines = [
            Line2D([0], [0], lw=2),
            Line2D([0], [0], linestyle="--", lw=2),
        ]
        labels = ["Ground Truth", "Predicted"]
        axs.legend(lines, labels)

    plt.show()


def set_prediction_to_state(pred, min_max_vals, action_size):
    """
    Set the prediction to the state for the next time step. This requires first
    denormalizing the prediction and then normalizing it with the normalization
    parameters of the state.
    """

    # Denormalize the prediction
    pred_denorm = denormalize(
        pred,
        min_max_vals["y_min_vals"],
        min_max_vals["y_max_vals"],
    )

    # Normalize the prediction
    pred_norm = normalize(
        pred_denorm,
        min_max_vals["x_min_vals"][:-action_size],
        min_max_vals["x_max_vals"][:-action_size],
    )

    return pred_norm


def add_delta_to_state(state, delta, min_max_vals, action_size):
    """
    Add the delta to the state for the next time step. This requires first denormalizing
    the state and the delta, adding the delta to the state, and then normalizing the
    state again.

    """

    # Denormalize the state
    state_denorm = denormalize(
        state,
        min_max_vals["x_min_vals"][:-action_size],
        min_max_vals["x_max_vals"][:-action_size],
    )

    # Denormalize the delta
    delta_denorm = denormalize(
        delta,
        min_max_vals["y_min_vals"],
        min_max_vals["y_max_vals"],
    )

    # Add the delta to the state
    state_denorm = state_denorm + delta_denorm

    # Normalize the state
    state = normalize(
        state_denorm,
        min_max_vals["x_min_vals"][:-action_size],
        min_max_vals["x_max_vals"][:-action_size],
    )

    return state


def get_state_from_delta(cfg: TrainConfig, x, y, action_size, min_max_vals):
    """
    Recover the state trajectory from the delta trajectory.
    """

    if cfg.training_objective == "forward" and cfg.predict_delta:
        # Get the initial state
        if cfg.normalization:
            state = denormalize(
                x[:, [0], :-action_size],
                min_max_vals["x_min_vals"][:-action_size],
                min_max_vals["x_max_vals"][:-action_size],
            )
        else:
            state = x[:, [0], :-action_size]

        # Sum the deltas along the trajectory and add them to the initial state
        return torch.cumsum(y, dim=1) + state
    else:
        return y
