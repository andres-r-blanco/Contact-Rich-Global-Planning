"""
This module contains the implementation of an MDN-RNN network, which is a Recurrent 
Neural Network (RNN) with a Mixture Density Network (MDN) output layer. The RNN has a 
Long Short-Term Memory (LSTM) architecture.

For details see: https://arxiv.org/pdf/1803.10122
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from dataclasses import dataclass

from cfg.network_cfg import RNNConfig


class MDNRNN(nn.Module):
    """MDN-RNN model implementation."""

    def __init__(self, rnn_config: RNNConfig):
        super(MDNRNN, self).__init__()
        assert rnn_config.input_size > 0, "Input size must be greater than 0."
        assert rnn_config.output_size > 0, "Output size must be greater than 0."
        self.input_size = rnn_config.input_size
        self.output_size = rnn_config.output_size
        self.hidden_size = rnn_config.hidden_size
        self.num_gaussians = rnn_config.num_gaussians

        # The LSTM layer
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)

        # The output layer
        self.fc = nn.Linear(
            self.hidden_size,
            self.num_gaussians * (2 * self.output_size + 1),
        )  # 2 * output_size = mean and variance, +1 = mixing coefficient pi

    def forward(self, x, hidden=None):
        # Forward pass through the LSTM layer
        x, hidden = self.rnn(x, hidden)

        # Forward pass through the output layer
        x = self.fc(x)

        # Split the output into pi, mu, and sigma
        pi, mu, sigma = torch.split(
            x,
            [
                self.num_gaussians,
                self.num_gaussians * self.output_size,
                self.num_gaussians * self.output_size,
            ],
            dim=-1,
        )

        # Softmax the pi values to ensure they sum to 1
        pi = F.softmax(pi, dim=-1)

        # Exponentiate the sigma values to ensure they are positive
        sigma = torch.exp(sigma)

        # View mu and sigma as (batch_size, sequence_length, num_gaussians, output_size)
        mu = mu.view(mu.shape[0], mu.shape[1], self.num_gaussians, self.output_size)
        sigma = sigma.view(
            sigma.shape[0], sigma.shape[1], self.num_gaussians, self.output_size
        )

        return pi, mu, sigma, hidden

    def loss(self, pi, mu, sigma, target):
        # Calculate the log loss
        dist = Normal(mu, sigma)

        # Expand the target to match the number of Gaussians
        target = target.unsqueeze(-2).expand(-1, -1, self.num_gaussians, -1)

        # Calculate the log probability of the target
        log_prob = dist.log_prob(target)

        # Sum over the state size
        log_prob = torch.sum(log_prob, dim=-1)

        # Scale the log probability by the mixing coefficient
        log_prob += torch.log(pi)

        # Logsumexp over the Gaussian components
        log_prob = torch.logsumexp(log_prob, dim=-1)

        return -torch.mean(log_prob)

    def compute_loss(self, x, target):
        """
        Compute the loss of the MDN-RNN model.

        Args:
            x (torch.Tensor) [Batch, Sequence, Input]: The input sequence.
            target (torch.Tensor) [Batch, Sequence, Output]: The target sequence.

        Returns:
            loss (torch.Tensor): The loss of the model.
        """

        # Forward pass
        pi, mu, sigma, _ = self.forward(x)

        # Calculate the loss
        loss = self.loss(pi, mu, sigma, target)

        return loss

    def sample(self, pi, mu, sigma, temperature):
        """
        Return a sample from the MDN given pi, mu, and sigma.

        Args:
            pi (torch.Tensor) [Batch, Sequence, Gaussians]: The mixing coefficients.
            mu (torch.Tensor) [Batch, Sequence, Gaussians, Output]: The means of the Gaussians.
            sigma (torch.Tensor) [Batch, Sequence, Gaussians, Output]: The standard deviations of the Gaussians.
            temperature (float): The stochasticity of the sample in the range of [0, 1].

        Returns:
            sample (torch.Tensor) [Batch, Sequence, Output]: A sample from the MDN.
        """

        # Sample the Gaussian
        pi = F.softmax(torch.log(pi) / temperature, dim=-2)
        cat_dist = Categorical(pi)
        gauss_idx = cat_dist.sample()

        # Sample from the chosen Gaussian
        gauss_idx = (
            gauss_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.output_size)
        )
        chosen_mu = mu.gather(-2, gauss_idx).squeeze(-2)
        chosen_sigma = sigma.gather(-2, gauss_idx).squeeze(-2) * math.sqrt(temperature)

        dist = Normal(chosen_mu, chosen_sigma)
        sample = dist.sample()

        return sample

    def predict(self, x, hidden=None, temperature=1.0):
        """
        Compute the stochastic prediction of the MDN-RNN model.

        Args:
            x (torch.Tensor) [Batch, 1, Input]: The input sequence.
            hidden (tuple): The hidden state of the LSTM.
            temperature (float): The stochasticity of the sample in the range of [0, 1].

        Returns:
            sample (torch.Tensor) [Batch, 1, Output]: The predicted next step.
            hidden (tuple): The updated hidden state of the LSTM.
        """

        # Forward pass
        pi, mu, sigma, hidden = self(x, hidden)

        # Sample from the MDN
        sample = self.sample(pi, mu, sigma, temperature)

        return sample, hidden
