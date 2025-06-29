"""
This module contains the implementation of a basic Multi-Layer Perceptron (MLP) network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfg.network_cfg import MLPConfig, AEConfig, DYNConfig, NetworkConfig

class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""
	
	def __init__(self, simnorm_dim):
		super().__init__()
		self.dim = simnorm_dim
	
	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)
		
	def __repr__(self):
		return f"SimNorm(dim={self.dim})"

class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0., act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = x.float()
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        if self.act:
            return self.act(self.ln(x))
        else:
            return self.ln(x)
    
    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        repr_act = f", act={self.act.__class__.__name__}" if self.act else ", act=None"
        return f"NormedLinear(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"bias={self.bias is not None}{repr_dropout}{repr_act})"
    
class Linear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0., act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.act = act
        self.dropout = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = x.float()
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        if self.act:
            return self.act(x)
        else:
            return x
    
    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        repr_act = f", act={self.act.__class__.__name__}" if self.act else ", act=None"
        return f"Linear(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"bias={self.bias is not None}{repr_dropout}{repr_act})"
     
class NormedMLP(nn.Module):
    """MLP model implementation."""

    def __init__(self, input_size, output_size, hidden_size, num_layers, act, dropout=0., simnorm_dim=8, simnorm=True):
        super(NormedMLP, self).__init__()

        # Initialize the MLP network
        if simnorm:
            self.mlp = nn.Sequential(
                NormedLinear(input_size, hidden_size, dropout=dropout, act=act),
                *[NormedLinear(hidden_size, hidden_size, act=act)]
                * (num_layers),
                NormedLinear(hidden_size, output_size, act=SimNorm(simnorm_dim)),
            )
        else:
            self.mlp = nn.Sequential(
                NormedLinear(input_size, hidden_size, dropout=dropout, act=act),
                *[NormedLinear(hidden_size, hidden_size, act=act)]
                * (num_layers),
                NormedLinear(hidden_size, output_size, act=None),
            )

    def forward(self, x):
        # Forward pass through the MLP network
        return self.mlp(x)
    
class MLP(nn.Module):
    """MLP model implementation."""

    def __init__(self, input_size, output_size, hidden_size, num_layers, act, dropout=0., simnorm_dim=8, simnorm=True):
        super(MLP, self).__init__()

        # Initialize the MLP network
        if simnorm:
            self.mlp = nn.Sequential(
                Linear(input_size, hidden_size, dropout=dropout, act=act),
                *[Linear(hidden_size, hidden_size, act=act)]
                * (num_layers),
                Linear(hidden_size, output_size, act=SimNorm(simnorm_dim)),
            )
        else:
            self.mlp = nn.Sequential(
                Linear(input_size, hidden_size, dropout=dropout, act=act),
                *[Linear(hidden_size, hidden_size, act=act)]
                * (num_layers),
                Linear(hidden_size, output_size, act=None),
            )

    def forward(self, x):
        # Forward pass through the MLP network
        return self.mlp(x)


class AEMLP(nn.Module):
    """UNet encoder-decoder model implementation where input and output are same size."""
    def __init__(self, mlp_config: MLPConfig):
        super(AEMLP, self).__init__()
        assert mlp_config.input_size > 0, "Input size must be greater than 0."
        assert mlp_config.output_size > 0, "Output size must be greater than 0."
        self.input_size = mlp_config.input_size
        self.output_size = mlp_config.input_size
        self.hidden_size = mlp_config.hidden_size
        self.num_layers = mlp_config.num_layers
        self.activation = mlp_config.activation

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()

        # Initialize the MLP network such that encoder decreases in size to latent size and decoder increases back to input size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),
            activation,
            nn.Linear(64, 64),
            activation,
            nn.Linear(64, 32),
            activation,
            nn.Linear(32, 16),
            activation,
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            activation,
            nn.Linear(16, 32),
            activation,
            nn.Linear(32, 64),
            activation,
            nn.Linear(64, 64),
            activation,
            nn.Linear(64, self.input_size),
        )

    def forward(self, x):
        # Forward pass through the
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss(self, z, x, pred):
        # Loss is MSE + KL Divergence
        kl_div = 0.5 * torch.sum(torch.exp(z) + z**2 - 1.0 - z)
        mse = F.mse_loss(pred, x)
        # print ("mse loss: ", mse)
        return mse + kl_div
        
    def compute_loss(self, x, target):
        """
        Compute 

        Args:
            x (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss
        """

        # Forward pass
        z = self.encoder(x)
        pred = self.decoder(z)

        # Calculate the loss
        loss = self.loss(z, x, pred)

        return loss
    
    def predict(self, x, hidden=None, temperature=None):
        """
        Compute the prediction of the network.

        Args:
            x (torch.Tensor): The input tensor.
            hidden (torch.Tensor): Not used in the MLP.
            temperature (float): Not used in the MLP.

        Returns:
            pred (torch.Tensor): The predicted output.
            hidden (torch.Tensor): Not used in the MLP.
        """

        # Forward pass
        hidden = self.encoder(x)
        pred = self.decoder(hidden)

        return pred, hidden
        
# Implementing the Variational Autoencoder class
class VAE(nn.Module):
    def __init__(self, mlp_config: NetworkConfig):
        super(VAE, self).__init__()
        assert mlp_config.input_size > 0, "Input size must be greater than 0."

        self.input_size = mlp_config.input_size
        self.hidden_size = mlp_config.hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout
        self.simnorm_dim = mlp_config.simnorm_dim
        self.num_layers = mlp_config.AE_num_layers
        self.enable_simnorm = mlp_config.enable_AE_simnorm

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()

        # Encoder
        if self.enable_simnorm :
            self.encoder = MLP(self.input_size, self.hidden_size * 2, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim)
        else:
            self.encoder = MLP(self.input_size, self.hidden_size * 2, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)
        # self.encoder = MLP(self.input_size, self.hidden_size * 2, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)
  
        # Decoder
        self.decoder = MLP(self.hidden_size, self.input_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)


    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        # print(log_var)
        z = mu + std * torch.randn_like(std)
        # z = mu
        x_recon = self.decoder(z)
        return x_recon, mu, log_var, z
# class VAE(nn.Module):
#     def __init__(self, mlp_config: NetworkConfig):
#         super(VAE, self).__init__()
#         assert mlp_config.input_size > 0, "Input size must be greater than 0."

#         self.input_size = mlp_config.input_size
#         self.hidden_size = mlp_config.hidden_size
#         self.activation = mlp_config.activation
#         self.dropout = mlp_config.dropout
#         self.simnorm_dim = mlp_config.simnorm_dim
#         self.num_layers = mlp_config.AE_num_layers
#         self.enable_simnorm = mlp_config.enable_AE_simnorm

#         # Define the activation function
#         if self.activation == "relu":
#             activation = nn.ReLU()
#         elif self.activation == "tanh":
#             activation = nn.Tanh()
#         elif self.activation == "mish":
#             activation = nn.Mish()
#         elif self.activation == "silu":
#             activation = nn.SiLU()

#         # Encoder
#         if self.enable_simnorm :
#             self.encoder = MLP(self.input_size, self.hidden_size, self.hidden_size, self.num_layers - 1, activation, self.dropout, self.simnorm_dim)
#         else:
#             self.encoder = MLP(self.input_size, self.hidden_size, self.hidden_size, self.num_layers - 1, activation, self.dropout, self.simnorm_dim, simnorm=False)
#         # self.encoder = MLP(self.input_size, self.hidden_size * 2, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)
#         self.mean_layer = NormedLinear(self.hidden_size, self.hidden_size, act=activation)
#         self.var_layer = NormedLinear(self.hidden_size, self.hidden_size, act=activation)
#         # self.mean_layer = nn.Linear(self.hidden_size, self.hidden_size)
#         # self.var_layer = nn.Linear(self.hidden_size, self.hidden_size)
#         # Decoder
#         self.decoder = MLP(self.hidden_size, self.input_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)


#     def forward(self, x):
#         h = self.encoder(x)
#         mu = self.mean_layer(h)
#         log_var = self.var_layer(h)
#         # print(mu.max(), mu.min())
#         # print(log_var.max(), log_var.min())
#         std = torch.exp(0.5 * log_var)
#         z = mu + std * torch.randn_like(std)
#         # print(z.max(), z.min())
#         x_recon = self.decoder(z)
#         return x_recon, mu, log_var, z

    
    def loss(self, x, x_recon, mu, log_var):
        # reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        # KL loss
        # kl_div = kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1))

        # kl_div = torch.tensor(0)
        # kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # kl_div = kl_div / x.size(0)
        return recon_loss, kl_div

    def compute_loss(self, x):
        x_recon, mu, log_var, z = self.forward(x)
        recon_loss, kl_loss = self.loss(x, x_recon, mu, log_var)
        return recon_loss, kl_loss
    
# Implementing the Autoencoder class
class AE(nn.Module):
    def __init__(self, mlp_config: NetworkConfig):
        super(AE, self).__init__()
        assert mlp_config.input_size > 0, "Input size must be greater than 0."

        self.input_size = mlp_config.input_size
        self.hidden_size = mlp_config.hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout
        self.simnorm_dim = mlp_config.simnorm_dim
        self.num_layers = mlp_config.AE_num_layers
        self.enable_simnorm = mlp_config.enable_AE_simnorm

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()

        # self.encoder = nn.Sequential(
        #         nn.Linear(self.input_size, self.hidden_size),
        #         activation,
        #         *[nn.Linear(self.hidden_size, self.hidden_size),
        #         activation,] * (self.num_layers),
        #         nn.Linear(self.hidden_size, self.hidden_size),
        #         # activation
        #     )
        
        # self.decoder = nn.Sequential(
        #         nn.Linear(self.hidden_size, self.hidden_size),
        #         activation,
        #         *[nn.Linear(self.hidden_size, self.hidden_size),
        #         activation,] * (self.num_layers),
        #         nn.Linear(self.hidden_size, self.input_size),
        #         # activation
        #     )

        # self.encoder = nn.Sequential(
        #         nn.Linear(self.input_size, 128),
        #         activation,
        #         nn.Linear(128, 256),
        #         activation,
        #         nn.Linear(256, 512),
        #         activation,
        #         nn.Linear(512, 1024),
        #     )
        
        # self.decoder = nn.Sequential(
        #         nn.Linear(1024, 512),
        #         activation,
        #         nn.Linear(512, 256),
        #         activation,
        #         nn.Linear(256, 128),
        #         activation,
        #         nn.Linear(128, self.input_size),
        #     )

        # self.encoder = nn.Sequential(
        #         NormedLinear(self.input_size, 128, dropout=self.dropout, act=activation),
        #         NormedLinear(128, 256, dropout=self.dropout, act=activation),
        #         NormedLinear(256, 512, dropout=self.dropout, act=activation),
        #         NormedLinear(512, 1024, act=activation),
        #     )
        
        # self.decoder = nn.Sequential(
        #         NormedLinear(1024, 512, dropout=self.dropout, act=activation),
        #         NormedLinear(512, 256, dropout=self.dropout, act=activation),
        #         NormedLinear(256, 128, dropout=self.dropout, act=activation),
        #         NormedLinear(128, self.input_size, act=activation),
        #     )

        # Encoder
        if self.enable_simnorm:
            self.encoder = MLP(self.input_size, self.hidden_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim)
        else:
            self.encoder = MLP(self.input_size, self.hidden_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)
        # self.encoder = MLP(self.input_size, self.hidden_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)
  
        # Decoder
        self.decoder = MLP(self.hidden_size, self.input_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)


    def forward(self, x):
        h = self.encoder(x)
        x_recon = self.decoder(h)
        return x_recon, h

    
    def loss(self, x, x_recon):
        # reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        return recon_loss

    def compute_loss(self, x):
        x_recon, h = self.forward(x)
        loss = self.loss(x, x_recon)
        return loss
    
    

class Dynamics(nn.Module):
    def __init__(self, mlp_config: NetworkConfig):
        super(Dynamics, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = mlp_config.hidden_size * mlp_config.sequence_length + mlp_config.action_size
        self.output_size = mlp_config.hidden_size
        self.hidden_size = mlp_config.Dyn_hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout
        self.simnorm_dim = mlp_config.simnorm_dim
        self.num_layers = mlp_config.Dyn_num_layers
        self.action_size = mlp_config.action_size
        self.sequence_length = mlp_config.sequence_length
        self.enable_simnorm = mlp_config.enable_AE_simnorm

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()

        # Dynamics model (can be changed with horizon)
        if self.enable_simnorm:
            self.dynamics = NormedMLP(self.input_size, self.output_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim)
        else:
            self.dynamics = NormedMLP(self.input_size, self.output_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)    
        # self.dynamics = nn.Sequential(
        #         NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
        #         NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
        #         NormedLinear(self.hidden_size, self.output_size, act=activation),
        #     )
        # self.dynamics = nn.Sequential(
        #         nn.Linear(self.input_size, 512),
        #         activation,
        #         nn.Linear(512, 512),
        #         activation,
        #         nn.Linear(512, 256)
        #     )
        
    def forward(self, hidden, action):
        # choose action with horizon
        combined_input = torch.cat([hidden, action], dim=-1)
        y_pred = self.dynamics(combined_input)

        return y_pred

    def loss(self, y, y_pred):
        # MSE loss
        mse = F.mse_loss(y, y_pred)
        # print ("mse loss: ", mse)
        return mse

        # # Calculate Ldyn(φ)
        # Ldyn = F.kl_div(F.log_softmax(y.detach(), dim=-1), F.softmax(y_pred, dim=-1), reduction='batchmean')
        # Ldyn = torch.max(torch.tensor(1.0), Ldyn)

        # # Calculate Lrep(φ)
        # Lrep = F.kl_div(F.log_softmax(y, dim=-1), F.softmax(y_pred.detach(), dim=-1), reduction='batchmean')
        # Lrep = torch.max(torch.tensor(1.0), Lrep)

        # kl_loss = 0.8 * Ldyn + 0.2 * Lrep
        # return kl_loss
        
        # # Define mixing proportions
        # uniform_proportion = 0.01
        # nn_proportion = 0.99

        # # Create uniform distribution with the same shape as y_pred
        # num_classes = y_pred.size(-1)
        # uniform_dist = torch.full_like(y_pred, fill_value=1.0 / num_classes)

        # # Compute softmax probabilities for the predictions
        # pred_probs = F.softmax(y_pred, dim=-1)
        # target_probs = F.softmax(y, dim=-1)

        # # Mix neural network output with uniform distribution
        # mixed_pred_probs = uniform_proportion * uniform_dist + nn_proportion * pred_probs
        # mixed_target_probs = uniform_proportion * uniform_dist + nn_proportion * target_probs

        # # Compute KL divergence for Ldyn
        # Ldyn = F.kl_div(F.log_softmax(mixed_target_probs.detach(), dim=-1), mixed_pred_probs, reduction='batchmean')
        # Ldyn = torch.max(torch.tensor(1.0), Ldyn)

        # # Compute KL divergence for Lrep
        # Lrep = F.kl_div(F.log_softmax(mixed_target_probs, dim=-1), mixed_pred_probs.detach(), reduction='batchmean')
        # Lrep = torch.max(torch.tensor(1.0), Lrep)

        # # Combine KL losses
        # kl_loss = 0.8 * Ldyn + 0.2 * Lrep
        # return kl_loss
    
    def compute_loss(self, hidden, action, target):
        # Forward pass
        pred = self.forward(hidden, action)

        # Calculate the loss
        loss = self.loss(pred, target)

        return loss
