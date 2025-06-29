"""
This module contains the implementation of a basic Multi-Layer Perceptron (MLP) network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..cfg.network_cfg import MLPConfig, AEConfig, DYNConfig, NetworkConfig

from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torchvision.transforms as transforms
import clip
from .coordconv import CoordConv1d, CoordConv2d, CoordConv3d

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
            self.encoder = MLP(self.input_size, self.hidden_size * 3, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim)
        else:
            self.encoder = MLP(self.input_size, self.hidden_size * 3, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)
        # self.encoder = MLP(self.input_size, self.hidden_size * 3, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)
  
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
#         # self.encoder = MLP(self.input_size, self.hidden_size * 3, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)
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

        self.input_size = mlp_config.input_size + 512
        self.output_size = mlp_config.input_size + 512
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
        self.encoder = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        # if self.enable_simnorm:
        #     self.encoder = MLP(self.input_size, self.hidden_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim)
        # else:
        #     self.encoder = MLP(self.input_size, self.hidden_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)
        # self.encoder = MLP(self.input_size, self.hidden_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)
  
        # Decoder
        self.decoder = nn.Sequential(
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )
        # self.decoder = MLP(self.hidden_size, self.input_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)


    def forward(self, x):
        h = self.encoder(x)
        x_recon = self.decoder(h)
        
        return x_recon, h

    def focal_loss(self, inputs, targets, alpha=0.9, gamma=3):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')    
        # pt = torch.exp(-BCE_loss)
        pt = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_loss = alpha_t * ((1 - pt) ** gamma) * BCE_loss
        return focal_loss.mean()
    
    def loss(self, x, x_recon):
        # reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        return recon_loss
    
    def compute_loss(self, x, x_label):
        x_recon, h = self.forward(x)
        mse_loss = self.loss(x_label, x_recon)
        return mse_loss
    
    

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
        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )
        # if self.enable_simnorm:
        #     self.dynamics = NormedMLP(self.input_size, self.output_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim)
        # else:
        #     self.dynamics = NormedMLP(self.input_size, self.output_size, self.hidden_size, self.num_layers, activation, self.dropout, self.simnorm_dim, simnorm=False)    
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

class Binary_predictor(nn.Module):
    def __init__(self, mlp_config):
        super(Binary_predictor, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = mlp_config.input_size + mlp_config.action_size
        self.output_size = 1
        self.hidden_size = mlp_config.binary_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )

        # self.dynamics = nn.Sequential(
        #         nn.Linear(self.input_size, 256),
        #         activation,
        #         nn.Linear(256, 64),
        #         activation,
        #         nn.Linear(64, 1)
        #     )

    def forward(self, state, action):
        # choose action with horizon
        combined_input = torch.cat([state, action], dim=-1)
        y_pred = self.dynamics(combined_input)
        y_pred = torch.sigmoid(y_pred)
        return y_pred

    def loss(self, y, y_pred):
        bce_loss = F.binary_cross_entropy(y_pred, y)
        return bce_loss

    def compute_loss(self, hidden, action, target):
        # Forward pass
        pred = self.forward(hidden, action)
        # Calculate the loss
        loss = self.loss(target, pred)
        return loss
    

class Binary_predictor_img(nn.Module):
    def __init__(self, mlp_config, image_channels, image_height, image_width):
        super(Binary_predictor_img, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = mlp_config.input_size + mlp_config.action_size
        self.output_size = 1
        self.hidden_size = mlp_config.binary_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Define convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate the size of the flattened image features
        conv_output_height = image_height // 4
        conv_output_width = image_width // 4
        conv_output_size = 64 * conv_output_height * conv_output_width

        self.input_size += conv_output_size

        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, state, action, image):
        # Process image input through convolutional layers
        image = image.repeat(state.shape[0], 1, 1, 1)
        conv_output = self.conv_layers(image)
        conv_output = conv_output.view(conv_output.size(0), -1)  # Flatten the conv output
        conv_output = conv_output.unsqueeze(1)

        # Combine state, action, and flattened image inputs
        combined_input = torch.cat([state, action, conv_output], dim=-1)
        y_pred = self.dynamics(combined_input)
        y_pred = torch.sigmoid(y_pred)
        return y_pred

    def loss(self, y, y_pred):
        bce_loss = F.binary_cross_entropy(y_pred, y)
        return bce_loss

    def compute_loss(self, hidden, action, image, target):
        # Forward pass
        pred = self.forward(hidden, action, image)
        # Calculate the loss
        loss = self.loss(target, pred)
        return loss
    

class Binary_predictor_img_pos(nn.Module):
    def __init__(self, mlp_config, image_channels, image_height, image_width):
        super(Binary_predictor_img_pos, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = mlp_config.input_size + mlp_config.action_size + 3
        self.output_size = 1
        self.hidden_size = mlp_config.binary_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Define convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate the size of the flattened image features
        conv_output_height = image_height // 4
        conv_output_width = image_width // 4
        conv_output_size = 64 * conv_output_height * conv_output_width

        self.input_size += conv_output_size

        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, state, action, image, pos):
        # Process image input through convolutional layers
        image = image.repeat(state.shape[0], 1, 1, 1)
        conv_output = self.conv_layers(image)
        conv_output = conv_output.view(conv_output.size(0), -1)  # Flatten the conv output
        conv_output = conv_output.unsqueeze(1)

        pos = pos.repeat(state.shape[0], 1)
        pos = pos.unsqueeze(1)

        # Combine state, action, and flattened image inputs
        combined_input = torch.cat([state, action, conv_output, pos], dim=-1)
        y_pred = self.dynamics(combined_input)
        y_pred = torch.sigmoid(y_pred)
        return y_pred

    def loss(self, y, y_pred):
        bce_loss = F.binary_cross_entropy(y_pred, y)
        return bce_loss

    def compute_loss(self, hidden, action, image, pos, target):
        # Forward pass
        pred = self.forward(hidden, action, image, pos)
        # Calculate the loss
        loss = self.loss(target, pred)
        return loss

class Binary_predictor_resnet(nn.Module):
    def __init__(self, mlp_config, link_id = -1):
        super(Binary_predictor_resnet, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        if link_id == 0:
            self.input_size = 14 + 9 + mlp_config.action_size + 512
        elif link_id == 1:
            self.input_size = 14 + 9 + mlp_config.action_size + 512
        elif link_id == 2:
            self.input_size = 14 + 19 + mlp_config.action_size + 512
        elif link_id == 3:
            self.input_size = 14 + 19 + mlp_config.action_size + 512
        else:
            self.input_size = mlp_config.input_size + mlp_config.action_size + 512
        self.output_size = 1
        self.hidden_size = mlp_config.binary_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )

        # weights = ResNet50_Weights.DEFAULT
        # self.resnet = resnet50(weights=weights)
        # self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        # self.resnet.eval()

        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # self.preprocess = weights.transforms()
        # torch.backends.cudnn.benchmark = True

        # self.dynamics = nn.Sequential(
        #         nn.Linear(self.input_size, 256),
        #         activation,
        #         nn.Linear(256, 64),
        #         activation,
        #         nn.Linear(64, 1)
        #     )

    def forward(self, state, action, img):
        img = img.repeat(state.shape[0], 1)
        img = img.unsqueeze(1)
        # imgs = img.squeeze(1)
        # imgs = imgs.permute(0, 3, 1, 2)
        # imgs = torch.stack([self.preprocess(img) for img in imgs])
        # img_features = self.resnet(imgs)
        # img_features = img_features.view(img_features.size(0), -1)
        # img_features = img_features.unsqueeze(1)

        # choose action with horizon
        combined_input = torch.cat([state, action, img], dim=-1)
        y_pred = self.dynamics(combined_input)
        y_pred = torch.sigmoid(y_pred)
        return y_pred

    def loss(self, y, y_pred):
        bce_loss = F.binary_cross_entropy(y_pred, y)
        return bce_loss

    def compute_loss(self, hidden, action, img, target):
        # Forward pass
        pred = self.forward(hidden, action, img)
        # Calculate the loss
        loss = self.loss(target, pred)
        return loss
    
class Binary_predictor_resnet_taxel(nn.Module):
    def __init__(self, mlp_config, link_id = -1):
        super(Binary_predictor_resnet_taxel, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = 14 + 1 + mlp_config.action_size + 512
        self.output_size = 1
        self.hidden_size = mlp_config.binary_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )

        # weights = ResNet50_Weights.DEFAULT
        # self.resnet = resnet50(weights=weights)
        # self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        # self.resnet.eval()

        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # self.preprocess = weights.transforms()
        # torch.backends.cudnn.benchmark = True

        # self.dynamics = nn.Sequential(
        #         nn.Linear(self.input_size, 256),
        #         activation,
        #         nn.Linear(256, 64),
        #         activation,
        #         nn.Linear(64, 1)
        #     )

    def forward(self, state, action, img):
        img = img.repeat(state.shape[0], 1)
        img = img.unsqueeze(1)
        # imgs = img.squeeze(1)
        # imgs = imgs.permute(0, 3, 1, 2)
        # imgs = torch.stack([self.preprocess(img) for img in imgs])
        # img_features = self.resnet(imgs)
        # img_features = img_features.view(img_features.size(0), -1)
        # img_features = img_features.unsqueeze(1)

        # choose action with horizon
        combined_input = torch.cat([state, action, img], dim=-1)
        y_pred = self.dynamics(combined_input)
        # y_pred = torch.sigmoid(y_pred)
        return y_pred

    def loss(self, y, y_pred):
        # bce_loss = F.binary_cross_entropy(y_pred, y)
        mse_loss = F.mse_loss(y_pred, y)
        return mse_loss

    def compute_loss(self, hidden, action, img, target):
        # Forward pass
        pred = self.forward(hidden, action, img)
        # Calculate the loss
        loss = self.loss(target, pred)
        return loss
    
class Binary_predictor_resnet_taxel_pos(nn.Module):
    def __init__(self, mlp_config, taxel_id = -1):
        super(Binary_predictor_resnet_taxel_pos, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = 14 + 1 + mlp_config.action_size + 512 + 3
        self.output_size = 1
        self.hidden_size = mlp_config.binary_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )


        # self.dynamics = nn.Sequential(
        #         nn.Linear(self.input_size, 256),
        #         activation,
        #         nn.Linear(256, 64),
        #         activation,
        #         nn.Linear(64, 1)
        #     )

    def forward(self, state, action, img_features, pos):
        img_features = img_features.repeat(state.shape[0], 1)
        img_features = img_features.unsqueeze(1)

        pos = pos.repeat(state.shape[0], 1)
        pos = pos.unsqueeze(1)

        # choose action with horizon
        combined_input = torch.cat([state, action, img_features, pos], dim=-1)
        y_pred = self.dynamics(combined_input)
        y_pred = torch.sigmoid(y_pred)
        return y_pred

    def loss(self, y, y_pred):
        bce_loss = F.binary_cross_entropy(y_pred, y)
        return bce_loss

    def compute_loss(self, hidden, action, img_features, pos, target):
        # Forward pass
        pred = self.forward(hidden, action, img_features, pos)
        # Calculate the loss
        loss = self.loss(target, pred)
        return loss
    
class Binary_predictor_resnet_taxel_pos_neighboring(nn.Module):
    def __init__(self, mlp_config, taxel_id = -1):
        super(Binary_predictor_resnet_taxel_pos_neighboring, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = 14 + 3 + mlp_config.action_size + 512 + 3
        self.output_size = 1
        self.hidden_size = mlp_config.binary_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )


        # self.dynamics = nn.Sequential(
        #         nn.Linear(self.input_size, 256),
        #         activation,
        #         nn.Linear(256, 64),
        #         activation,
        #         nn.Linear(64, 1)
        #     )

    def forward(self, state, action, img_features, pos):
        img_features = img_features.repeat(state.shape[0], 1)
        img_features = img_features.unsqueeze(1)

        pos = pos.repeat(state.shape[0], 1)
        pos = pos.unsqueeze(1)

        # choose action with horizon
        combined_input = torch.cat([state, action, img_features, pos], dim=-1)
        y_pred = self.dynamics(combined_input)
        y_pred = torch.sigmoid(y_pred)
        return y_pred

    def loss(self, y, y_pred):
        bce_loss = F.binary_cross_entropy(y_pred, y)
        return bce_loss

    def compute_loss(self, hidden, action, img_features, pos, target):
        # Forward pass
        pred = self.forward(hidden, action, img_features, pos)
        # Calculate the loss
        loss = self.loss(target, pred)
        return loss
    
class Reg_predictor_resnet_taxel_case3(nn.Module):
    def __init__(self, mlp_config, link_id = -1):
        super(Reg_predictor_resnet_taxel_case3, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = 14 + 1 + mlp_config.action_size + 512 + 56
        self.output_size = 1
        self.hidden_size = mlp_config.binary_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )

        # weights = ResNet50_Weights.DEFAULT
        # self.resnet = resnet50(weights=weights)
        # self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        # self.resnet.eval()

        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # self.preprocess = weights.transforms()
        # torch.backends.cudnn.benchmark = True

        # self.dynamics = nn.Sequential(
        #         nn.Linear(self.input_size, 256),
        #         activation,
        #         nn.Linear(256, 64),
        #         activation,
        #         nn.Linear(64, 1)
        #     )

    def forward(self, state, action, img, one_hot):
        img = img.repeat(state.shape[0], 1)
        img = img.unsqueeze(1)
        # one_hot = one_hot.repeat(state.shape[0], 1)
        # one_hot = one_hot.unsqueeze(1)
        # imgs = img.squeeze(1)
        # imgs = imgs.permute(0, 3, 1, 2)
        # imgs = torch.stack([self.preprocess(img) for img in imgs])
        # img_features = self.resnet(imgs)
        # img_features = img_features.view(img_features.size(0), -1)
        # img_features = img_features.unsqueeze(1)

        # choose action with horizon
        combined_input = torch.cat([state, action, img, one_hot], dim=-1)
        y_pred = self.dynamics(combined_input)
        # y_pred = torch.sigmoid(y_pred)
        return y_pred

    def loss(self, y, y_pred):
        # bce_loss = F.binary_cross_entropy(y_pred, y)
        mse_loss = F.mse_loss(y_pred, y)
        return mse_loss

    def compute_loss(self, hidden, action, img, target, one_hot):
        # Forward pass
        pred = self.forward(hidden, action, img, one_hot)
        # Calculate the loss
        loss = self.loss(target, pred)
        return loss
    
class Reg_predictor_resnet_taxel_case2(nn.Module):
    def __init__(self, mlp_config):
        super(Reg_predictor_resnet_taxel_case2, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = 14 + 56 + mlp_config.action_size + 512
        self.output_size = 56
        self.hidden_size = mlp_config.binary_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )

        # weights = ResNet50_Weights.DEFAULT
        # self.resnet = resnet50(weights=weights)
        # self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        # self.resnet.eval()

        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # self.preprocess = weights.transforms()
        # torch.backends.cudnn.benchmark = True

        # self.dynamics = nn.Sequential(
        #         nn.Linear(self.input_size, 256),
        #         activation,
        #         nn.Linear(256, 64),
        #         activation,
        #         nn.Linear(64, 1)
        #     )

    def forward(self, state, action, img):
        img = img.repeat(state.shape[0], 1)
        img = img.unsqueeze(1)
        # one_hot = one_hot.repeat(state.shape[0], 1)
        # one_hot = one_hot.unsqueeze(1)
        # imgs = img.squeeze(1)
        # imgs = imgs.permute(0, 3, 1, 2)
        # imgs = torch.stack([self.preprocess(img) for img in imgs])
        # img_features = self.resnet(imgs)
        # img_features = img_features.view(img_features.size(0), -1)
        # img_features = img_features.unsqueeze(1)

        # choose action with horizon
        combined_input = torch.cat([state, action, img], dim=-1)
        y_pred = self.dynamics(combined_input)
        # y_pred = torch.sigmoid(y_pred)
        return y_pred

    def loss(self, y, y_pred):
        # bce_loss = F.binary_cross_entropy(y_pred, y)
        mse_loss = F.mse_loss(y_pred, y)
        return mse_loss

    def compute_loss(self, hidden, action, img, target):
        # Forward pass
        pred = self.forward(hidden, action, img)
        # Calculate the loss
        loss = self.loss(target, pred)
        return loss
    
class WM_predictor_force(nn.Module):
    def __init__(self, mlp_config, num_mod):
        super(WM_predictor_force, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = mlp_config.hidden_size * num_mod
        self.output_size = 56
        self.hidden_size = mlp_config.hidden_size * num_mod
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )


    def forward(self, state):
        y_pred = self.dynamics(state)
        return y_pred

    def loss(self, y, y_pred):
        mse_loss = F.mse_loss(y_pred, y)
        return mse_loss

    def compute_loss(self, hidden, target):
        # Forward pass
        pred = self.forward(hidden)
        # Calculate the loss
        loss = self.loss(target, pred)
        return loss
    
class WM_predictor_force_baseline(nn.Module):
    def __init__(self, mlp_config):
        super(WM_predictor_force_baseline, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = mlp_config.hidden_size * 2
        self.output_size = 56
        self.hidden_size = mlp_config.hidden_size * 2
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )


    def forward(self, state):
        y_pred = self.dynamics(state)
        return y_pred

    def loss(self, y, y_pred):
        mse_loss = F.mse_loss(y_pred, y)
        return mse_loss

    def compute_loss(self, hidden, target):
        # Forward pass
        pred = self.forward(hidden)
        # Calculate the loss
        loss = self.loss(target, pred)
        return loss
    

class WM_encoder(nn.Module):
    def __init__(self, mlp_config: NetworkConfig):
        super(WM_encoder, self).__init__()
        assert mlp_config.input_size > 0, "Input size must be greater than 0."

        self.input_size = mlp_config.input_size
        self.hidden_size = mlp_config.hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Encoder for scalar inputs
        self.scalar_encoder = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.coordconv_encoder = nn.Sequential(
            CoordConv1d(1, 16, 1, with_r=True),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 56, self.hidden_size),
            nn.ReLU(),
        )

        # Encoder for image inputs
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, self.hidden_size),  # 假设输入图像大小为 64x64
            nn.ReLU(),
        )

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, self.hidden_size),  # 假设输入图像大小为 64x64
            nn.ReLU(),
        )

    def forward(self, scalar_input, image_input):
        force = scalar_input[:, 14:].unsqueeze(1)
        pos = scalar_input[:, :14]
        # Encode scalar input
        pos_encoded = self.scalar_encoder(pos)
        force_encoded = self.coordconv_encoder(force)

        # Encode image input
        image_encoded = self.image_encoder(image_input)
        # image_encoded = image_encoded.unsqueeze(1)

        # Encode depth input
        # depth_encoded = self.depth_encoder(depth_input)

        # Concatenate the encoded scalar and image features
        combined_encoded = torch.cat((pos_encoded, force_encoded, image_encoded), dim=-1)

        return combined_encoded
    
class WM_encoder_rgb(nn.Module):
    def __init__(self, mlp_config: NetworkConfig):
        super(WM_encoder_rgb, self).__init__()
        assert mlp_config.input_size > 0, "Input size must be greater than 0."

        self.input_size = mlp_config.input_size
        self.hidden_size = mlp_config.hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Encoder for scalar inputs
        self.scalar_encoder = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.coordconv_encoder = nn.Sequential(
            CoordConv1d(1, 16, 1, with_r=True),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 56, self.hidden_size),
            nn.ReLU(),
        )

        # Encoder for image inputs
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, self.hidden_size),  # 假设输入图像大小为 64x64
            nn.ReLU(),
        )

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, self.hidden_size),  # 假设输入图像大小为 64x64
            nn.ReLU(),
        )

    def forward(self, image_input):
        # force = scalar_input[:, 14:].unsqueeze(1)
        # pos = scalar_input[:, :14]
        # Encode scalar input
        # pos_encoded = self.scalar_encoder(pos)
        # force_encoded = self.coordconv_encoder(force)

        # Encode image input
        image_encoded = self.image_encoder(image_input)
        # image_encoded = image_encoded.unsqueeze(1)

        # Encode depth input
        # depth_encoded = self.depth_encoder(depth_input)

        # Concatenate the encoded scalar and image features
        # combined_encoded = torch.cat((pos_encoded, force_encoded, image_encoded), dim=-1)

        return image_encoded
    
class WM_encoder_rgbt(nn.Module):
    def __init__(self, mlp_config: NetworkConfig):
        super(WM_encoder_rgbt, self).__init__()
        assert mlp_config.input_size > 0, "Input size must be greater than 0."

        self.input_size = mlp_config.input_size
        self.hidden_size = mlp_config.hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Encoder for scalar inputs
        self.scalar_encoder = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.coordconv_encoder = nn.Sequential(
            CoordConv1d(1, 16, 1, with_r=True),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 56, self.hidden_size),
            nn.ReLU(),
        )

        # Encoder for image inputs
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, self.hidden_size),  # 假设输入图像大小为 64x64
            nn.ReLU(),
        )

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, self.hidden_size),  # 假设输入图像大小为 64x64
            nn.ReLU(),
        )

    def forward(self, scalar_input, image_input):
        force = scalar_input[:, 14:].unsqueeze(1)
        # pos = scalar_input[:, :14]
        # Encode scalar input
        # pos_encoded = self.scalar_encoder(pos)
        force_encoded = self.coordconv_encoder(force)

        # Encode image input
        image_encoded = self.image_encoder(image_input)
        # image_encoded = image_encoded.unsqueeze(1)

        # Encode depth input
        # depth_encoded = self.depth_encoder(depth_input)

        # Concatenate the encoded scalar and image features
        combined_encoded = torch.cat((force_encoded, image_encoded), dim=-1)

        return combined_encoded
    
class WM_encoder_rgbtp(nn.Module):
    def __init__(self, mlp_config: NetworkConfig):
        super(WM_encoder_rgbtp, self).__init__()
        assert mlp_config.input_size > 0, "Input size must be greater than 0."

        self.input_size = mlp_config.input_size
        self.hidden_size = mlp_config.hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Encoder for scalar inputs
        self.scalar_encoder = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.coordconv_encoder = nn.Sequential(
            CoordConv1d(1, 16, 1, with_r=True),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 56, self.hidden_size),
            nn.ReLU(),
        )

        # Encoder for image inputs
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, self.hidden_size),  # 假设输入图像大小为 64x64
            nn.ReLU(),
        )

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, self.hidden_size),  # 假设输入图像大小为 64x64
            nn.ReLU(),
        )

    def forward(self, scalar_input, image_input):
        force = scalar_input[:, 14:].unsqueeze(1)
        pos = scalar_input[:, :14]
        # Encode scalar input
        pos_encoded = self.scalar_encoder(pos)
        force_encoded = self.coordconv_encoder(force)

        # Encode image input
        image_encoded = self.image_encoder(image_input)
        # image_encoded = image_encoded.unsqueeze(1)

        # Encode depth input
        # depth_encoded = self.depth_encoder(depth_input)

        # Concatenate the encoded scalar and image features
        combined_encoded = torch.cat((pos_encoded, force_encoded, image_encoded), dim=-1)

        return combined_encoded
    
class WM_encoder_rgbtpd(nn.Module):
    def __init__(self, mlp_config: NetworkConfig):
        super(WM_encoder_rgbtpd, self).__init__()
        assert mlp_config.input_size > 0, "Input size must be greater than 0."

        self.input_size = mlp_config.input_size
        self.hidden_size = mlp_config.hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Encoder for scalar inputs
        self.scalar_encoder = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.coordconv_encoder = nn.Sequential(
            CoordConv1d(1, 16, 1, with_r=True),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 56, self.hidden_size),
            nn.ReLU(),
        )

        # Encoder for image inputs
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, self.hidden_size),  # 假设输入图像大小为 64x64
            nn.ReLU(),
        )

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, self.hidden_size),  # 假设输入图像大小为 64x64
            nn.ReLU(),
        )

    def forward(self, scalar_input, image_input, depth_input):
        force = scalar_input[:, 14:].unsqueeze(1)
        pos = scalar_input[:, :14]
        # Encode scalar input
        pos_encoded = self.scalar_encoder(pos)
        force_encoded = self.coordconv_encoder(force)

        # Encode image input
        image_encoded = self.image_encoder(image_input)
        # image_encoded = image_encoded.unsqueeze(1)

        # Encode depth input
        depth_encoded = self.depth_encoder(depth_input)

        # Concatenate the encoded scalar and image features
        combined_encoded = torch.cat((pos_encoded, force_encoded, image_encoded, depth_encoded), dim=-1)

        return combined_encoded
    
class WM_encoder_vit_clip(nn.Module):
    def __init__(self, mlp_config: NetworkConfig):
        super(WM_encoder_vit_clip, self).__init__()
        assert mlp_config.input_size > 0, "Input size must be greater than 0."

        self.input_size = mlp_config.input_size
        self.hidden_size = mlp_config.hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Encoder for scalar inputs
        self.scalar_encoder = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

        # Encoder for scalar inputs
        self.coordconv_encoder = nn.Sequential(
            CoordConv1d(1, 16, 1, with_r=True),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 56, self.hidden_size),
            nn.ReLU(),
        )

        self.rgb_linear = nn.Linear(512, self.hidden_size)
        self.depth_linear = nn.Linear(512, self.hidden_size)

    def preprocess_image(self, batch_tensor, target_size=(224, 224)):

        batch_tensor_resized = F.interpolate(batch_tensor, size=target_size, mode='bilinear', align_corners=False)

        mean = torch.tensor([0.48145466, 0.4578275 , 0.40821073]).view(1, 3, 1, 1).to(batch_tensor.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(batch_tensor.device)
        
        batch_tensor_resized = batch_tensor_resized / 255.0  # 假设输入张量的像素值范围是 [0, 255]

        batch_tensor_normalized = (batch_tensor_resized - mean) / std
        
        return batch_tensor_normalized

    def forward(self, scalar_input, image_input, depth_input):
        force = scalar_input[:, 14:].unsqueeze(1)
        pos = scalar_input[:, :14]
        
        # Encode scalar input
        pos_encoded = self.scalar_encoder(pos)
        force_encoded = self.coordconv_encoder(force)

        # Preprocess and encode image input using CLIP
        image_input = self.preprocess_image(image_input)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
        
        # Normalize image features to unit length
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_encoded = self.rgb_linear(image_features.float())

        depth_input = depth_input.repeat(1, 3, 1, 1)
        depth_input = self.preprocess_image(depth_input)
        with torch.no_grad(): 
            dapth_features = self.clip_model.encode_image(depth_input)
        
        # Normalize image features to unit length
        dapth_features /= dapth_features.norm(dim=-1, keepdim=True)
        depth_encoded = self.rgb_linear(dapth_features.float())

        # Concatenate the encoded scalar and image features
        combined_encoded = torch.cat((pos_encoded, force_encoded, image_encoded.squeeze(0), depth_encoded.squeeze(0)), dim=-1)

        return combined_encoded
    
class WM_encoder_baseline(nn.Module):
    def __init__(self, mlp_config: NetworkConfig):
        super(WM_encoder_baseline, self).__init__()
        assert mlp_config.input_size > 0, "Input size must be greater than 0."

        mlp_config.hidden_size = mlp_config.hidden_size

        self.input_size = mlp_config.input_size
        self.hidden_size = mlp_config.hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout

        # Define the activation function
        if self.activation == "relu":
            activation = nn.ReLU()
        elif self.activation == "tanh":
            activation = nn.Tanh()
        elif self.activation == "mish":
            activation = nn.Mish()
        elif self.activation == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Encoder for scalar inputs
        self.scalar_encoder = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.coordconv_encoder = nn.Sequential(
            CoordConv1d(1, 16, 1, with_r=True),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 56, self.hidden_size),
            nn.ReLU(),
        )

    def forward(self, scalar_input):
        force = scalar_input[:, 14:].unsqueeze(1)
        pos = scalar_input[:, :14]
        # Encode scalar input
        pos_encoded = self.scalar_encoder(pos)
        force_encoded = self.coordconv_encoder(force)
        
        combined_encoded = torch.cat((pos_encoded, force_encoded), dim=-1)

        return combined_encoded

    
    
class WM_dynamics(nn.Module):
    def __init__(self, mlp_config: NetworkConfig, num_mod):
        super(WM_dynamics, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = mlp_config.hidden_size * 4 + mlp_config.action_size
        self.output_size = mlp_config.hidden_size * 4
        self.hidden_size = mlp_config.Dyn_hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout


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
        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )

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
    
    def compute_loss(self, hidden, action, target):
        # Forward pass
        pred = self.forward(hidden, action)

        # Calculate the loss
        loss = self.loss(pred, target)

        return loss
    
class WM_LSTMDynamics(nn.Module):
    def __init__(self, mlp_config: NetworkConfig, num_mod, num_layers=4, dropout=0.1):
        super(WM_LSTMDynamics, self).__init__()

        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = mlp_config.hidden_size * num_mod  # 拼接的输入维度
        self.horizon = mlp_config.horizon  # 输出序列的长度 H
        self.output_size = mlp_config.hidden_size * num_mod  # 输出的潜在状态维度
        self.hidden_size = mlp_config.Dyn_hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout
        self.action_size = mlp_config.action_size // self.horizon

        # Activation function
        if self.activation == "relu":
            self.activation_fn = nn.ReLU()
        elif self.activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif self.activation == "mish":
            self.activation_fn = nn.Mish()
        elif self.activation == "silu":
            self.activation_fn = nn.SiLU()

        # Embedding layers
        self.embedding = nn.Linear(self.input_size, self.hidden_size)  # 输入状态嵌入层
        self.action_embedding = nn.Linear(self.action_size, self.hidden_size)  # 单步动作嵌入层

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.hidden_size * 2,  # 状态和动作嵌入拼接后的大小
            hidden_size=self.hidden_size,   # LSTM 隐藏层大小
            num_layers=num_layers,         # LSTM 层数
            batch_first=True,              # batch 在第一个维度
            dropout=dropout
        )

        # 输出层，预测从 t 到 t+H 的所有潜在状态
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, hidden, action_seq):
        # hidden: (batch_size, hidden_size)
        # action_seq: (batch_size, action_size) 
        action_ori = action_seq.reshape(-1, self.horizon, self.action_size)

        # Step 1: 状态嵌入
        s_t = self.embedding(hidden)  # (batch_size, hidden_size)

        # Step 2: 动作嵌入
        actions = self.action_embedding(action_ori)  # (batch_size, horizon, hidden_size)

        # Step 3: 将初始状态扩展到 horizon 时间步
        s_t = s_t.unsqueeze(1).repeat(1, self.horizon, 1)  # (batch_size, horizon, hidden_size)

        # Step 4: 将状态和动作拼接
        x = torch.cat([s_t, actions], dim=-1)  # (batch_size, horizon, hidden_size * 2)

        # Step 5: 使用 LSTM 处理
        lstm_output, _ = self.lstm(x)  # lstm_output: (batch_size, horizon, hidden_size)

        # Step 6: 通过输出层预测
        output = self.output_layer(lstm_output)  # (batch_size, horizon, output_size)

        return output

    def loss(self, y, y_pred):
        # MSE loss
        mse = F.mse_loss(y, y_pred)
        return mse

    def compute_loss(self, hidden, action_seq, target):
        # Forward pass
        pred = self.forward(hidden, action_seq)

        # Calculate the loss
        loss = self.loss(pred, target)

        return loss

    
class WM_TransformerDynamics(nn.Module):
    def __init__(self, mlp_config: NetworkConfig, num_mod, num_layers=4, num_heads=4, dropout=0.1):
        super(WM_TransformerDynamics, self).__init__()

        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = mlp_config.hidden_size * num_mod  # 拼接的输入维度
        self.horizon = mlp_config.horizon  # 输出序列的长度 H
        self.output_size = mlp_config.hidden_size * num_mod  # 输出的潜在状态维度
        self.hidden_size = mlp_config.Dyn_hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout
        self.action_size = mlp_config.action_size // self.horizon
        

        # Activation function
        if self.activation == "relu":
            self.activation_fn = nn.ReLU()
        elif self.activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif self.activation == "mish":
            self.activation_fn = nn.Mish()
        elif self.activation == "silu":
            self.activation_fn = nn.SiLU()

        # Transformer Layer (encoder-decoder)
        self.embedding = nn.Linear(self.input_size, self.hidden_size)  # 输入状态嵌入层
        self.action_embedding = nn.Linear(self.action_size, self.hidden_size)  # 单步动作嵌入层

        self.transformer = nn.Transformer(
            d_model=self.hidden_size * 2,      # 每个时间步的特征维度
            nhead=num_heads,               # 注意力头的数量
            num_encoder_layers=num_layers,  # 编码器层数
            num_decoder_layers=num_layers,  # 解码器层数
            dim_feedforward=self.hidden_size * 8,  # 前馈神经网络的维度
            dropout=dropout,               # Dropout
        )

        # 输出层，预测从 t 到 t+H 的所有潜在状态
        self.output_layer = nn.Linear(self.hidden_size * 2, self.output_size)

    def position_encoding(self, pos):
        """
        生成简单的正弦余弦位置编码
        """
        pe = torch.zeros(pos.size(0), pos.size(1), self.hidden_size * 2).to(pos.device)
        for i in range(self.hidden_size):
            if i % 2 == 0:
                pe[:, :, i] = torch.sin(pos.float() / (10000 ** (i / self.hidden_size)))
            else:
                pe[:, :, i] = torch.cos(pos.float() / (10000 ** ((i - 1) / self.hidden_size)))
        return pe

    def forward(self, hidden, action_seq):
        # hidden: (batch_size, hidden_size)
        # action_seq: (batch_size, action_size) 
        action_ori = action_seq.reshape(-1, self.horizon, self.action_size)

        # Step 1: 状态嵌入
        s_t = self.embedding(hidden)  # (batch_size, hidden_size)
        
        # Step 2: 动作嵌入
        actions = self.action_embedding(action_ori)  # (batch_size, horizon, hidden_size)

        # Step 3: 添加位置编码
        pos = torch.arange(0, self.horizon).unsqueeze(0).repeat(s_t.size(0), 1).to(s_t.device)
        pos_embedding = self.position_encoding(pos)

        # Step 4: 将初始状态扩展到horizon时间步
        s_t = s_t.unsqueeze(1).repeat(1, self.horizon, 1)  # (batch_size, horizon, hidden_size)

        # Step 5: 将状态和动作拼接
        x = torch.cat([s_t, actions], dim=-1)  # (batch_size, horizon, hidden_size * 2)

        # Step 6: 添加位置编码
        x = x + pos_embedding  # (batch_size, horizon, hidden_size * 2)

        # Step 7: 转换形状为 (horizon, batch_size, hidden_size * 2)
        x = x.permute(1, 0, 2)  # (horizon, batch_size, hidden_size * 2)

        # Step 8: 使用 Transformer 处理
        x = x.double()
        transformer_output = self.transformer(x, x)  # 编码器和解码器输入相同

        # Step 9: 通过输出层预测
        output = self.output_layer(transformer_output)  # (horizon, batch_size, output_size)
        output = output.permute(1, 0, 2)  # (batch_size, horizon, output_size)

        # # Step 1: 将当前潜在状态和动作序列连接
        # input_seq = torch.cat([hidden, action_seq], dim=-1) # (batch_size, input_size)
        
        # # Step 2: 输入嵌入层
        # embedded_input = self.embedding(input_seq)  # (batch_size, 1, hidden_size)
        
        # # Step 3: 使用 Transformer 进行编码和解码
        # # transformer 输入需要是 (horizon, batch_size, hidden_size) 形式
        # embedded_input = embedded_input.unsqueeze(0).repeat(self.horizon, 1, 1)

        # # 由于我们的任务是一次性预测多个时刻的潜在状态，解码器的输入与编码器相同
        # transformer_output = self.transformer(embedded_input, embedded_input)  # (horizon, batch_size, hidden_size)

        # # Step 4: 从 transformer 的输出中获取潜在状态
        # output = self.output_layer(transformer_output)  # (horizon, batch_size, output_size)
        # output = output.permute(1, 0, 2)  # (batch_size, horizon, output_size)

        return output

    def loss(self, y, y_pred):
        # MSE loss
        mse = F.mse_loss(y, y_pred)
        return mse

    def compute_loss(self, hidden, action_seq, target):
        # Forward pass
        pred = self.forward(hidden, action_seq)

        # Calculate the loss
        loss = self.loss(pred, target)

        return loss
    
class WM_TransformerDynamics_baseline(nn.Module):
    def __init__(self, mlp_config: NetworkConfig, num_layers=4, num_heads=4, dropout=0.1):
        super(WM_TransformerDynamics_baseline, self).__init__()

        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = mlp_config.hidden_size * 2  # 拼接的输入维度
        self.horizon = mlp_config.horizon  # 输出序列的长度 H
        self.output_size = mlp_config.hidden_size * 2  # 输出的潜在状态维度
        self.hidden_size = mlp_config.Dyn_hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout
        self.action_size = mlp_config.action_size // self.horizon
        

        # Activation function
        if self.activation == "relu":
            self.activation_fn = nn.ReLU()
        elif self.activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif self.activation == "mish":
            self.activation_fn = nn.Mish()
        elif self.activation == "silu":
            self.activation_fn = nn.SiLU()

        # Transformer Layer (encoder-decoder)
        self.embedding = nn.Linear(self.input_size, self.hidden_size)  # 输入状态嵌入层
        self.action_embedding = nn.Linear(self.action_size, self.hidden_size)  # 单步动作嵌入层

        self.transformer = nn.Transformer(
            d_model=self.hidden_size * 2,      # 每个时间步的特征维度
            nhead=num_heads,               # 注意力头的数量
            num_encoder_layers=num_layers,  # 编码器层数
            num_decoder_layers=num_layers,  # 解码器层数
            dim_feedforward=self.hidden_size * 8,  # 前馈神经网络的维度
            dropout=dropout,               # Dropout
        )

        # 输出层，预测从 t 到 t+H 的所有潜在状态
        self.output_layer = nn.Linear(self.hidden_size * 2, self.output_size)

    def position_encoding(self, pos):
        """
        生成简单的正弦余弦位置编码
        """
        pe = torch.zeros(pos.size(0), pos.size(1), self.hidden_size * 2).to(pos.device)
        for i in range(self.hidden_size):
            if i % 2 == 0:
                pe[:, :, i] = torch.sin(pos.float() / (10000 ** (i / self.hidden_size)))
            else:
                pe[:, :, i] = torch.cos(pos.float() / (10000 ** ((i - 1) / self.hidden_size)))
        return pe

    def forward(self, hidden, action_seq):
        # hidden: (batch_size, hidden_size)
        # action_seq: (batch_size, action_size)
        action_ori = action_seq
        # action_ori = action_seq.reshape(-1, self.horizon, self.action_size)

        # Step 1: 状态嵌入
        s_t = self.embedding(hidden)  # (batch_size, hidden_size)
        if s_t.shape[0] == 1:
            s_t = s_t.repeat(action_ori.shape[0], 1)
        # Step 2: 动作嵌入
        actions = self.action_embedding(action_ori)  # (batch_size, horizon, hidden_size)

        # Step 3: 添加位置编码
        pos = torch.arange(0, self.horizon).unsqueeze(0).repeat(s_t.size(0), 1).to(s_t.device)
        pos_embedding = self.position_encoding(pos)

        # Step 4: 将初始状态扩展到horizon时间步
        s_t = s_t.unsqueeze(1).repeat(1, self.horizon, 1)  # (batch_size, horizon, hidden_size)

        # Step 5: 将状态和动作拼接
        x = torch.cat([s_t, actions], dim=-1)  # (batch_size, horizon, hidden_size * 2)

        # Step 6: 添加位置编码
        x = x + pos_embedding  # (batch_size, horizon, hidden_size * 2)

        # Step 7: 转换形状为 (horizon, batch_size, hidden_size * 2)
        x = x.permute(1, 0, 2)  # (horizon, batch_size, hidden_size * 2)

        # Step 8: 使用 Transformer 处理
        transformer_output = self.transformer(x, x)  # 编码器和解码器输入相同

        # Step 9: 通过输出层预测
        output = self.output_layer(transformer_output)  # (horizon, batch_size, output_size)
        output = output.permute(1, 0, 2)  # (batch_size, horizon, output_size)

        # # Step 1: 将当前潜在状态和动作序列连接
        # input_seq = torch.cat([hidden, action_seq], dim=-1) # (batch_size, input_size)
        
        # # Step 2: 输入嵌入层
        # embedded_input = self.embedding(input_seq)  # (batch_size, 1, hidden_size)
        
        # # Step 3: 使用 Transformer 进行编码和解码
        # # transformer 输入需要是 (horizon, batch_size, hidden_size) 形式
        # embedded_input = embedded_input.unsqueeze(0).repeat(self.horizon, 1, 1)

        # # 由于我们的任务是一次性预测多个时刻的潜在状态，解码器的输入与编码器相同
        # transformer_output = self.transformer(embedded_input, embedded_input)  # (horizon, batch_size, hidden_size)

        # # Step 4: 从 transformer 的输出中获取潜在状态
        # output = self.output_layer(transformer_output)  # (horizon, batch_size, output_size)
        # output = output.permute(1, 0, 2)  # (batch_size, horizon, output_size)

        return output

    def loss(self, y, y_pred):
        # MSE loss
        mse = F.mse_loss(y, y_pred)
        return mse

    def compute_loss(self, hidden, action_seq, target):
        # Forward pass
        pred = self.forward(hidden, action_seq)

        # Calculate the loss
        loss = self.loss(pred, target)

        return loss
    
class WM_dynamics_baseline(nn.Module):
    def __init__(self, mlp_config: NetworkConfig):
        super(WM_dynamics_baseline, self).__init__()
        assert mlp_config.action_size > 0, "Action size must be greater than 0."

        self.input_size = mlp_config.hidden_size + mlp_config.action_size
        self.output_size = mlp_config.hidden_size
        self.hidden_size = mlp_config.Dyn_hidden_size
        self.activation = mlp_config.activation
        self.dropout = mlp_config.dropout


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
        self.dynamics = nn.Sequential(
            NormedLinear(self.input_size, self.hidden_size, dropout=self.dropout, act=activation),
            NormedLinear(self.hidden_size, self.hidden_size, dropout=self.dropout, act=activation),
            nn.Linear(self.hidden_size, self.output_size),
        )

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
    
    def compute_loss(self, hidden, action, target):
        # Forward pass
        pred = self.forward(hidden, action)

        # Calculate the loss
        loss = self.loss(pred, target)

        return loss