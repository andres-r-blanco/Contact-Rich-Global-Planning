from dataclasses import dataclass


@dataclass
class NetworkConfig:
    """Configuration for a neural network."""

    input_size: int = 0  # set automatically
    """The number of features in the network input (state+action)."""

    output_size: int = 0  # set automatically
    """The number of features in the network output (state)."""

    hidden_size: int = 512
    """The number of features in the hidden state h."""

    action_size: int = 0  # set automatically
    """The number of features in the network action (state)."""

    sequence_length: int = 0  # set automatically
    """The length of the input sequence."""

    dropout: float = 0.0
    """The dropout rate in the hidden layers."""

    simnorm_dim: int = 8
    """The number of features in the simplicial normalization."""

    AE_num_layers: int = 2
    """The number of hidden layers in the AE network."""

    Dyn_num_layers: int = 1
    """The number of hidden layers in the dynamics network."""

    activation: str = "relu"
    """The activation function to use in the hidden layers."""

    Dyn_hidden_size: int = 512
    """The number of features in the hidden layers of dynamics model."""

    enable_AE_simnorm: bool = False
    """Whether to enable simplicial normalization in the AE."""

    binary_size: int = 256

    horizon: int = 10


@dataclass
class RNNConfig(NetworkConfig):
    """Configuration for the MDN-RNN model."""

    num_gaussians: int = 5
    """The number of Gaussians in the MDN."""


@dataclass
class MLPConfig(NetworkConfig):
    """Configuration for the MLP model."""

    num_layers: int = 1
    """The number of hidden layers in the network."""

    activation: str = "mish"
    """The activation function to use in the hidden layers."""

@dataclass
class AEConfig(NetworkConfig):
    """Configuration for the Autoencoder model."""

    num_layers: int = 1
    """The number of hidden layers in the network."""

    activation: str = "mish"
    """The activation function to use in the hidden layers."""

@dataclass
class DYNConfig(NetworkConfig):
    """Configuration for the Autoencoder model."""

    num_layers: int = 1
    """The number of hidden layers in the network."""

    activation: str = "mish"
    """The activation function to use in the hidden layers."""

    mlp_hidden_size: int = 2048
    """The number of features in the hidden layers of dynamics model."""

    




    