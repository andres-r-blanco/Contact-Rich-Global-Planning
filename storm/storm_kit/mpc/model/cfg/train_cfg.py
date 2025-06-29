from dataclasses import dataclass, field

from .network_cfg import NetworkConfig, RNNConfig, MLPConfig, AEConfig


@dataclass
class TrainConfig:
    """Configuration for training."""
    seed: int = 88
    """The random seed for reproducibility."""

    root_dir: str = ""  # set automatically
    """The root directory for the project."""

    log_dir: str = "logs"
    """The directory where the logs are stored."""

    data_dir: str = "/home/angchen/xac/data/world_model/test_data_h5_2"
    """The directory where the training data is stored."""

    train_separate: bool = False
    """Whether to train autoencoder and dynamics model separately."""

    num_epochs: int = 50
    """The number of epochs to train the model for."""

    train_AE_epochs: int = 100
    """The number of epochs to train the autoencoder for."""

    train_Dyn_epochs: int = 100
    """The number of epochs to train the dynamics model for."""

    batch_size: int = 8
    """The number of samples in a batch."""

    data_size: int = 35
    """The number of samples in the dataset."""

    sequence_length: int = 1
    """The length of the sequence of input states (history states)."""

    horizon: int = 10
    """The number of steps to predict into the future."""

    recon_loss_enabled: bool = True
    """Whether to include the reconstruction loss in the training objective."""

    AE_type: str = "AE"
    """The type of autoencoder to use."""

    obs_types: tuple = (
        "joint_pos",
        "joint_vel",
        "skin",
    )

    train_split: float = 0.9
    """The fraction of the data to use for training."""

    normalization: bool = False
    """Whether to normalize the data."""

    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    """The configuration for the neural network, either RNNConfig or MLPConfig."""

    log_wandb: bool = True
    """Whether to log the training run to Weights & Biases."""
    
    lr: float = 1e-4
    """The learning rate for the joint optimizer."""

    lr_decay: float = 0.9999
    """The exponential learning rate decay for the joint optimizer."""

    temperature: float = 0.1
    """The temperature for sampling from the MDN. 0 results in greedy sampling."""

    lr_ae: float = 3e-4
    """The learning rate for the AE optimizer."""

    lr_decay_ae: float = 0.9999
    """The exponential learning rate decay for the AE optimizer."""

    lr_dyn: float = 1e-1
    """The learning rate for the Dynamics optimizer."""

    lr_decay_dyn: float = 0.9999
    """The exponential learning rate decay for the Dynamics optimizer."""

    train_taxel: bool = True

    link_name: str = "Link_5"

    filter_all_zeros: bool = False

    data_save_dir: str = "/home/angchen/xac/data/world_model/preprocessed_data"

    mod: str = 'rgbtp'

    enc: str = 'cnn'

    dyn: str = 'mlp'


