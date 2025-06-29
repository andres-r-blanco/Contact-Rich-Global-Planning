from dataclasses import dataclass, field

@dataclass
class EnvConfig:
    """Configuration for the environment."""

    executable_file: str = None

    robot_urdf: str = None

    taxel_cfg: str = None
    """Taxel configuration file name."""

    joint_state_topic: str = None
    """The joint state topic."""

    skin_topic: str = None
    """The skin topic."""

