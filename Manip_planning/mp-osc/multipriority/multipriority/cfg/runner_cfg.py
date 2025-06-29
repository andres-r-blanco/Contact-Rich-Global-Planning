from dataclasses import dataclass, field

@dataclass
class RunnerConfig:
    """Configuration for running MP-OSC experiments."""

    sim: bool = False
    """Whether to run in simulation mode."""

    enable_trainer: bool = False
    """Whether to enable the trainer."""

    is_train: bool = False
    """Whether to run training or inference."""

    reference_trajectory: str = None
    """The reference trajectory file name used for tracking."""

    priority_policy: str = "fixed"
    """The priority policy to use, either 'cb', 'fixed', or 'heuristic'."""

    skeleton_initial_priority: dict = None
    """Skeleton initial priority configuration."""

    skeleton_contact_params: str = None
    """The skeleton contact parameters file name."""

    tracker_lookahead: float = 0.04
    """The lookahead distance for the tracker."""

    trainer_env_cfg: dict = None

    task_controller_cfg: dict = None
    """Task controller configuration."""

    # TODO: Add this param to env config
    # robot_urdf: str = None
    # priority_fn_config: PriorityConfig = field(default_factory=CBConfig)