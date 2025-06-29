import rospy
import yaml
import multipriority
from multipriority.runner import Runner
from multipriority.utils import load_yaml, load_trajectory
from multipriority.cfg.runner_cfg import RunnerConfig
from dataclasses import asdict

if __name__ == "__main__":
    rospy.init_node("mp_runner")
    exp_config = RunnerConfig(**(load_yaml("sim_exp_config.yaml")))
    # TODO: Load reference trajectory
    trajectory = load_trajectory("reference_trajectory_w_contacts.npz")
    runner = Runner(**asdict(exp_config))
    # runner.update_reference_trajectory(trajectory)
    runner.run()
    # TODO: Sim experiments with feedback will still require rospy.spin()
    rospy.spin()