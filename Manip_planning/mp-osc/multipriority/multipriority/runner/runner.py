import numpy as np
import rospy
import time

from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

from multipriority.envs.toy_env_sim import ToyOneEnv
from multipriority.envs.toy_env_ros import RealToyEnv
from multipriority.utils import Tracker, load_yaml
from multipriority.controllers import TaskController, ContactController, MultiPriorityController

class Runner:
    def __init__(self, sim=False, enable_trainer=False, is_train=False, reference_trajectory=None, **kwargs):
        if sim:
            # TODO: Find a better place for this first step
            self.env = ToyOneEnv(**kwargs); self.env._step()
            self.action_fn = self.sim_action_fn
            self.execute_fn = self.sim_execute_fn
            self.add_gravity = True
        else:
            self.env = RealToyEnv(**kwargs)
            self.action_fn = self.real_action_fn
            self.execute_fn = self.real_execute_fn
            self.add_gravity = False
            self.action_pub = rospy.Publisher('/joint_space_compliant_controller/command', JointTrajectoryPoint, queue_size=1)
        
        if enable_trainer:
            self.train_runner = Runner(sim=True, enable_trainer=False)
        
        if is_train:
            self.run = self.run_training
        else:
            self.run = self.run_inference
    
        self.preference_history = []
        self.priority_policy = kwargs.get("priority_policy", "fixed") # cb / fixed / heuristic
        if self.priority_policy == "cb":
            self.ranked_controllers = self.ranked_controllers_cb
        elif self.priority_policy == "fixed":
            self.ranked_controllers = self.ranked_controllers_fixed
        elif self.priority_policy == "heuristic":
            self.ranked_controllers = self.ranked_controllers_heuristic
        self.priority_fn = None
        self.reference_trajectory = reference_trajectory
        self.init_pos = None
        # TODO: Add this param to config
        self.tracker = Tracker(reference_trajectory, lookahead=kwargs.get("tracker_lookahead", 0.04))
        self.task_controller = TaskController(self.env.robot, kwargs.get("task_controller_cfg"))
        # initialize one contact controller for each taxel
        self.contact_controllers = [ContactController(taxel) for taxel in self.env.robot.taxels]
        # TODO: Update these params
        self.skeleton_contact_params = load_yaml(kwargs.get("skeleton_contact_params", 'skeleton_real_params.yaml'))
       
        skeleton_initial_priority = kwargs.get("skeleton_initial_priority", None)
        self.mp_controller = MultiPriorityController(self.skeleton_contact_params, skeleton_initial_priority, self.env.robot, self.task_controller)
        self.runner_config = kwargs
        
        self.num_skeleton_ids = 9
        self.last_time = time.time()
        
        # TODO: remove this
        self.goal_set = False
        self.goal_ori = None

    def __repr__(self):
        return str(self)
    
    def update_init_pose(self, joint_position):
        pass

    def update_preference_history(self, preference):
        pass

    def update_reference_trajectory(self, reference_trajectory):
        self.reference_trajectory = reference_trajectory
        # TODO: Use lookahead from config
        self.tracker = Tracker(reference_trajectory, lookahead=0.05)

    def update_priority_policy(self, priority_policy):
        pass

    def sim_action_fn(self):
        pass

    def sim_execute_fn(self, action):
        if np.isnan(action).any():
            print ("NAN action")
            self.env.step(np.zeros(7, dtype=np.float64))
        else:
            self.env.step(action)

    def real_action_fn(self):
        pass

    def update_active_taxels(self):
        forces = self.env.robot.taxel_forces
        if self.priority_policy != "heuristic":
            skeletonwise_taxels = [[] for _ in range(self.num_skeleton_ids + 1)]
            taxels = self.env.robot.taxels
            for i, txl in enumerate(taxels):
                # if i != 12:
                #     continue
                if taxels[i].contactF > self.env.robot.sensor_threshold:
                    # skeleton_id = self.env.get_skeleton_id(taxels[i])
                    # TODO: get_obs should update taxel skeleton ID
                    skeletonwise_taxels[taxels[i].skeleton_id].append(i)
                    self.env.robot.taxels[i].skeleton_params = self.skeleton_contact_params[f'b{taxels[i].skeleton_id}']

            # sort taxels in each skeleton ID by deviation from max force
            for i in range(1, self.num_skeleton_ids + 1):
                skeletonwise_taxels[i] = sorted(skeletonwise_taxels[i], key=lambda x: taxels[x].max_force - taxels[x].contactF)

            return skeletonwise_taxels
        else:
            active_taxels = [i for i, txl in enumerate(self.env.robot.taxels) if txl.contactF > self.env.robot.sensor_threshold]
            return active_taxels

    def real_execute_fn(self, action):
        # TODO: Clean up once working for pose and force control both
        # TODO: Add these params to config
        control_mode = self.runner_config.get("control_mode", "position")
        policy_control_period = self.runner_config.get("policy_control_period", 0.1)
        cmd = JointTrajectoryPoint()
        # TODO: Add action clipping
        # self.action_pub.publish(action)
        # TODO: if goal reached then set cmd_qdd to 0
        if self.task_controller.get_state_error()[0] < 0.02:
            action.fill(0)

        action = action.reshape(1, -1)
        if control_mode == 'velocity':
            act = action * policy_control_period + np.array(self.env.robot.current_joint_positions).reshape(1, -1)
        if control_mode == 'position':
            cmd_qd = action * policy_control_period + np.array(self.env.robot.current_joint_velocities).reshape(1, -1)
            curr_pos =  np.array(self.env.robot.current_joint_positions).reshape(-1)
            # move to 0 to 2pi
            init_val = curr_pos[-1,]
            curr_pos = (curr_pos + 2 * np.pi) % (2 * np.pi)
            cmd_q = cmd_qd * policy_control_period + curr_pos
            
            # move to 0 to 2pi again
            cmd_q = (cmd_q + 2 * np.pi) % (2 * np.pi)
            cmd_q = curr_pos + (cmd_q - curr_pos) * min(0.015, np.linalg.norm(cmd_q - curr_pos)) / np.linalg.norm(cmd_q - curr_pos)
            cmd_q = (cmd_q + 2 * np.pi) % (2 * np.pi)
            # finally move to -pi to pi
            cmd_q = np.array(cmd_q[0])
            cmd_q[cmd_q >= np.pi] -= 2 * np.pi
            cmd_q[cmd_q < -np.pi] += 2 * np.pi
            norm = np.linalg.norm(cmd_qd)
            if norm < 1e-3:
                norm = 1
            cmd_qd /= norm
            cmd_qd *= min(0.9, norm)
            out_q = cmd_q.tolist()
            if type(out_q[0]) == list:
                out_q = out_q[0]
            out_qd = cmd_qd.tolist()[0]
            cmd.positions = out_q
            cmd.velocities = out_qd

        if time.time() - self.last_time > policy_control_period:
            # print ("Publishing\n")
            self.action_pub.publish(cmd)
            self.last_time = time.time()

    def ranked_controllers_cb(self, active_taxels):
        pass

    def ranked_controllers_fixed(self, active_taxels):
        # TODO: Add feature to sort skeleton IDs by priority first
        # TODO: Remove temporary code
        priority = np.arange(len(active_taxels) + 1).tolist() # self.get_priority_fixed()
        priority[-1] = -1
        active_controllers = []
        for skid in priority:
            if skid == -1:
                active_controllers.append(self.task_controller)
            else:
                active_controllers.extend(self.contact_controllers[taxel] for taxel in active_taxels[skid])
        # TODO: There must be a better way to do this
        for i, controller in enumerate(active_controllers):
            if isinstance(controller, ContactController):
                if i < len(active_controllers) - 1:
                    controller.set_desired_force(null=False)
                else:
                    controller.set_desired_force(null=True)
        return active_controllers

    def ranked_controllers_heuristic(self, active_taxels):
        pass

    def run_inference(self):
        init_pos = [90, 293.64, 182, 230.15, 359.25, 332.14, 90.87]
        # init_pos = [10,10,10,10,10,30,0]
        # init_pos = [0] * 7
        for i in range(len(init_pos)):
            if init_pos[i] > 180:
                init_pos[i] -= 360

        # keep running while loop until ctrl+c
        for _ in range(100):
            self.env.robot.setJointPositionsDirectly(init_pos)
            obs = self.env.get_obs()
            self.env._step()

        try:
            while True:
                obs = self.env.get_obs()
                if obs is None:
                    continue
                
                if not self.goal_set:
                    # generate reference trajectory with current position
                    # such that the robot moves straight along the specified axis
                    # for 5 cm
                    current_pos = self.env.robot.current_ee_pos
                    move_axis = 'z'
                    if move_axis == 'x':
                        goal_pos = [current_pos[0] - 0.2, current_pos[1], current_pos[2]]
                    elif move_axis == 'y':
                        goal_pos = [current_pos[0], current_pos[1] - 0.2, current_pos[2]]
                    elif move_axis == 'z':
                        goal_pos = [current_pos[0], current_pos[1], current_pos[2] + 0.04]
                    self.goal_pos = goal_pos
                    self.goal_ori = obs['robot']['orientations'][self.env.robot.ee_idx]
                    self.reference_trajectory = [[np.array(goal_pos), obs['robot']['orientations'][self.env.robot.ee_idx]]]
                    self.tracker.trajectory = self.reference_trajectory
                    self.goal_set = True

                goal_pos, goal_ori = self.tracker.get_next_waypoint(self.env.robot.current_ee_pos)
                # # TODO: Move viz cube stuff to sim env
                # # self.env.viz_cube.setTransform(position=goal_pos)
                self.task_controller.set_goal_pose(goal_pos, self.goal_ori)

                active_taxels = self.update_active_taxels()
                self.mp_controller.active_controllers = self.ranked_controllers(active_taxels)
                print ("Active Controllers: ", self.mp_controller.active_controllers)
                action = self.mp_controller.update(obs, add_gravity=self.add_gravity)
                self.execute_fn(action)
                
        except KeyboardInterrupt:
            print("Shutting down...")

    def run_training(self):
        pass
