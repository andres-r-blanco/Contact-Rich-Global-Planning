import pybullet as p
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.setGravity(0,0,-10)

# Define the half extents of the box
boxHalfLength = 1.0
boxHalfWidth = 1.0
boxHalfHeight = 1.0

# Create the collision shape for the box
colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])

# Define the position where you want to place the box
boxPosition = [2, 2, 1]

# Create the multi-body with the box collision shape and set mass to 0 to make it fixed
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=boxPosition)

planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("C:/Users/arbxe/OneDrive/Desktop/code stuff/EmPRISE repos/mp-osc/mp-osc/multipriority/urdfs/gen3_7dof_vision_with_skin_sim.urdf",cubeStartPos, cubeStartOrientation)
p.stepSimulation()
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
n = p.getNumJoints(robot)
print(cubePos,cubeOrn)
print("balls")


time.sleep(10)

p.disconnect()




import sys
sys.path.insert(1, r"C:\Users\arbxe\OneDrive\Desktop\code stuff\EmPRISE repos\mp-osc\mp-osc\pybullet-planning-master")
import pybullet as p
import pybullet_data
import numpy as np
import time
import random

# Constants
MAX_ITER = 1000
STEP_SIZE = 0.05
GOAL_THRESHOLD = 0.1

def create_box(boxPosition,box_dims):
    # Define the half extents of the box
    boxHalfLength = box_dims[0]
    boxHalfWidth = box_dims[1]
    boxHalfHeight = box_dims[2]

    # Create the collision shape for the box
    colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])

    # Create the multi-body with the box collision shape and set mass to 0 to make it fixed
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=boxPosition)
    return

from random import random
import time

from .utils import irange, argmin, RRT_ITERATIONS, apply_alpha, RED, INF, elapsed_time


class TreeNode(object):

    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

    #def retrace(self):
    #    if self.parent is None:
    #        return [self]
    #    return self.parent.retrace() + [self]

    def retrace(self):
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def clear(self):
        self.node_handle = None
        self.edge_handle = None

    def draw(self, env, color=apply_alpha(RED, alpha=0.5)):
        # https://github.mit.edu/caelan/lis-openrave
        from manipulation.primitives.display import draw_node, draw_edge
        self.node_handle = draw_node(env, self.config, color=color)
        if self.parent is not None:
            self.edge_handle = draw_edge(
                env, self.config, self.parent.config, color=color)

    def __str__(self):
        return 'TreeNode(' + str(self.config) + ')'
    __repr__ = __str__


def configs(nodes):
    if nodes is None:
        return None
    return list(map(lambda n: n.config, nodes))


def rrt(start, goal_sample, distance_fn, sample_fn, extend_fn, collision_fn, goal_test=lambda q: False,
        goal_probability=.2, max_iterations=RRT_ITERATIONS, max_time=INF):
    """
    :param start: Start configuration - conf
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param sample_fn: Sample function - sample_fn()->conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_iterations: Maximum number of iterations - int
    :param max_time: Maximum runtime - float
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    start_time = time.time()
    if collision_fn(start):
        return None
    if not callable(goal_sample):
        g = goal_sample
        goal_sample = lambda: g
    nodes = [TreeNode(start)]
    for i in irange(max_iterations):
        if elapsed_time(start_time) >= max_time:
            break
        goal = random() < goal_probability or i == 0
        s = goal_sample() if goal else sample_fn()

        last = argmin(lambda n: distance_fn(n.config, s), nodes)
        for q in extend_fn(last.config, s):
            if collision_fn(q):
                break
            last = TreeNode(q, parent=last)
            nodes.append(last)
            if goal_test(last.config):
                return configs(last.retrace())
        else:
            if goal:
                return configs(last.retrace())
    return None

def load_robot(urdf_path):
    """Loads the robot arm from the URDF file."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    #p.setGravity(0, 0, -9.8)
    camera_distance = 1.5  # Adjust this value to zoom in/out
    camera_yaw = 50  # Camera yaw angle
    camera_pitch = -30  # Camera pitch angle
    camera_target = [0, 0, 0.5]  # Target position to look at

    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target)
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(urdf_path, [0, 0, 0],useFixedBase=1)
    return robot_id

def random_config(robot_id, joint_indices):
    """Generate a random configuration for the robot."""
    joint_positions = []
    for joint_index in joint_indices:
        joint_info = p.getJointInfo(robot_id, joint_index)
        joint_lower_limit = joint_info[8]
        joint_upper_limit = joint_info[9]
        joint_positions.append(random.uniform(joint_lower_limit, joint_upper_limit))
    return joint_positions

def distance(c1, c2):
    """Calculate the distance between two configurations."""
    return np.linalg.norm(np.array(c1) - np.array(c2))

def steer(q_nearest, q_rand):
    """Steer from q_nearest towards q_rand by STEP_SIZE."""
    direction = np.array(q_rand) - np.array(q_nearest)
    norm = np.linalg.norm(direction)
    if norm > STEP_SIZE:
        direction = (direction / norm) * STEP_SIZE
    return list(np.array(q_nearest) + direction)

def check_collision(robot_id, joint_indices, q):
    """Check if a configuration causes a collision."""
    for i, joint_index in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_index, q[i])
    p.performCollisionDetection()
    contacts = p.getContactPoints(bodyA=robot_id)
    return len(contacts) > 0

def rrt(robot_id, joint_indices, start_config, goal_config):
    """RRT algorithm to find a path from start_config to goal_config."""
    tree = {tuple(start_config): None}  # RRT tree, stores parent relationships
    nodes = [start_config]

    for _ in range(MAX_ITER):
        q_rand = random_config(robot_id, joint_indices)
        q_nearest = min(nodes, key=lambda node: distance(node, q_rand))
        q_new = steer(q_nearest, q_rand)

        if not check_collision(robot_id, joint_indices, q_new):
            tree[tuple(q_new)] = tuple(q_nearest)
            nodes.append(q_new)

            # Check if the new configuration is close to the goal
            if distance(q_new, goal_config) < GOAL_THRESHOLD:
                tree[tuple(goal_config)] = tuple(q_new)
                return tree, q_new

    return None, None  # Failed to find a path

def extract_path(tree, goal_config):
    """Extracts the path from the RRT tree."""
    path = [goal_config]
    while tree[tuple(path[-1])] is not None:
        path.append(tree[tuple(path[-1])])
    path.reverse()
    return path

def visualize_path(robot_id, joint_indices, path):
    """Visualizes the planned path in PyBullet."""
    p.setRealTimeSimulation(1)
    for config in path:
        for i, joint_index in enumerate(joint_indices):
            p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, config[i])
        time.sleep(1)

    print("Path visualization completed!")
    p.setRealTimeSimulation(0)


if __name__ == "__main__":
    # Load the robot
    urdf_path = "C:/Users/arbxe/OneDrive/Desktop/code stuff/EmPRISE repos/mp-osc/mp-osc/multipriority/urdfs/GEN3_URDF_V12.urdf"
    robot_id = load_robot(urdf_path)
        # Define the position where you want to place the box
    box_position = [-0.6, 0, 0.5]
    box_dims = [0.4,0.05,0.4]
    create_box(box_position,box_dims)
    time.sleep(10)
    # Lock the base of the robot to prevent it from falling over
    # Get joint indices (non-fixed joints)
    joint_indices = [i for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
    print(p.getNumJoints(robot_id))
    print(joint_indices)
    num_j = 0*len(joint_indices)
    for i in range(0,num_j):
        config = [0.0] * num_j
        moving_joint = i
        values = [0.2, 0.5, -1, 0]
        print(moving_joint)
        path = [config]
        for j in values:
            config[moving_joint] = j
            path.append(config.copy())
        visualize_path(robot_id, joint_indices, path)  
    
   
    # Define start and goal configurations
    start_config = [0.0] * len(joint_indices)
    goal_config = [0.5] * len(joint_indices)  # Example goal configuration
    print(start_config)
    print(goal_config)


    # Plan the path using RRT
    path = [start_config, goal_config]
    visualize_path(robot_id, joint_indices, path)
    time.sleep(3)

    tree, last_node = rrt(robot_id, joint_indices, start_config, goal_config)
    
    if tree:
        path = extract_path(tree, goal_config)
        print("Path found!")
        visualize_path(robot_id, joint_indices, path)
        print(path)
    else:
        print("Failed to find a path.")

    # Disconnect from PyBullet
    p.disconnect()