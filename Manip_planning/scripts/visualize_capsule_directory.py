import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import sys

# === Load capsule_data from file ===
capsule_path = r"C:\Users\arbxe\EmPRISE code\CRGP\Manip_planning\mp-osc\multipriority\urdfs\gen3_capsules_directory.py"
urdf_path = r"C:\Users\arbxe\EmPRISE code\CRGP\Manip_planning\mp-osc\multipriority\urdfs\GEN3_URDF_V12_w_taxels.urdf"

# Dynamically import the capsule_data dictionary
capsule_dir = os.path.dirname(capsule_path)
sys.path.append(capsule_dir)
from gen3_capsules_directory import capsule_data  # assumes the file defines capsule_data directly
# === Helper functions ===
def add_capsule_visuals(capsule_data, robot_id, link_name_to_index):
    capsule_ids = []
    for link_name, capsules in capsule_data.items():

        link_index = link_name_to_index.get(link_name)
        if link_index is None:
            print(f"[WARN] Link '{link_name}' not found in URDF.")
            continue

        for capsule in capsules:
            p1 = np.array(capsule['p1_local'])
            p2 = np.array(capsule['p2_local'])
            radius = capsule['radius']

            center = (p1 + p2) / 2
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            z_axis = direction / length

            default_axis = np.array([0, 0, 1])
            axis = np.cross(default_axis, z_axis)
            angle = np.arccos(np.clip(np.dot(default_axis, z_axis), -1.0, 1.0))
            if np.linalg.norm(axis) < 1e-6:
                quat = [0, 0, 0, 1] if angle < 1e-6 else [1, 0, 0, 0]
            else:
                axis /= np.linalg.norm(axis)
                quat = p.getQuaternionFromAxisAngle(axis.tolist(), angle)

            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_CAPSULE,
                radius=radius,
                length=length,
                visualFramePosition=center.tolist(),
                visualFrameOrientation=quat,
                rgbaColor=[1, 0, 0, 0.4]  # red, semi-transparent
            )

            body_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id,
                baseCollisionShapeIndex=-1,
                basePosition=[0, 0, 0],
                useMaximalCoordinates=True
            )
            # print(f"Created capsule visual for link '{link_name}' with name '{capsule['name']}' and quaternion {quat} and radius {radius}")
            # input()
            capsule_ids.append((body_id, link_index, center, quat))

    return capsule_ids

def update_capsules(capsule_ids, robot_id):
    for body_id, link_idx, local_pos, local_quat in capsule_ids:
        print(f"Updating capsule {body_id} for link {link_idx}")
        if link_idx == -1:
            # Skip base link, as getLinkState is invalid for base (-1)
            continue
        link_state = p.getLinkState(robot_id, link_idx, computeForwardKinematics=True)
        link_world_pos = link_state[4]
        link_world_ori = link_state[5]
        world_pos, world_ori = p.multiplyTransforms(link_world_pos, link_world_ori, local_pos, local_quat)
        p.resetBasePositionAndOrientation(body_id, world_pos, world_ori)

# === PyBullet Setup ===
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

robot_id = p.loadURDF(urdf_path, useFixedBase=True)

# Get link name to index mapping
link_name_to_index = {p.getBodyInfo(robot_id)[0].decode('utf-8'): -1}
for i in range(p.getNumJoints(robot_id)):
    link_name = p.getJointInfo(robot_id, i)[12].decode('utf-8')
    link_name_to_index[link_name] = i
q = [-0.2, 0.4, 0.4, 0.9, 0.9, 0.9, 0]
for i, value in enumerate(q):
    p.resetJointState(robot_id, i, value)
# Create and track capsules
capsule_ids = add_capsule_visuals(capsule_data, robot_id, link_name_to_index)
update_capsules(capsule_ids, robot_id)
# Run simulation loop
c = 0
while p.isConnected():
    # update_capsules(capsule_ids, robot_id)
    p.stepSimulation()
    time.sleep(1/240)

