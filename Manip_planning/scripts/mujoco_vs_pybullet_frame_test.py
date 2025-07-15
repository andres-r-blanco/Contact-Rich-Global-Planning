import os
import sys
INIT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TACTILE_KINOVA_URDF_FOR_MUJOCO = os.path.join(INIT_PATH, "Manip_planning/mp-osc/multipriority/urdfs/gen3_taxels_with_sites_mjcf copy.xml")
TACTILE_KINOVA_URDF = os.path.join(INIT_PATH, "Manip_planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12_w_taxels.urdf")

import mujoco
import pybullet as p
import pybullet_data
import time
import mujoco.viewer


def main():
    # viz_pybullet_urdf()
    viz_mujoco_collision_bodies()

def viz_pybullet_urdf():

    # Start PyBullet in GUI mode
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, 0)

    # Load the URDF
    robot_id = p.loadURDF(TACTILE_KINOVA_URDF, useFixedBase=True)

    # Optional: set camera
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0.5])
    dot([0.2, 0, 0], color=[1, 0, 0, 1], dot_radius=0.05)      # Red dot at x=0.2
    dot([0, 0.2, 0], color=[0, 1, 0, 1], dot_radius=0.05)      # Green dot at y=0.2
    dot([0, 0, 0.2], color=[0, 0, 1, 1], dot_radius=0.05)      # Blue dot at z=0.2
    print("Visualizing URDF in PyBullet — Close the window to exit")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()

def dot(pos,color = [1,0,0,1],dot_radius = 0.01):
    # Create a small visual sphere at the end-effector's position

    # Create a visual shape for the dot (small sphere)
    dot_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=dot_radius, rgbaColor=color)
    # Create the body for the visual shape (this places the sphere at the end-effector position)
    p.createMultiBody(
        baseMass=0,  # Mass of 0 means it's static and won't be affected by physics
        baseVisualShapeIndex=dot_visual_shape,
        basePosition=pos
    )


def viz_mujoco_collision_bodies():
    # Load model
    model = mujoco.MjModel.from_xml_path(TACTILE_KINOVA_URDF_FOR_MUJOCO)
    data = mujoco.MjData(model)
    
    # Disable gravity to keep robot still
    model.opt.gravity[:] = 0
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    # Set visual geoms fully transparent (group 2 = visuals in your MJCF)
    for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] == 2:  # visual meshes
            model.geom_rgba[geom_id, 3] = 0.6  # fully transparent
        
        if model.geom_group[geom_id] == 3:  # collision capsules/boxes
            print(f"Geom {geom_id} is a collision geom with type {model.geom_type[geom_id]}")
            model.geom_group[geom_id] = 0
            model.geom_rgba[geom_id] = [1.0, 0.1, 0.1, 0.6]  # red, semi-transparent

        # Optional: obstacles (group 4 or others)
        if model.geom_group[geom_id] == 4:
            model.geom_rgba[geom_id] = [0.0, 1.0, 0.0, 0.5]  # green obstacles

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Visualizing collision geoms — Press ESC to exit")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == '__main__':
    main()