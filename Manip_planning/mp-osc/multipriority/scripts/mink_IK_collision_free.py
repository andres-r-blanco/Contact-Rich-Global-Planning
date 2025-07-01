import numpy as np
import mujoco
import mink
import xml.etree.ElementTree as ET
from lxml import etree
import os
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
from itertools import product


class MinkIKSolver:
    def __init__(self, mjcf_path, obstacle_name_list = []):
        self.mjcf_path = mjcf_path
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)
        self.rate = RateLimiter(frequency=200.0, warn=False)
        self.obstacle_name_list = obstacle_name_list

        # Reset to home keyframe if exists
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if keyframe_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
            self.home_qpos = self.data.qpos.copy()
        else:
            # print("[WARNING] No 'home' keyframe found in MJCF. Using default qpos=0.")
            self.home_qpos = np.zeros(self.model.nq)

        # Set end-effector site name
        if self.model.nsite > 0:
            self.ee_site = "ee_site"
        else:
            raise RuntimeError("No site found in the MJCF model for end-effector.")

        # Build collision pairs (robot vs obstacle only)
        self.collision_pairs = self.build_collision_pairs()

        # Build tasks
        self.posture_task = mink.PostureTask(self.model, cost=1e-3)
        
        end_effector_task = mink.FrameTask(
            frame_name=self.ee_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        )
        end_effector_task_no_ori = mink.FrameTask(
            frame_name=self.ee_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1e-6,
        )
        self.tasks = [[end_effector_task, self.posture_task]]
        self.tasks_no_ori = [[end_effector_task_no_ori, self.posture_task]]
        for i in range(28):
            if i < 10:
                taxel_name = f"Taxel_100{i}"
            else:
                taxel_name = f"Taxel_10{i}"
            taxel_site_name = f"{taxel_name}_contact_site"
            taxel_task = mink.FrameTask(
                frame_name=taxel_site_name,
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1e-6,
            )
            taxel_task_no_ori = mink.FrameTask(
                frame_name=taxel_site_name,
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1e-6,
            )
            self.tasks.append([taxel_task,self.posture_task])
            self.tasks_no_ori.append([taxel_task_no_ori,self.posture_task])


        # Build limits
        limits = [
            mink.ConfigurationLimit(model=self.configuration.model),
            mink.CollisionAvoidanceLimit(
                model=self.configuration.model,
                geom_pairs=self.collision_pairs,
                minimum_distance_from_collisions = 0.0,
                collision_detection_distance = 0.02,
                bound_relaxation = 0.0,
                gain = 0.85
            ),
        ]

        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(self.model.njnt)
        ]
        max_velocities = {name: np.pi for name in joint_names if name is not None}
        velocity_limit = mink.VelocityLimit(self.model, max_velocities)
        limits.append(velocity_limit)

        self.limits = limits

    def build_collision_pairs(self):
        robot_geom_ids = []
        obstacle_geom_ids = [] #TODO maybe remove

        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name is None:
                continue

            if geom_name.startswith("box_") or geom_name.startswith("cylinder_"):
                obstacle_geom_ids.append(geom_id)
            elif "Taxel" not in geom_name:
                robot_geom_ids.append(geom_id)
        # print(f"Robot Geom IDs: {robot_geom_ids}")
        # # print(f"obstacle_name_list: {self.obstacle_name_list}")
        # print(f"Obstacle Geom IDs: {obstacle_geom_ids}")
        # input()
        pairs = [
            (robot_geom_ids, obstacle_geom_ids),
        ]
        return pairs

    def solve(self, goal_pos, goal_quat=None, initial_qpos=None, site_target = 0, verbose=False, iterations = 50, pos_threshold = 1e-4, ori_threshold = 1e-2, viz= False):
        if initial_qpos is None:
            initial_qpos = self.home_qpos
        
        if site_target is None or site_target < 0: 
            site_target = 0
        else:
            site_target = int(site_target+1)
        
        self.configuration.update(initial_qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

        # Set target for end-effector, if goal_quat is None will ignore orientation
        if goal_quat is None:
            goal_quat = [1, 0, 0, 0]
            current_task = self.tasks_no_ori[site_target]
            no_ori = True
        else:
            goal_quat = pybullet_to_mujoco_quat(goal_quat)
            current_task = self.tasks[site_target]
            no_ori = False
        current_site_task = current_task[0]
            
            
        # print(f"goal_pos: {goal_pos}, goal_quat: {goal_quat}, initial_qpos: {initial_qpos}")
        tb = mink.SE3(np.concatenate([goal_quat,goal_pos]))
        # print(f"Setting end-effector target: {tb}")
        current_site_task.set_target(tb)
        
        # IK settings (other than the thresholds)
        solver = "daqp"
        max_iters = iterations
        # input("Press Enter to continue...")
        for i in range(max_iters):
            try:
                vel = mink.solve_ik(
                    self.configuration, current_task, self.rate.dt, solver, limits=self.limits
                )
            except mink.NoSolutionFound:
                # print(f"[FAILURE] IK failed at iteration {i+1}. Collision constraint couldn't be satisfied.")
                return None  # <-- gracefully exit

            self.configuration.integrate_inplace(vel, self.rate.dt)
            err = current_site_task.compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            if no_ori:
                ori_achieved = True
            else:
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
            if pos_achieved and ori_achieved:
                # print(f"Iteration {i+1}: pos_error={err[:3]}, ori_error={err[3:]}, pos_achieved={pos_achieved}, ori_achieved={ori_achieved}")
                # print(f"qpos achieved: {self.configuration.q}")
                if viz: visualize_model(self.model, self.data, point= goal_pos, robot_q=self.configuration.q)
                return self.configuration.q
        # print(f"Iteration {i+1}: pos_error={err[:3]}, ori_error={err[3:]}, pos_achieved={pos_achieved}, ori_achieved={ori_achieved}")
        
        # input(f"IK converged in {i+1} iterations. Press Enter to continue...")
        return None


### Helper functions below remain the same ###

def inject_obstacles(mjcf_path, obstacle_list):
    # Create new MJCF root
    mujoco = ET.Element('mujoco', {'model': 'temp_mj_gen3_scene'})

    # Add include for the robot file
    robot_file = os.path.basename(mjcf_path)
    include = ET.SubElement(mujoco, 'include')
    include.set('file', robot_file)

    # Add visual section
    visual = ET.SubElement(mujoco, 'visual')
    headlight = ET.SubElement(visual, 'headlight', {
        'diffuse': "0.6 0.6 0.6",
        'ambient': "0.1 0.1 0.1",
        'specular': "0 0 0"
    })
    global_light = ET.SubElement(visual, 'global', {
        'azimuth': "120",
        'elevation': "-20"
    })

    # Add asset section
    asset = ET.SubElement(mujoco, 'asset')
    texture = ET.SubElement(asset, 'texture', {
        'name': "grid",
        'type': "2d",
        'builtin': "checker",
        'rgb1': ".2 .3 .4",
        'rgb2': ".1 0.15 0.2",
        'width': "512",
        'height': "512",
        'mark': "cross",
        'markrgb': ".8 .8 .8"
    })
    material = ET.SubElement(asset, 'material', {
        'name': "grid",
        'texture': "grid",
        'texrepeat': "1 1",
        'texuniform': "true"
    })

    # Add worldbody
    worldbody = ET.SubElement(mujoco, 'worldbody')
    ET.SubElement(worldbody, 'light', {'pos': "0 0 1.5", 'directional': "true"})
    ET.SubElement(worldbody, 'geom', {
        'name': "floor",
        'size': "1 1 0.01",
        'type': "plane",
        'material': "grid"
    })

    # Add obstacles
    obstacle_name_list = []
    for obs in obstacle_list:
        body = ET.SubElement(worldbody, 'body', {
            'name': obs['name'],
            'pos': ' '.join([f"{x:.6f}" for x in obs['pos']]),
            'quat': ' '.join([f"{x:.6f}" for x in obs['quat']])
        })

        geom = ET.SubElement(body, 'geom', {
            'name': obs['name'],
            'contype': "1",
            'conaffinity': "1",
            'rgba': "0.3 0.3 0.3 1",
        })

        if obs['type'] == 'box':
            geom.set('type', 'box')
            size = obs['size']
            geom.set('size', ' '.join([f"{x:.6f}" for x in size]))
        elif obs['type'] == 'cylinder':
            geom.set('type', 'cylinder')
            radius, height = obs['size']
            geom.set('size', f"{radius:.6f} {height/2:.6f}")
        else:
            raise ValueError(f"Unknown obstacle type: {obs['type']}")

        obstacle_name_list.append(obs['name'])

    xml_str = ET.tostring(mujoco, encoding='utf-8')
    parsed = etree.fromstring(xml_str)
    pretty_xml = etree.tostring(parsed, pretty_print=True, encoding='utf-8', xml_declaration=True)

    # Write pretty file
    dir_path = os.path.dirname(mjcf_path)
    new_filename = os.path.join(dir_path, f"temp_mj_gen3_scene.xml")
    with open(new_filename, 'wb') as f:
        f.write(pretty_xml)
    # print(f"Saved wrapper MJCF to {new_filename} with obstacles: {obstacle_name_list}")
    # visualize_mjcf(new_filename)

    return new_filename, obstacle_name_list

def indent(elem, level=0):
    i = "\n" + level*"  "  # 2 spaces per indent level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level+1)
            if not child.tail or not child.tail.strip():
                child.tail = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def build_mujoco_obstacle_list(box_inputs, cylinder_inputs):
    obstacles = []

    for i, box in enumerate(box_inputs):
        obstacles.append({
            'type': 'box',
            'name': f'box_{i}',
            'pos': box['pos'],
            'quat': [1, 0, 0, 0],  # assuming no rotation
            'size': box['dims']
        })

    for i, cyl in enumerate(cylinder_inputs):
        quat = cyl['quat'] if 'quat' in cyl else [0, 0, 0, 1]
        quat = pybullet_to_mujoco_quat(quat)
        obstacles.append({
            'type': 'cylinder',
            'name': f'cylinder_{i}',
            'pos': cyl['pos'],
            'quat': quat,
            'size': [cyl['radius'], cyl['height']]
        })

    return obstacles


def visualize_model(m, d, point=None, robot_q=None):
    if robot_q is not None:
        d.qpos[:] = robot_q
        mujoco.mj_forward(m, d)

    # if point is not None:
    #     site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "debug_target")
    #     d.site_xpos[site_id] = point
    #     mujoco.mj_forward(m, d)
    #     m.site_size[site_id] = np.array([0.05, 0, 0])
    #     m.site_rgba[site_id] = np.array([1, 0, 0, 1])

    with mujoco.viewer.launch_passive(m, d) as viewer:
        print(f"Visualizing — close the window to continue...")
        while viewer.is_running():
            viewer.sync()

     
def pybullet_to_mujoco_quat(pybullet_quat):
    """
    Convert a PyBullet quaternion to MuJoCo quaternion.
    
    Arguments:
        pybullet_quat: list or array-like of 4 elements [x, y, z, w] (PyBullet convention)
        apply_cylinder_fix: if True, applies extra rotation to account for cylinder Y→Z axis difference
    
    Returns:
        mujoco_quat: list of 4 elements [w, x, y, z] (MuJoCo convention)
    """
    # Convert input PyBullet quaternion [x, y, z, w] to scipy format
    q_pb = np.array(pybullet_quat)
    r_pb = R.from_quat(q_pb)
    
    # Get MuJoCo-style quaternion: [w, x, y, z]
    q_mj = r_pb.as_quat()  # still [x, y, z, w]
    mujoco_quat = [q_mj[3], q_mj[0], q_mj[1], q_mj[2]]
    return mujoco_quat

# def get_mujoco_ee_world_pose(model, data, q, ee_site_name):
#     """
#     Sets joint positions in MuJoCo and returns end-effector world-frame position.
#     """
#     # Update joint state
#     data.qpos[:len(q)] = q
#     mujoco.mj_forward(model, data)

#     # Get site position in world frame
#     site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
#     ee_pos_world = np.array(data.site_pos[site_id])
#     ee_quat_world = np.array(data.site_quat[site_id])  # If you also want orientation

#     return ee_pos_world, ee_quat_world