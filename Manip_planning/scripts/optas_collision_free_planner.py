import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET


INIT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, INIT_PATH + "/Manip_planning/optas/example")
sys.path.insert(0, INIT_PATH + "/Manip_planning/mp-osc/multipriority/scripts")
sys.path.insert(0, INIT_PATH + "/Manip_planning/optas")
sys.path.insert(0, INIT_PATH + "/Manip_planning")
sys.path.insert(1, INIT_PATH + "/Manip_planning/mp-osc/pybullet_planning_master")
sys.path.insert(1, INIT_PATH + "/Manip_planning/mp-osc/multipriority/urdfs")
from pybullet_api import *

from optas.templates import Manager
import optas

# === Load capsule_data from file ===
from gen3_capsules_directory import capsule_data as link_collision_capsule_data

TACTILE_KINOVA_URDF = INIT_PATH + "/Manip_planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12_w_taxels.urdf"

from pybullet_tools.utils import add_data_path, create_box, create_cylinder, create_capsule, quat_from_euler, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF
    
from contact_manip_rrt import dot

MANIP_WEIGHT = 0.4

class SimpleJointSpacePlanner(Manager):
    def __init__(self, filename, ee_link, capsule_obstacle_names, taxel_info, duration):
        self.duration = duration
        self.ee_link = ee_link
        self.filename = filename
        self.capsule_obstacle_names = capsule_obstacle_names
        self.taxel_info = taxel_info
        super().__init__()

    def setup_solver(self, orientation_constraint=False):
        T = 20  # number of time steps
        dt = self.duration / float(T - 1)
        robot_model_input = {}
        robot_model_input["time_derivs"] = [0, 1]
        robot_model_input["urdf_filename"] = self.filename
        
        self.robot = optas.RobotModel(**robot_model_input)
        self.name = self.robot.get_name()
        builder = optas.OptimizationBuilder(T=T, robots=self.robot, derivs_align=True)

        qn = builder.add_parameter("nominal_joint_state", self.robot.ndof)
        qc = builder.add_parameter("current_joint_state", self.robot.ndof)
        pg = builder.add_parameter("position_goal", 3)
        og = builder.add_parameter("orientation_goal", 4)

        # Constraint: initial configuration
        builder.fix_configuration(self.name, config=qc)

        # Constraint: final pose
        qF = builder.get_model_state(self.name, -1)
        pF = self.robot.get_global_link_position(self.ee_link, qF)
        oF = self.robot.get_global_link_quaternion(self.ee_link, qF)
        builder.add_equality_constraint("final_position", pF, pg)
        if orientation_constraint:
            builder.add_equality_constraint("final_orientation", oF, og)

        # Constraint: dynamics
        builder.integrate_model_states(self.name, time_deriv=1, dt=dt)

        # Constraint: keep end-effector above zero
        zpad = 0.05  # cm
        for t in range(T):
            q = builder.get_model_state(self.name, t)

            # Cost: nominal pose
            builder.add_cost_term(f"nominal_{t}", 0.1 * optas.sumsqr(q - qn))

            p = self.robot.get_global_link_position(self.ee_link, q)
            z = p[2]
            zsafe = z + zpad
            builder.add_geq_inequality_constraint(f"eff_safe_{t}", zsafe)

            # p = self.robot.get_global_link_position("lbr_link_3", q)
            # z = p[2]
            # zsafe = z + zpad
            # builder.add_geq_inequality_constraint(f"elbow_safe_{t}", zsafe)

        # Cost: minimize joint velocity
        dQ = builder.get_model_states(self.name, time_deriv=1)
        w_min_vel = 0.1
        builder.add_cost_term("minimize_velocity", w_min_vel * optas.sumsqr(dQ))

        # Cost: minmize joint acceleration
        ddQ = (dQ[:, 1:] - dQ[:, :-1]) / dt
        w_min_acc = 10
        builder.add_cost_term("minimize_acceleration", w_min_acc * optas.sumsqr(ddQ))

        # Constraint: link2obstacle capsule collision avoidance
        obstacle_capsules = builder.capsule_collision_avoidance_constraints(self.name, self.capsule_obstacle_names, link_collision_capsule_data, link_names=None)
        
        #Cost: manipulability at taxels
        builder.add_manipulability_cost(self.name, MANIP_WEIGHT, obstacle_capsules, self.taxel_info)

        # Constraint: final velocity is zero
        builder.fix_configuration(self.name, t=-1, time_deriv=1)

        solver = optas.CasADiSolver(builder.build()).setup("ipopt")
        return solver

    def is_ready(self):
        return True
    
    def reset(self, params):
        self.solver.reset_parameters(params)

    # def reset(self, qc, pg, og, qn):
    #     self.solver.reset_parameters(
    #         {
    #             "current_joint_state": qc,
    #             "position_goal": pg,
    #             "orientation_goal": og,
    #             "nominal_joint_state": qn,
    #         }
    #     )

    def get_target(self):
        return self.solution

    def plan(self):
        self.solve()
        solution = self.get_target()
        plan = self.solver.interpolate(solution[f"{self.name}/q"], self.duration)
        return plan

def setup_obstacles(object_reduction = 0.0):
    obstacle_dimensions = []
    box1_position = [0.35, -0.3, 0.13]
    box1_dims = [0.26-object_reduction,1-object_reduction,0.14-object_reduction]
    col_box_id1 = create_box(box1_position,box1_dims)
    cyl_position1 = (0.35,-0.3,0.3)
    cyl_quat1 = quat_from_euler([3.14/2, 0, 0])
    rad1 = 0.18 - object_reduction
    h1 = 1 - 2*object_reduction
    # col_cyl_id1 = create_cylinder(rad1, h1, cyl_position1, cyl_quat1)
    col_cyl_id1 = create_capsule(rad1, h1, cyl_position1, cyl_quat1)
    # p.changeVisualShape(col_cyl_id1, -1, rgbaColor=[0.2,0.2,0.2,1])
    capsule_obstacle_names = ["capsule1"]
    rot = R.from_quat(cyl_quat1)
    axis_dir = rot.apply([0, 0, 1])
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    cyl_axis_start = np.array(cyl_position1) - (h1/2) * axis_dir
    cyl_axis_end = np.array(cyl_position1) + (h1/2) * axis_dir
    capsule_obstacle_dimensions = [(cyl_axis_start,cyl_axis_end, rad1)]
    obstacles = [col_cyl_id1,col_box_id1]
    return obstacles,obstacle_dimensions, capsule_obstacle_names,capsule_obstacle_dimensions

def main(gui=True):
    hz = 250
    dt = 1.0 / float(hz)
    pb = PyBullet(dt, gui=gui)
    robot_urdf = TACTILE_KINOVA_URDF
    # kuka = KukaLBR()
    robot = FixedBaseRobot(robot_urdf)

    # q0 = np.deg2rad([0, 45, 0, -90, 0, -45, 0])
    # q0 = np.deg2rad([60, 45, 0, -90, 0, -45, 0])
    # q0 = np.array([-0.2,0.4,0.4,0.9,0.9,0.9,0])
    q0 = np.array([1.6,0.7,0,1.5,0,1.5,0])
    # q0 = np.array([1.8,1.3,-1.4,0.7,-0.5,0.8,0]) # flat on other side
    robot.reset(q0)

    duration = 4.0  # seconds
    _,_, capsule_obstacle_names,capsule_obstacle_dimensions = setup_obstacles(object_reduction=0.0)
    taxel_info = parse_urdf_for_taxels(robot_urdf, taxel_prefix="Taxel_")
    planner = SimpleJointSpacePlanner(robot_urdf, "EndEffector_Link", capsule_obstacle_names, taxel_info, duration)
    
    #setup collision constraint parameters
    params = {}
    for link_name, capsules in link_collision_capsule_data.items():
        for capsule in capsules:
            name = capsule['name']
            params[name + "_position1"] = capsule['p1_local']
            params[name + "_position2"] = capsule['p2_local']
            params[name + "_radii"] = capsule['radius']
    for i, obstacle_name in enumerate(capsule_obstacle_names):
        obs_dims = capsule_obstacle_dimensions[i]
        params[obstacle_name + "_position1"] = obs_dims[0]
        params[obstacle_name + "_position2"] = obs_dims[1]
        params[obstacle_name + "_radii"] =  obs_dims[2]
    for taxel_info_tuple in taxel_info:
        taxel_name, parent_link_name, local_pos = taxel_info_tuple
        params[taxel_name + "_local_position"] = local_pos
    

    qc = robot.q()
    # pg = [0.4, 0.3, 0.4]
    og = [0, 1, 0, 0]
    pg = [0.60, -0.42,  0.41]
    # og = None
    dot(pg, [0.5,0.9,0.5,1])

    # planner.reset(qc, pg, og, q0)
    params["current_joint_state"] = qc
    params["position_goal"] = pg
    params["orientation_goal"] = og
    params["nominal_joint_state"] = q0

    planner.reset(params)
    plan = planner.plan()

    pb.start()
    
    start = time.time()
    while True:
        t = time.time() - start
        if t < duration:
            # robot.cmd(plan(t))
            robot.reset(plan(t))
        else:
            print("Completed motion")
            break

    if gui:
        while True:
            pass

    pb.stop()
    pb.close()

    return 0

def parse_urdf_for_taxels(urdf_path, taxel_prefix="Taxel_"):
    """
    Parses a URDF file and returns a list of (taxel_name, parent_link, local_position) tuples.
    Taxel position is taken from the <visual><origin xyz=...> of the link.
    Assumes taxels are links with names starting with 'Taxel_' and are attached via a fixed joint.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    taxel_info = []

    # Build a mapping from taxel link name to parent link via joint
    taxel_parent_map = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is not None and parent is not None:
            taxel_name = child.attrib["link"]
            if taxel_name.startswith(taxel_prefix):
                parent_link = parent.attrib["link"]
                taxel_parent_map[taxel_name] = parent_link

    # Now get the position from the link's visual/origin
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        if link_name.startswith(taxel_prefix):
            taxel_name = link_name
            visual = link.find("visual")
            if visual is not None:
                origin_elem = visual.find("origin")
                if origin_elem is not None:
                    xyz = origin_elem.attrib.get("xyz", "0 0 0")
                    local_pos = np.array([float(x) for x in xyz.split()])
                else:
                    local_pos = np.zeros(3)
            else:
                local_pos = np.zeros(3)
            parent_link_name = taxel_parent_map.get(taxel_name, None)
            taxel_info.append((taxel_name, parent_link_name, local_pos))
    return taxel_info

if __name__ == "__main__":
    main()
