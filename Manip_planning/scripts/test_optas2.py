import optas
import numpy as np
import sys
from optas.templates import Manager

INIT_PATH = r"/home/rishabh/Andres"
sys.path.insert(0, INIT_PATH + "/Manip_planning/optas/example")
sys.path.insert(0, INIT_PATH + "/Manip_planning/mp-osc/multipriority/scripts")
sys.path.insert(1, INIT_PATH + "/Manip_planning/mp-osc/pybullet_planning_master")
from pybullet_api import *

TACTILE_KINOVA_URDF = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12_w_taxels.urdf"

from pybullet_tools.utils import add_data_path, create_box, create_cylinder, quat_from_euler, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF
    
from contact_manip_rrt import dot

class SimpleJointSpacePlanner(Manager):
    def __init__(self, filename, ee_link, duration):
        self.duration = duration
        self.ee_link = ee_link
        self.filename = filename
        super().__init__()

    def setup_solver(self,orientation_constraint=False):
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

        # Constraint: final velocity is zero
        builder.fix_configuration(self.name, t=-1, time_deriv=1)

        solver = optas.CasADiSolver(builder.build()).setup("ipopt")
        return solver

    def is_ready(self):
        return True

    def reset(self, qc, pg, og, qn):
        self.solver.reset_parameters(
            {
                "current_joint_state": qc,
                "position_goal": pg,
                "orientation_goal": og,
                "nominal_joint_state": qn,
            }
        )

    def get_target(self):
        return self.solution

    def plan(self):
        self.solve()
        solution = self.get_target()
        plan = self.solver.interpolate(solution[f"{self.name}/q"], self.duration)
        return plan

def setup_obstacles(object_reduction = 0.05):
    obstacle_dimensions = []
    box1_position = [0.35, -0.3, 0.13]
    box1_dims = [0.26-object_reduction,1-object_reduction,0.14-object_reduction]
    col_box_id1 = create_box(box1_position,box1_dims)
    cyl_position1 = (0.35,-0.3,0.3)
    cyl_quat1 = quat_from_euler([3.14/2, 0, 0])
    rad1 = 0.18 - object_reduction
    h1 = 1 - 2*object_reduction
    col_cyl_id1 = create_cylinder(rad1, h1, cyl_position1, cyl_quat1)
    # p.changeVisualShape(col_cyl_id1, -1, rgbaColor=[0.2,0.2,0.2,1])
    
    obstacles = [col_cyl_id1,col_box_id1]
    return obstacles, obstacle_dimensions

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
    robot.reset(q0)

    duration = 4.0  # seconds
    planner = SimpleJointSpacePlanner(robot_urdf, "EndEffector_Link", duration)
    setup_obstacles()
    qc = robot.q()
    # pg = [0.4, 0.3, 0.4]
    og = [0, 1, 0, 0]
    pg = [0.60, -0.42,  0.41]
    # og = None
    dot(pg, [0.5,0.9,0.5,1])

    planner.reset(qc, pg, og, q0)
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


if __name__ == "__main__":
    main()
