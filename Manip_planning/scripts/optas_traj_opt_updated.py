
# Updated optas trajectory planning script with collision avoidance and manipulability optimization

import os
import sys
import pathlib
import optas
import casadi as ca

INIT_PATH = r"/home/rishabh/Andres"
sys.path.insert(0, INIT_PATH + "/Manip_planning/optas/example")
sys.path.insert(1, INIT_PATH + "/Manip_planning/mp-osc/pybullet_planning_master")
from optas.templates import Manager

import pybullet_api as p
from pybullet_tools.utils import (
    create_box, create_cylinder, quat_from_euler,
    set_camera_pose, get_sample_fn
)

initial_configuration = optas.np.deg2rad([0, 30, 0, -90, 0, -30, 0])
target_position = optas.DM([0.50, -0.30, 0.50])
LINK_SPHERE_RADIUS = 0.05
OBSTACLE1_POS = optas.DM([0.35, -0.30, 0.13])
OBSTACLE1_RADIUS = optas.DM(0.50)
OBSTACLE2_POS = optas.DM([0.35, -0.30, 0.60])
OBSTACLE2_RADIUS = optas.DM(0.20)

class Planner(Manager):
    def setup_solver(self):
        self.T = 50
        self.Tmax = 10.0
        t = optas.linspace(0, self.Tmax, self.T)
        self.dt = float((t[1] - t[0]).toarray()[0, 0])
        link_ee = "lbr_link_ee"
        filename = os.path.join(INIT_PATH, "Manip_planning", "optas", "example", "robots", "kuka_lbr", "med7.urdf.xacro")

        robot_model_input = {"time_derivs": [0, 1], "xacro_filename": filename}
        self.kuka = optas.RobotModel(**robot_model_input)
        self.kuka_name = self.kuka.get_name()

        builder = optas.OptimizationBuilder(T=self.T, robots=[self.kuka])
        qc = builder.add_parameter("qc", self.kuka.ndof)
        builder.fix_configuration(self.kuka_name, config=qc)
        builder.fix_configuration(self.kuka_name, time_deriv=1)
        builder.integrate_model_states(self.kuka_name, time_deriv=1, dt=self.dt)

        Q = builder.get_model_states(self.kuka_name)
        dQ = builder.get_model_states(self.kuka_name, time_deriv=1)

        pos = self.kuka.get_global_link_position_function(link_ee, n=self.T)
        pos_ee = pos(Q)
        quat = self.kuka.get_global_link_quaternion_function(link_ee, n=self.T)
        pc = self.kuka.get_global_link_position(link_ee, qc)
        Rc = self.kuka.get_global_link_rotation(link_ee, qc)
        quatc = self.kuka.get_global_link_quaternion(link_ee, qc)

        builder.add_equality_constraint("final_pos", pos_ee[:, self.T-1], target_position)
        builder.add_equality_constraint("no_eff_rot", quat(Q), quatc)

        self.link_names = [f"lbr_link_{i}" for i in range(1, 8)] + ["lbr_link_ee"]
        builder.sphere_collision_avoidance_constraints(self.kuka_name,
            obstacle_names=["obstacle1", "obstacle2"], link_names=self.link_names)

        manip_cost = 0
        for link in self.link_names:
            for k in range(self.T):
                J = self.kuka.get_global_link_linear_jacobian(link, Q[:, k])
                mu = ca.sqrt(ca.det(J @ J.T))
                manip_cost += -mu
        builder.add_cost_term("manipulability", 1.0 * manip_cost)

        builder.add_cost_term("min_joint_vel", 0.01 * optas.sumsqr(dQ))
        acc_cost = 0
        for k in range(self.T - 1):
            acc_cost += optas.sumsqr(dQ[:, k+1] - dQ[:, k])
        builder.add_cost_term("min_accel", 0.001 * acc_cost)

        optimization = builder.build()
        solver = optas.CasADiSolver(optimization).setup("ipopt")
        return solver

    def reset(self, qc):
        params = {
            "qc": optas.DM(qc),
            "obstacle1_position": OBSTACLE1_POS, "obstacle1_radii": OBSTACLE1_RADIUS,
            "obstacle2_position": OBSTACLE2_POS, "obstacle2_radii": OBSTACLE2_RADIUS,
        }
        for link in self.link_names:
            params[f"{link}_radii"] = optas.DM(LINK_SPHERE_RADIUS)
        self.solver.reset_parameters(params)
        Q0 = optas.diag(optas.DM(qc)) @ optas.DM.ones(self.kuka.ndof, self.T)
        self.solver.reset_initial_seed({f"{self.kuka_name}/q/x": Q0})

    def get_target(self):
        return self.solution

    def plan(self):
        self.solve()
        self.solution = self.get_target()
        plan = self.solver.interpolate(self.solution[f"{self.kuka_name}/q"], self.Tmax)
        return lambda t: plan(t)

def main(gui=True):
    planner = Planner()
    planner.reset(initial_configuration)
    plan = planner.plan()

    pb = p.PyBullet(1.0 / 50.0, gui=gui)
    set_camera_pose(camera_point=[0.9, 0.2, 1], target_point=[0.35, -0.2, 0.13])
    create_box([0.35, -0.3, 0.13], [0.26, 1.0, 0.14])
    create_cylinder(0.18, 1.0, [0.35, -0.3, 0.3], quat_from_euler([3.14 / 2, 0, 0]))

    kuka = p.KukaLBR()
    kuka.reset(plan(0.0))
    pb.start()

    import time
    start_time = time.time()
    while True:
        t = time.time() - start_time
        if t > planner.Tmax:
            break
        kuka.cmd(plan(t))
        time.sleep(1.0 / 50.0 * gui)

    pb.stop()
    pb.close()

if __name__ == "__main__":
    sys.exit(main())