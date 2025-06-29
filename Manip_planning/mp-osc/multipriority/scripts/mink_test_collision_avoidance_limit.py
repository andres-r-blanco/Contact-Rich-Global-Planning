import mujoco
import mujoco.viewer
import numpy as np
from numpy.linalg import norm
import mink
from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.utils import get_body_geom_ids
from absl.testing import absltest
from pathlib import Path
import tempfile
import time

TACTILE_KINOVA_URDF_FOR_MUJOCO = Path(
    "/home/rishabh/Andres/Manip planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12_for_mujoco.urdf"
)


class TestCollisionAvoidanceLimit(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Wrap URDF in MJCF and add obstacle
        scene_xml = f"""
<mujoco>
  <compiler angle="radian" coordinate="local" meshdir="." texturedir="." />
  <option gravity="0 0 -9.81" />
  <compiler urdf="true" />
  <default>
    <joint limited="true"/>
  </default>

  <worldbody>
    <body name="obstacle_sphere" pos="0.3 0.3 0.4">
      <geom name="obstacle_geom" type="sphere" size="0.05"
            rgba="1 0 0 0.5" contype="1" conaffinity="1"/>
    </body>
  </worldbody>

  <include file="{TACTILE_KINOVA_URDF_FOR_MUJOCO.name}" />
</mujoco>
"""

        # Write URDF and scene into temp dir
        cls.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(cls.temp_dir.name)

        urdf_copy = temp_path / TACTILE_KINOVA_URDF_FOR_MUJOCO.name
        urdf_copy.write_text(TACTILE_KINOVA_URDF_FOR_MUJOCO.read_text())

        scene_path = temp_path / "scene_with_obstacle.xml"
        scene_path.write_text(scene_xml)

        # Load model
        cls.model = mujoco.MjModel.from_xml_path(str(scene_path))
        cls.data = mujoco.MjData(cls.model)

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")

        # Velocity limits
        velocities = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i): np.pi
            for i in range(self.model.njnt)
        }
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model, velocities),
        ]

        # Add self-collision and obstacle avoidance
        g1 = get_body_geom_ids(self.model, self.model.body("SphericalWrist1_Link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("HalfArm2_Link").id)

        obstacle_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom"
        )
        ee_geom_ids = get_body_geom_ids(self.model, self.model.body("Bracelet_Link").id)

        collision_limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[
                (g1, g2),  # self collision
                (ee_geom_ids, [obstacle_geom_id]),
            ],
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.1,
        )
        self.limits.append(collision_limit)

    def test_ik_to_target_position(self):
        configuration = self.configuration
        model = configuration.model
        data = configuration.data

        # Use the last body
        ee_body = model.body(model.nbody - 1).name
        print(f"Using end-effector body: {ee_body}")

        # Target position
        target_position = np.array([0.5, 0.5, 0.5])
        target_transform = mink.SE3.from_translation(target_position)

        task = mink.FrameTask(
            frame_name=ee_body,
            frame_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        task.set_target(target_transform)

        dt = 5e-3
        solver = "quadprog"

        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Moving end-effector to", target_position)
            for step in range(300):
                task.set_target(target_transform)
                velocity = mink.solve_ik(
                    configuration, [task], limits=self.limits, dt=dt, solver=solver
                )
                configuration.integrate_inplace(velocity, dt)
                data.qpos[:] = configuration.q
                mujoco.mj_forward(model, data)

                err = norm(task.compute_error(configuration))
                print(f"Step {step:03d} | Error: {err:.5f}")
                viewer.sync()
                time.sleep(0.05)

                if err < 1e-4 and np.allclose(velocity, 0.0):
                    print(f"Converged in {step} steps.")
                    break

            print("IK complete. Press Enter to close viewer.")
            input()


if __name__ == "__main__":
    test = TestCollisionAvoidanceLimit()
    test.setUpClass()
    test.setUp()
    test.test_ik_to_target_position()
