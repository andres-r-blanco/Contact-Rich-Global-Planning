from dm_control import URDF
from dm_control import mjcf

# Load URDF
robot = URDF.load("/home/rishabh/Andres/Manip planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12_for_mujoco.urdf")

# Convert to MJCF
mjcf_model = mjcf.from_urdf_model(robot)

# Save to XML
with open("/home/rishabh/Andres/Manip planning/mp-osc/multipriority/urdfs/mjcf_gen3_with_taxels.xml", "w") as f:
    f.write(mjcf_model.to_xml_string())