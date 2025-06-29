import xml.etree.ElementTree as ET
import os

urdf_path = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12_for_mujoco.urdf"
mesh_base = os.path.dirname(urdf_path)

tree = ET.parse(urdf_path)
root = tree.getroot()

for mesh in root.findall(".//mesh"):
    mesh_file = mesh.attrib["filename"]
    # if "package://" in mesh_file:
    #     mesh_file = mesh_file.replace("package://", "")
    mesh_file = os.path.join(mesh_base, mesh_file)
    if not os.path.exists(mesh_file):
        print(f"❌ Missing: {mesh_file}")
    else:
        print(f"✅ Found: {mesh_file}")

for mesh in root.findall(".//mesh"):
    print(mesh.attrib["filename"])
