import yaml
import pprint
import numpy as np
import math

def normalize(v):
    norm = math.sqrt(sum(x**2 for x in v))
    return [x / norm for x in v]

def cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]

def normal_to_rotation_matrix(normal):
    """
    Construct a 3x3 rotation matrix from a surface normal vector.
    The normal becomes the Z-axis of the local frame.
    Returns the rotation matrix as a list of rows (row-major), SE(3)-compatible.
    """
    z = normalize(normal)

    # Pick an 'up' vector not parallel to z
    up = [0, 1, 0] if abs(z[1]) < 0.99 else [1, 0, 0]

    x = normalize(cross(up, z))
    y = cross(z, x)  # Already orthogonal if x and z are

    # Return matrix in row-major format for SE(3)
    R = [
        [x[0], y[0], z[0]],
        [x[1], y[1], z[1]],
        [x[2], y[2], z[2]],
    ]

    return R

def vector_to_rpy(normal):
    """
    Convert a normal vector to RPY (roll, pitch, yaw) angles.
    """
    z = np.array([0, 0, 1])  # Z-axis
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # Normalize vector
    
    # Calculate yaw (rotation about Z-axis)
    yaw = np.arctan2(normal[1], normal[0])
    
    # Calculate pitch (rotation about Y-axis)
    xy_proj = np.linalg.norm(normal[:2])  # Projection on XY plane
    pitch = np.arctan2(-normal[2], xy_proj)
    
    # Roll is assumed to be zero since it's irrelevant for normals
    roll = 0
    
    return roll, pitch, yaw

def generate_taxel_urdf(yaml_file, urdf_file, output_file):
    """
    Process a YAML file with taxel data and append <link> and <joint> elements to a URDF file.
    """
    # Mapping of link indices to names
    link_map = {
        0: "base_link",
        1: "Shoulder_Link",
        2: "HalfArm1_Link",
        3: "HalfArm2_Link",
        4: "ForeArm_Link",
        5: "SphericalWrist1_Link",
        6: "SphericalWrist2_Link"
    }

    # Load the YAML data
    with open(yaml_file, 'r') as f:
        taxel_data = yaml.safe_load(f)

    # Read the existing URDF
    with open(urdf_file, 'r') as f:
        urdf_content = f.read()

    # Append new links and joints
    new_elements = []
    taxel_contact_matrix = []
    for taxel_id, taxel in taxel_data.items():
        position = taxel["Position"]
        normal = taxel["Normal"]
        parent_link_index = taxel["Link"]
        d = 0
        for i in range(len(position)):
            position[i] = position[i] + d*normal[i]
        # Get the parent link name
        parent_link = link_map.get(parent_link_index+1, "Unknown_Link")

        # Compute RPY from the normal vector
        roll, pitch, yaw = vector_to_rpy(normal)
        rot_matrix = normal_to_rotation_matrix(normal)
        taxel_tuple = ([position[0], position[1], position[2]], rot_matrix)
        taxel_contact_matrix.append(taxel_tuple)
    pprint.pprint(taxel_contact_matrix)

  #       # Generate the XML for this taxel
  #       taxel_name = f"Taxel_{taxel_id}"
  #       link_element = f"""
  # <link name="{taxel_name}">
  #   <inertial>
  #     <origin xyz="0 0 0" rpy="0 0 0" />
  #     <mass value="0" />
  #     <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0" />
  #   </inertial>
  #   <visual>
  #     <origin xyz="{position[0]} {position[1]} {position[2]}" rpy="{roll} {pitch} {yaw}" />
  #     <geometry>
  #       <mesh filename="package://meshes/bracelet_link.STL" />
  #     </geometry>
  #     <material name="carbon_fiber" />
  #   </visual>
  #   <collision>
  #     <origin xyz="{position[0]} {position[1]} {position[2]}" rpy="{roll} {pitch} {yaw}" />
  #     <geometry>
  #       <mesh filename="package://meshes/bracelet_link.STL" />
  #     </geometry>
  #   </collision>
  # </link>"""

  #       joint_element = f"""
  # <joint name="{taxel_name}_joint" type="fixed">
  #   <origin xyz="0 0 0" rpy="0 0 0" />
  #   <parent link="{parent_link}" />
  #   <child link="{taxel_name}" />
  #   <axis xyz="0 0 0" />
  # </joint>"""

  #       new_elements.append(link_element)
  #       new_elements.append(joint_element)

  #   # Write the updated URDF to the output file
  #   with open(output_file, 'w') as f:
  #       f.write(urdf_content)
  #       f.write("\n".join(new_elements))

# Example usage
input_path = r"/home/rishabh/Andres/Manip planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12.urdf"
output_path = r"/home/rishabh/Andres/Manip planning/mp-osc/multipriority/urdfs/GEN3_URDF_V12_w_taxels.urdf"
taxel_path = r"/home/rishabh/Andres/Manip planning/mp-osc/multipriority/configs/real_taxel_data_v2.yaml"
generate_taxel_urdf(taxel_path, input_path, output_path)
