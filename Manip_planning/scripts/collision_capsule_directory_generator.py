import xml.etree.ElementTree as ET
import numpy as np
import os


def euler_xyz_to_matrix(euler):
    """Convert XYZ Euler angles (in radians) to rotation matrix"""
    rx, ry, rz = euler
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    R_x = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])

    R_y = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    R_z = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])

    return R_z @ R_y @ R_x

def apply_mujoco_to_pybullet(vec):
    x,y,z = vec
    return [x,y,z]  


def extract_capsules_with_rotation(mjcf_path):
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    link_capsules = {}
    
    def recursive_find_capsules(body_element, current_link):
        for child in body_element:
            if child.tag == "geom" and child.attrib.get("class") == "collision_capsule":
                geom_name = child.attrib.get("name", "unnamed_capsule")
                radius, half_length = map(float, child.attrib["size"].split())
                pos = np.array(list(map(float, child.attrib.get("pos", "0 0 0").split())))
                euler_str = child.attrib.get("euler", "0 0 0")
                euler = np.array(list(map(float, euler_str.split())))
                R = euler_xyz_to_matrix(euler)

                axis = R @ np.array([0, 0, 1])
                p1_mujoco = pos - half_length * axis
                p2_mujoco = pos + half_length * axis

                # 🔁 Convert MuJoCo -> PyBullet frame
                p1 = apply_mujoco_to_pybullet(p1_mujoco)
                p2 = apply_mujoco_to_pybullet(p2_mujoco)

                capsule_data = {
                    "name": geom_name,
                    "p1_local": [round(x, 6) for x in p1],
                    "p2_local": [round(x, 6) for x in p2],
                    "radius": round(radius, 6)
                }

                link_capsules.setdefault(current_link, []).append(capsule_data)

            elif child.tag == "body":
                child_name = child.attrib.get("name", current_link)
                recursive_find_capsules(child, current_link=child_name)

    for body in worldbody.findall("body"):
        root_name = body.attrib.get("name", "unnamed_root")
        recursive_find_capsules(body, root_name)

    return link_capsules


def save_capsules_as_py(output_path, capsule_dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("capsule_data = {\n")
        for link in sorted(capsule_dict.keys()):
            f.write(f'    "{link}": [\n')
            for capsule in capsule_dict[link]:
                f.write("        {\n")
                f.write(f'            "name": "{capsule["name"]}",\n')
                f.write(f'            "p1_local": {capsule["p1_local"]},\n')
                f.write(f'            "p2_local": {capsule["p2_local"]},\n')
                f.write(f'            "radius": {capsule["radius"]}\n')
                f.write("        },\n")
            f.write("    ],\n")
        f.write("}\n")


if __name__ == "__main__":
    mjcf_path = r"C:\Users\arbxe\EmPRISE code\CRGP\Manip_planning\mp-osc\multipriority\urdfs\gen3_taxels_with_sites_mjcf.xml"
    output_py = r"C:\Users\arbxe\EmPRISE code\CRGP\Manip_planning\mp-osc\multipriority\urdfs\gen3_capsules_directory.py"

    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"MJCF file not found at: {mjcf_path}")

    capsules = extract_capsules_with_rotation(mjcf_path)
    save_capsules_as_py(output_py, capsules)

    total_caps = sum(len(v) for v in capsules.values())
    print(f"Saved {total_caps} capsules across {len(capsules)} links to {output_py}")
