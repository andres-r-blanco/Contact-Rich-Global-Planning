import trimesh
import numpy as np
import os

MESH_DIR = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/urdfs/meshes/"
meshes = [
    "base_link.STL",
    "shoulder_link.STL",
    "half_arm_1_link.STL",
    "half_arm_2_link.STL",
    "forearm_link.STL",
    "spherical_wrist_1_link.STL",
    "spherical_wrist_2_link.STL",
    "bracelet_link.STL"
]

def fit_capsule_pca(mesh_path):
    mesh = trimesh.load(mesh_path)
    verts = mesh.vertices - mesh.center_mass  # center the mesh

    # PCA: eigen decomposition of covariance matrix
    cov = np.cov(verts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]

    # Project points onto major axis
    projections = verts @ major_axis
    min_proj = projections.min()
    max_proj = projections.max()
    center_along_axis = (min_proj + max_proj) / 2
    half_length = (max_proj - min_proj) / 2

    # Radius: max distance orthogonal to axis
    diffs = verts - np.outer(projections, major_axis)
    radius = np.max(np.linalg.norm(diffs, axis=1))

    # Full center in 3D
    capsule_center = mesh.center_mass + major_axis * center_along_axis

    # Compute rotation matrix
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, major_axis)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, major_axis)
    if s < 1e-8:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

    euler = trimesh.transformations.euler_from_matrix(R, axes='sxyz')

    return capsule_center, euler, radius, half_length

# Output MJCF lines
for mesh_file in meshes:
    full_path = os.path.join(MESH_DIR, mesh_file)
    center, rotation, radius, half_length = fit_capsule_pca(full_path)

    pos_str = " ".join([f"{v:.5f}" for v in center])
    euler_str = " ".join([f"{v:.5f}" for v in rotation])
    size_str = f"{radius:.5f} {half_length:.5f}"

    print(f'<geom name="{mesh_file[:-4]}_collision" type="capsule" pos="{pos_str}" euler="{euler_str}" size="{size_str}" group="3"/>')
