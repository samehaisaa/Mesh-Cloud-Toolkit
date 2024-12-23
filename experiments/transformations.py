import open3d as o3d
import numpy as np
from experiments.transformation_params import get_rotation_matrix, get_translation_vector

def apply_rigid_transformation(cloud, theta, axis='z', x_trans=0.0, y_trans=0.0, z_trans=0.0):
    """
    Apply a rigid transformation (rotation and translation) to a point cloud.
    
    Parameters:
    - cloud (o3d.geometry.PointCloud): The input point cloud.
    - theta (float): The angle of rotation (in radians).
    - axis (str): The axis to rotate around ('x', 'y', or 'z').
    - x_trans, y_trans, z_trans (float): The translation vector components.
    
    Returns:
    - o3d.geometry.PointCloud: The transformed point cloud.
    """
    

    type  = cloud.get_geometry_type()
    print("current cloud type in apply rigid function : ", type)

    rotation_matrix = get_rotation_matrix(theta, axis)
    translation_vector = get_translation_vector(x_trans, y_trans, z_trans)
    print("just got rotation and translation vector  : " ,rotation_matrix )
    # Apply rotation and translation

    print("current cloud type in apply rigid function after applying rotation: ", type)

    print("rotated the cloud")
    transformed_cloud= cloud.translate(translation_vector)
    print("trannslated the cloud by : ", translation_vector)
    type  = transformed_cloud.get_geometry_type()

    
    return transformed_cloud
