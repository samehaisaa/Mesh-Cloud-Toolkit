import numpy as np

def get_rotation_matrix(theta, axis='z'):
    """
    Get a 3x3 rotation matrix for a given angle (theta) and axis.
    
    Parameters:
    - theta (float): The angle to rotate (in radians).
    - axis (str): The axis to rotate around ('x', 'y', or 'z').
    
    Returns:
    - np.ndarray: A 3x3 rotation matrix.
    """
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
    
    return rotation_matrix

def get_translation_vector(x=0.0, y=0.0, z=0.0):
    """
    Get a translation vector.
    
    Parameters:
    - x (float): The translation along the x-axis.
    - y (float): The translation along the y-axis.
    - z (float): The translation along the z-axis.
    
    Returns:
    - np.ndarray: A 3-element translation vector.
    """
    return np.array([x, y, z])

