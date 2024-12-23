import open3d as o3d
import os

def save_point_cloud(point_cloud, output_file):
    """
    Saves a point cloud to the specified file.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): The point cloud to save.
        output_file (str): Path to the output file.
    """
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise TypeError("The provided object is not a valid Open3D PointCloud.")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    o3d.io.write_point_cloud(output_file, point_cloud)
    print(f"Point cloud saved to: {output_file}")
