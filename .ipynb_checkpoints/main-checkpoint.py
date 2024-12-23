from mesh_processing.mesh_to_pointcloud import mesh_to_point_cloud
from mesh_processing.visualization import visualize_point_cloud
from mesh_processing.file_io import save_point_cloud
from experiments.transformations import apply_rigid_transformation
import open3d as o3d
import numpy as np
if __name__ == "__main__":
    mesh_file = "./data/visage.obj"  
    
    point_cloud = mesh_to_point_cloud(mesh_file, points_per_triangle=3, use_poisson_sampling=False)
    output_cloud = "./data/visage_cloud.ply"
    save_point_cloud(point_cloud, output_cloud)
    #visualize_point_cloud(point_cloud)

    point_cloud = o3d.io.read_point_cloud(output_cloud)


    theta = np.pi / 4  # 45 degrees rotation (in radians)
    axis = 'z'         # Rotate around the Z-axis
    x_trans, y_trans, z_trans = 1.0, 0.5, 0.0  # Translation values

    transformed_cloud = apply_rigid_transformation(point_cloud, theta, axis, x_trans, y_trans, z_trans)
    type  = transformed_cloud.get_geometry_type()
    print(type)
    visualize_point_cloud([transformed_cloud])
