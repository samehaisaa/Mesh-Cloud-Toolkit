from mesh_processing.mesh_to_pointcloud import mesh_to_point_cloud
from mesh_processing.visualization import visualize_point_cloud

if __name__ == "__main__":
    # Path to your .obj file
    mesh_file = "path/to/your/mesh.obj"  # Replace with actual path
    
    # Convert the mesh to point cloud
    point_cloud = mesh_to_point_cloud(mesh_file, points_per_triangle=3, use_poisson_sampling=False)
    
    # Visualize the point cloud
    visualize_point_cloud(point_cloud)
from mesh_processing.mesh_to_pointcloud import mesh_to_point_cloud
from mesh_processing.visualization import visualize_point_cloud

if __name__ == "__main__":
    # Path to your .obj file
    mesh_file = "path/to/your/mesh.obj"  # Replace with actual path
    
    # Convert the mesh to point cloud
    point_cloud = mesh_to_point_cloud(mesh_file, points_per_triangle=3, use_poisson_sampling=False)
    
    # Visualize the point cloud
    visualize_point_cloud(point_cloud)
