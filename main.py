import open3d as o3d

def mesh_to_point_cloud(mesh_file, points_per_triangle=2, use_poisson_sampling=False):
    """
    Convert a 3D mesh into a point cloud using uniform or Poisson disk sampling.

    Parameters:
        mesh_file (str): Path to the .obj mesh file.
        points_per_triangle (int): Number of points to sample per triangle (used for uniform sampling).
        use_poisson_sampling (bool): If True, use Poisson disk sampling instead of uniform sampling.

    Returns:
        point_cloud (o3d.geometry.PointCloud): The resulting point cloud.
    """
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    
    # Ensure the mesh has vertex normals
    mesh.compute_vertex_normals()
    
    # Estimate total number of triangles
    num_triangles = len(mesh.triangles)
    num_points = num_triangles * points_per_triangle  # Calculate total points to sample
    
    print(f"Mesh loaded with {num_triangles} triangles. Sampling {num_points} points.")

    # Perform sampling
    if use_poisson_sampling:
        print("Using Poisson disk sampling...")
        point_cloud = mesh.sample_points_poisson_disk(number_of_points=num_points)
    else:
        print("Using uniform sampling...")
        point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
    
    return point_cloud

def visualize_point_cloud(point_cloud, color=[1, 0, 0]):
    """
    Visualize a point cloud with a specified uniform color.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The point cloud to visualize.
        color (list): RGB values for uniform color (default is red).
    """
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    # Path to your .obj file
    mesh_file = "./visage.obj"  # Replace with the actual path
    
    # Convert the mesh to a point cloud
    point_cloud = mesh_to_point_cloud(mesh_file, points_per_triangle=3, use_poisson_sampling=False)
    
    # Visualize the resulting point cloud
    visualize_point_cloud(point_cloud)
    
    # Save the point cloud to a .ply file for further use
    output_file = "visage_cloud.ply"
    o3d.io.write_point_cloud(output_file, point_cloud)
    print(f"Point cloud saved to {output_file}")
