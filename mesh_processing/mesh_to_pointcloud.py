import open3d as o3d

def mesh_to_point_cloud(mesh_file, points_per_triangle=3, use_poisson_sampling=False):
    """
    Convert a 3D mesh into a point cloud using uniform or Poisson disk sampling.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    num_triangles = len(mesh.triangles)
    num_points = num_triangles * points_per_triangle
    print(f"Mesh loaded with {num_triangles} triangles. Sampling {num_points} points.")
    
    if use_poisson_sampling:
        print("Using Poisson disk sampling...")
        return mesh.sample_points_poisson_disk(number_of_points=num_points)
    else:
        print("Using uniform sampling...")
        return mesh.sample_points_uniformly(number_of_points=num_points)
