from mesh_processing.mesh_to_pointcloud import mesh_to_point_cloud
from mesh_processing.visualization import visualize_point_cloud
from mesh_processing.file_io import save_point_cloud
if __name__ == "__main__":
    mesh_file = "./data/visage.obj"  
    
    point_cloud = mesh_to_point_cloud(mesh_file, points_per_triangle=3, use_poisson_sampling=False)
    output_cloud = "./data/visage_cloud.ply"
    save_point_cloud(point_cloud, output_cloud)
    visualize_point_cloud(point_cloud)
