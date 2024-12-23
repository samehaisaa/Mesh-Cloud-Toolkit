import open3d as o3d

def visualize_point_cloud(point_cloud, color=[1, 0, 0]):
    """
    Visualize a point cloud with a specified uniform color.
    """
    point_cloud.paint_uniform_color(color)
    o3d.visualization.draw_geometries([point_cloud])
