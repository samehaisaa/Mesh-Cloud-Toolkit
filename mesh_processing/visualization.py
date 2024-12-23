import open3d as o3d

def visualize_point_cloud(point_cloud):
        o3d.visualization.draw_geometries([point_cloud])
