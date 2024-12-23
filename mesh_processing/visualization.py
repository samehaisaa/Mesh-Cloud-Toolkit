import open3d as o3d

def visualize_point_cloud(point_cloud, color=[1, 0, 0]):
        o3d.visualization.draw_geometries([point_cloud])
