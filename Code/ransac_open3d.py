import open3d as o3d
import numpy as np
import random

# Read the point cloud
pcd_point_cloud = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)
# function to visualize the point cloud

# Define the parameters for RANSAC
num_iterations = 1000  # Number of RANSAC iterations
distance_threshold = 0.03  # Inlier threshold for plane fitting
inlier_ratio = 0.85  # Minimum inlier ratio for pya valid plane

best_plane = None
best_inliers = []
best_num_inliers = 0

# Convert Open3D point cloud to a NumPy array
points = np.asarray(pcd.points)

for _ in range(num_iterations):
    # Randomly select 3 points
    random_indices = random.sample(range(len(points)), 3)
    sample_points = points[random_indices]

    # Compute the plane model using the selected points
    v1 = sample_points[1] - sample_points[0]
    v2 = sample_points[2] - sample_points[0]
    normal = np.cross(v1, v2)
    d = -np.dot(normal, sample_points[0])

    # Calculate the distance from the plane for all points
    distances = np.abs(np.dot(points, normal) + d) / np.linalg.norm(normal)

    # Count inliers
    inliers = np.where(distances < distance_threshold)[0]
    num_inliers = len(inliers)

    # Update the best model if we found a better one
    if num_inliers > best_num_inliers:
        best_num_inliers = num_inliers
        best_inliers = inliers
        best_plane = (normal, d)

    # Check for early termination
    if num_inliers > len(points) * inlier_ratio:
        break

# Fit the best model to the inliers

inlier_cloud = pcd.select_by_index(best_inliers)
red = np.array([1, 0, 0])
inlier_cloud.paint_uniform_color(red)
# Visualize the point cloud with the inliers
#mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(inlier_cloud, depth=6, width=0, scale=0.9, linear_fit=True)
#mesh.paint_uniform_color(red)

o3d.visualization.draw_geometries([pcd], 
                                zoom=1,
                                front=[0.4257, -0.2125, -0.8795],
                                lookat=[2.6172, 2.0475, 1.532],
                                up=[-0.0694, -0.9768, 0.2024])

# best_normal, best_d = best_plane
# plane_size = 1 # Adjust the size of the plane as needed
# plane_mesh = o3d.geometry.TriangleMesh.create_box(plane_size, plane_size, 0.01)

# # Rotate the plane to align with the best-fit normal
# #best_plane_model = o3d.geometry.Plane(np.array([best_normal[0], best_normal[1], best_normal[2], best_d]))
# red_color = [1.0, 0.0, 0.0]
# plane_mesh.paint_uniform_color(red_color)
# # Visualize the best-fit plane
# #best_plane_model.paint_uniform_color([1, 0, 0])
# o3d.visualization.draw_geometries(  [pcd, plane_mesh],
#                                     zoom=1,
#                                     front=[0.4257, -0.2125, -0.8795],
#                                     lookat=[2.6172, 2.0475, 1.532],
#                                     up=[-0.0694, -0.9768, 0.2024])

# print("Best-fit plane equation: {}x + {}y + {}z + {} = 0".format(best_normal[0], best_normal[1], best_normal[2], best_d))