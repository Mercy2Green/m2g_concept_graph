import open3d as o3d
import numpy as np
import cv2

# 读取深度图
depth_image = cv2.imread("/home/lg1/peteryu_workspace/m2g_concept_graph/dataset/test_1/depth_image/d_1.png", cv2.IMREAD_UNCHANGED)

# 读取RGB图像
rgb_image = cv2.imread("/home/lg1/peteryu_workspace/m2g_concept_graph/dataset/test_1/color_image/i_1.png")
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

# 相机内参矩阵，格式为[fx, 0, cx; 0, fy, cy; 0, 0, 1]
camera_matrix = np.array([[610.6340942382812, 0.0, 635.8567504882812], 
                          [0.0, 610.8860473632812, 355.1409912109375], 
                          [0.0, 0.0, 1.0]])

# 深度图转换为点云
height, width = depth_image.shape
i, j = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
z = depth_image / 1000.0  # 假设深度图的单位是毫米，转换为米
x = (i - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
y = (j - camera_matrix[1, 2]) * z / camera_matrix[1, 1]

# 将点云添加到Open3D格式
points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 添加颜色信息
colors = rgb_image.reshape(-1, 3) / 255.0  # 将颜色值归一化到[0, 1]
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可以选择保存点云到PLY文件
# o3d.io.write_point_cloud("point_cloud.ply", pcd)

# 可视化点云
o3d.visualization.draw_geometries([pcd])