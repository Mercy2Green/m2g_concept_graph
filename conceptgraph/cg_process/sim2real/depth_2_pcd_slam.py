import open3d as o3d
import numpy as np
import cv2

def load_image_and_depth(rgb_path, depth_path):
    # 读取深度图
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 读取RGB图像
    rgb_image = cv2.imread(rgb_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    return rgb_image, depth_image

def create_point_cloud(rgb_image, depth_image, camera_matrix):
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

    return pcd

def transform_point_cloud(pcd, transformation):
    pcd.transform(transformation)
    return pcd

# 相机内参矩阵，格式为[fx, 0, cx; 0, fy, cy; 0, 0, 1]
camera_matrix = np.array([[610.6340942382812, 0.0, 635.8567504882812], 
                          [0.0, 610.8860473632812, 355.1409912109375], 
                          [0.0, 0.0, 1.0]])

# 读取图像和深度图
rgb_image1, depth_image1 = load_image_and_depth("/home/lg1/peteryu_workspace/m2g_concept_graph/dataset/test_1/color_image/i_1.png",
                                                "/home/lg1/peteryu_workspace/m2g_concept_graph/dataset/test_1/depth_image/d_1.png")
rgb_image2, depth_image2 = load_image_and_depth("/home/lg1/peteryu_workspace/m2g_concept_graph/dataset/test_1/color_image/i_2.png",
                                                "/home/lg1/peteryu_workspace/m2g_concept_graph/dataset/test_1/depth_image/d_2.png")

# 创建点云
pcd1 = create_point_cloud(rgb_image1, depth_image1, camera_matrix)
pcd2 = create_point_cloud(rgb_image2, depth_image2, camera_matrix)

# 假设我们有两个点云的变换矩阵（来自激光雷达SLAM里程计）
t1 = np.array([
    [ 9.99999150e-01, -1.30226595e-03,  5.60731547e-05 , 9.42660415e-02],
    [ 1.30223135e-03,  9.99998964e-01,  6.12619348e-04 , 8.55091726e-03],
    [-5.68708900e-05, -6.12545808e-04,  9.99999811e-01,  8.35825052e-03],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]
])
t2 = np.array([
    [ 9.99993890e-01, -3.49036771e-03 , 1.94459508e-04 , 2.55896530e-01],
    [ 3.49020587e-03,  9.99993568e-01,  8.26505887e-04 , 7.53333084e-03],
    [-1.97343067e-04 ,-8.25822133e-04 , 9.99999640e-01 , 8.22413004e-03],
    [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]
])

# 变换点云到全局坐标系
pcd1_transformed = transform_point_cloud(pcd1, t1)
pcd2_transformed = transform_point_cloud(pcd2, t2)

# # 可视化单独的点云和变换后的点云
# print("Visualizing individual point clouds before transformation...")
# o3d.visualization.draw_geometries([pcd1, pcd2])

# print("Visualizing individual point clouds after transformation...")
# o3d.visualization.draw_geometries([pcd1_transformed, pcd2_transformed])

# # 合并点云
# pcd_combined = pcd1_transformed + pcd2_transformed

# # 可视化合并后的点云
# print("Visualizing combined point cloud...")
# o3d.visualization.draw_geometries([pcd_combined])


def register_point_clouds(source, target, initial_transformation):
    threshold = 0.02  # Distance threshold for ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

# Perform point cloud registration to refine the transformation
refined_t2 = register_point_clouds(pcd2, pcd1, t2)

# Apply the refined transformation
pcd2_refined = transform_point_cloud(pcd2, refined_t2)

# Combine point clouds
pcd_combined_refined = pcd1_transformed + pcd2_refined

# Visualize the refined combined point cloud
print("Visualizing refined combined point cloud...")
o3d.visualization.draw_geometries([pcd_combined_refined])