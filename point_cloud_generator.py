# reference implementation: https://github.com/mattcorsaro1/mj_pc
# with personal modifications


import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from typing import List
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation

def random_pc_jitter_aug(x, sigma=0.001, clip=0.003):
    """
    Point cloud jitter augmentation using NumPy.
    
    Args:
        x: Input point cloud array of shape (B, N, 3+) where B is batch size,
           N is number of points, and 3+ means at least xyz coordinates.
        sigma: Standard deviation of the Gaussian noise.
        clip: Maximum absolute value to clip the noise.
    
    Returns:
        jittered_points: Jittered point cloud with same shape as input.
    """
    N, _ = x.shape
    
    # Generate Gaussian noise and clip it
    noise = np.clip(
        np.random.randn(N, 3) * sigma,
        a_min=-clip, 
        a_max=clip
    )
    
    # Add noise to xyz coordinates (preserving other features if any)
    jittered_points = x.copy()
    jittered_points[..., :3] += noise
    
    return jittered_points


def random_point_dropout(points, max_dropout_ratio=0.1):
    """
    随机点丢弃增强 (支持带/不带batch的输入)
    Args:
        points: 输入点云，可以是:
                - 带batch: (B, N, C)
                - 不带batch: (N, C)
        max_dropout_ratio: 最大丢弃比例
    Returns:
        与输入形状相同的点云（保持原始点数，用零填充）
    """
    original_shape = points.shape
    has_batch = len(original_shape) == 3
    
    # 统一转为带batch的形式处理
    if not has_batch:
        points = np.expand_dims(points, axis=0)  # (N, C) -> (1, N, C)
    
    B, N, C = points.shape
    
    # 为每个样本生成不同的丢弃比例
    dropout_ratios = np.random.rand(B) * max_dropout_ratio
    keep_ratios = 1 - dropout_ratios
    
    # 生成随机掩码
    keep_nums = (N * keep_ratios).astype(np.int32)
    random_values = np.random.rand(B, N)
    
    # 为每个样本排序并保留前keep_nums个点
    sorted_indices = np.argsort(random_values, axis=1)
    mask = np.zeros((B, N), dtype=np.bool_)
    for i in range(B):
        mask[i, sorted_indices[i, :keep_nums[i]]] = True
    
    # 应用掩码并用零填充丢弃的点
    dropped_points = np.zeros_like(points)
    dropped_points[mask] = points[mask]
    
    # 如果原始输入不带batch，去掉batch维度
    if not has_batch:
        dropped_points = dropped_points[0]  # (1, N, C) -> (N, C)
    
    return dropped_points

def random_point_rotate(x, max_rotation_angle=6.0, max_translation=0.05):
    """
    点云随机旋转+平移增强
    
    Args:
        x: 输入点云数组，形状为 (B, N, 3+) 
           B是batch大小，N是点数，3+表示至少包含xyz坐标
        sigma: 高斯噪声的标准差
        clip: 噪声裁剪的最大绝对值
        max_rotation_angle: 最大旋转角度(度)
    
    Returns:
        augmented_points: 增强后的点云，形状与输入相同
    """
    # 确保输入是3D坐标点云
    assert x.shape[-1] >= 3, "输入点云必须至少包含xyz坐标"
    
    augmented_points = x.copy()
    B = x.shape[0] if len(x.shape) == 3 else 1
    
    # 为每个batch生成随机旋转矩阵
    for i in range(B):
        current_points = augmented_points[i] if len(x.shape) == 3 else augmented_points
        # 生成随机旋转角度(绕x,y,z轴)
        angles = np.random.uniform(-max_rotation_angle, max_rotation_angle, 3)
        
        # 创建旋转对象并应用
        rotation = Rotation.from_euler('xyz', angles, degrees=True)
        current_points[..., :3] = rotation.apply(current_points[..., :3])
    
        translation = np.random.uniform(-max_translation, max_translation, 3)
        current_points[..., :3] += translation

        if len(x.shape) == 3:
            augmented_points[i] = current_points
        else:
            augmented_points = current_points
    
    return augmented_points






"""
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""
def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# 
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""
class PointCloudGenerator(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self, sim, cam_id, img_height=84, img_width=84):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        # this should be aligned with rgb
        self.img_width = img_width
        self.img_height = img_height
        
        if not isinstance(cam_id, int):
            cam_id = sim.model.name2id(cam_id, 'camera')
        
        self.cam_id = cam_id
        
        # List of camera intrinsic matrices
        # self.cam_mats = []
        
        # compute camera intrinsic matrics
        # if not isinstance(cam_id, int):
        #     cam_id = self.sim.model.camera_name2id(self.cam_id)
        fovy = math.radians(self.sim.model.cam(self.cam_id).fovy[0])
        f = self.img_height / (2 * math.tan(fovy / 2))
        cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
        self.cam_mat = cam_mat

    def generateCroppedPointCloud(self, save_img_dir=None, crop_dist=None, to_world_cord=True):
        # o3d_clouds = []
        # cam_poses = []
        # depths = []
        
        # for cam_i in range(len(self.cam_names)):
        # Render and optionally save image from camera corresponding to cam_id
        color_img, depth = self.captureImage(self.cam_id, capture_depth=True)
        # depths.append(depth)
        # If directory was provided, save color and depth images
        #    (overwriting previous)
        if save_img_dir != None:
            self.saveImg(depth, save_img_dir, "depth_test_" + str(self.cam_id))
            self.saveImg(color_img, save_img_dir, "color_test_" + str(self.cam_id))

        # convert camera matrix and depth image to Open3D format, then
        #    generate point cloud
        
        od_cammat = cammat2o3d(self.cam_mat, self.img_width, self.img_height)
        od_depth = o3d.geometry.Image(depth)
        
        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)

        # Compute world to camera transformation matrix
        # cam_body_id = self.sim.model.cam_bodyid[self.cam_id]
        cam_pos = self.sim.data.cam_xpos[self.cam_id]
        c2b_r = rotMatList2NPRotMat(self.sim.data.cam_xmat[self.cam_id])
        # print(self.sim.model.cam_mat0)
        # print(c2b_r)

        b2w_r = quat2Mat([0, 1, 0, 0])
        c2w_r = np.matmul(c2b_r, b2w_r)
        # c2w_r = c2b_r
        c2w = posRotMat2Mat(cam_pos, c2w_r)
        # print(c2w)
        
        
        # # 设置点坐标

        # combined_cloud_colors = np.asarray(combined_cloud.colors)  # Get the colors, ranging [0,1].
        cloud_points = np.asarray(o3d_cloud.points)
        cloud_colors = color_img.reshape(-1, 3) # range [0, 255]

        # 距离裁剪前点云可视化

        # 如果有设置最大深度，则进行裁剪
        if crop_dist is not False:
            # 计算每个点的深度（z坐标）
            dist = np.linalg.norm(cloud_points, axis=1)
            # 创建掩码，保留深度小于max_depth的点
            mask = dist < crop_dist
            # 应用掩码
            cloud_points = cloud_points[mask]
            cloud_colors = cloud_colors[mask]

        pcd = o3d.geometry.PointCloud()
        # # 设置点坐标
        pcd.points = o3d.utility.Vector3dVector(cloud_points)
        
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=200)
        all_indices = np.arange(len(pcd.points))
        inlier_set = set(inliers)
        outliers = [idx for idx in all_indices if idx not in inlier_set]
        cloud_points = cloud_points[outliers]
        cloud_colors = cloud_colors[outliers]

        pcd = o3d.geometry.PointCloud()
        pcd_org = o3d.geometry.PointCloud()
        # # 设置点坐标
        pcd.points = o3d.utility.Vector3dVector(cloud_points)
        pcd_org.points = o3d.utility.Vector3dVector(cloud_points)
        
        org_cloud = pcd_org
        transformed_cloud = pcd.transform(c2w) 

        


        # 合并点和颜色
        transformed_cloud_points = np.asarray(transformed_cloud.points)
        transformed_combined_cloud = np.concatenate((transformed_cloud_points, cloud_colors), axis=1)

        org_cloud_points = np.asarray(org_cloud.points)
        org_combined_cloud = np.concatenate((org_cloud_points, cloud_colors), axis=1)

        
        if to_world_cord:
            return transformed_combined_cloud, org_combined_cloud, color_img
        else:
            return org_combined_cloud, depth


     
    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    # def depthimg2Meters(self, depth):
    #     extent = self.sim.model.stat.extent
    #     near = self.sim.model.vis.map.znear * extent
    #     far = self.sim.model.vis.map.zfar * extent
    #     image = near / (1 - depth * (1 - near / far))
    #     return image

    def add_depth_noise(self, depth, depth_dependent_noise=True, gaussion_noise_scale=0.01, depth_noise_scale=0.05):
        gaussion_noise = np.random.normal(0, gaussion_noise_scale, depth.shape)
        
        if depth_dependent_noise:
            depth_scale = depth_noise_scale * np.abs(depth)
            depth_noise = np.random.normal(0, depth_scale, depth.shape)
            noisy_depth = depth + gaussion_noise + depth_noise
        else:
            noisy_depth = depth + gaussion_noise
        
        noisy_depth = cv2.GaussianBlur(noisy_depth, (7, 7), 1)

        return noisy_depth

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, camera_name, capture_depth=True):
        # rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=camera_name, depth=capture_depth, device_id=device_id)
        img = self.sim.render(height=self.img_height, width=self.img_width, camera_id=self.cam_id, depth=False)

        if capture_depth:
            depth = self.sim.render(height=self.img_height, width=self.img_width, camera_id=self.cam_id, depth=True)
            depth_max = 3
            depth[depth >= depth_max] = depth_max
            # depth_convert = self.verticalFlip(self.add_depth_noise(depth))
            # depth = self.verticalFlip(depth)
            # # depth_convert = self.depthimg2Meters(depth_convert)
            # img = self.verticalFlip(img)
            return img.copy(), depth.copy()
        else:
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img/img.max()*255
        normalized_image = normalized_image.astype(np.uint8)
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".jpg")
