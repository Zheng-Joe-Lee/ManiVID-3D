import mujoco
import numpy as np
import open3d as o3d
from typing import Tuple
import os
from tqdm import tqdm
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dmc
from point_cloud_generator import PointCloudGenerator
import hydra
import cv2 
from scipy.spatial.transform import Rotation
import torch
import pytorch3d.ops as torch3d_ops

class PointCloudDataCollector:
    def __init__(self, cfg_task, output_dir):

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        self.cfg_task = cfg_task
        
        self.env = dmc.make(self.cfg_task.task.task_name, self.cfg_task.frame_stack,
                                 self.cfg_task.action_repeat, self.cfg_task.seed, two_cam=True, img_size=self.cfg_task.img_size, use_depth=self.cfg_task.use_depth, 
                                 use_pc=self.cfg_task.use_pc, max_points=self.cfg_task.max_points, add_rgb=self.cfg_task.add_rgb, downsample_method=self.cfg_task.downsample_method, crop_dist=self.cfg_task.crop_dist)
        self.sim = self.env.physics
        
        self.pc_generator_track = PointCloudGenerator(sim=self.sim,
                                                cam_id='track_cam',
                                                img_height=256, 
                                                img_width=256)
        
        self.pc_generator_fix = PointCloudGenerator(sim=self.sim,
                                                cam_id='fix_cam',
                                                img_height=256, 
                                                img_width=256)
        
        self.object_low = self.env._task.object_low
        self.object_high = self.env._task.object_high
        # print(self.object_low)
        # print(self.object_high)
        self.DOF_SETTING = {
            'franka': [None, None, 0., None, 0., None, 0.],
            'ur5': [None, None, None, 0., 0., 0.],
            'xarm': [None, None, None, 0., 0., 0.],
            'airplay': [None, None, None, 0., None, 0.],
        }
        self.CAM_SETTING = {
            'airplay': {
                "initial_point": [-0.52, 0, 1.0],
                "circle_center": [0.20, 0, 0.95],
                "default_val": [0, 0, 1],
                "min_val": [-75, 0, 0.8],
                "max_val": [75, 7.5, 0.9],
            },
            'ur5': {
                "initial_point": [1.9, 0, 0.75],
                "circle_center": [0.6, 0.0, 0.2],
                "default_val": [0, 0, 1],
                "min_val": [-75, -12.5, 0.8],
                "max_val": [75, 7.5, 1.1],
            },
            'franka': {
                "initial_point": [1.9, 0, 0.75],
                "circle_center": [0.6, 0.0, 0.2],
                "default_val": [0, 0, 1],
                "min_val": [-75, -12.5, 0.8],
                "max_val": [75, 7.5, 1.1],
            },
            'franka_dual': {
                "initial_point": [1.90, -0.5, 0.75],
                "circle_center": [0.4, 0.35, 0.35],
                "default_val": [0, 0, 1],
                "min_val": [-30, -12.5, 0.8],
                "max_val": [30, 7.5, 1.1],
            }
        }
        self.init_pose = self.sim._mjcf_wrapper.init_pose['qpos']
        # print(self.init_pose)
        
        
        self.metadata = {
            "scenes": [],
            "rand_params_joint": [],
            "rand_params_object": [],
            "num_points_track": [],
            "num_points_fix": []
        }
    

    
    
    def collect_data(self, num_scenes: int = 100, num_view: int = 100):

        domain, task = self.cfg_task.task.task_name.split('_', 1)
        dof_setting = np.asarray(self.DOF_SETTING[domain])
        if task=='dual_dex':
            cam_setting = self.CAM_SETTING['franka_dual']
        else:
            cam_setting = self.CAM_SETTING[domain]
        # print(dof_setting)
        none_mask = dof_setting == None
        dof_idx = np.nonzero(none_mask)[0]
        # print(dof_idx)

        for scene_id in tqdm(range(num_scenes), desc="Collecting scenes"):          

            object_pos = np.random.uniform(low=self.object_low, high=self.object_high)
            # object_pos = 0
            joint_noise = np.random.uniform(-np.pi/18, np.pi/18, size = len(self.init_pose) -2)
            gripper_noise = np.random.uniform(0, 0.04)
            joint_noise = np.concatenate((joint_noise,np.array([gripper_noise, -gripper_noise])))

            scene_data = {
                    "scene_id": scene_id,
                    "views": [],
                    "object_pos": object_pos,
                    "joint_noise": joint_noise
                }
            
            for view_id in range(num_view):
                
                self._place_scene(joint_noise, object_pos, cam_setting)

                gt_cloud_track, org_cloud, org_rgb = self.pc_generator_track.generateCroppedPointCloud(crop_dist=self.cfg_task.crop_dist, to_world_cord=True)
                # print(org_cloud.shape)
                gt_cloud_fix, _, _ = self.pc_generator_fix.generateCroppedPointCloud(crop_dist=self.cfg_task.crop_dist, to_world_cord=True)
                # print(gt_cloud_fix.shape)
                org_cloud = self._downsample_points(org_cloud, self.cfg_task.max_points, 'fps')
                gt_cloud_track = self._downsample_points(gt_cloud_track, self.cfg_task.max_points, 'fps')
                gt_cloud_fix = self._downsample_points(gt_cloud_fix, self.cfg_task.max_points, 'fps')

                
                view_data = {
                    "view_id": view_id,
                    "org_cloud": org_cloud[:, :3],
                    "gt_cloud_track": gt_cloud_track[:, :3],
                    "gt_cloud_fix": gt_cloud_fix[:, :3]
                }
                scene_data["views"].append(view_data)
            
            self._save_scene(scene_id, scene_data)
            self.metadata["scenes"].append(scene_id)
            self.metadata["num_points_track"].append(len(gt_cloud_track))
            self.metadata["num_points_fix"].append(len(gt_cloud_fix))
            self.metadata["rand_params_joint"].append(joint_noise.tolist())
            self.metadata["rand_params_object"].append(object_pos.tolist())
        
        # 保存元数据
        self._save_metadata()

    def _downsample_points(self, point_cloud:np.ndarray, num_points:int, method:str='uniform'):
        """
        support different point cloud sampling methods
        point_cloud: (N, 6), xyz+rgb
        """
        if num_points == 'all': # use all points
            return point_cloud
        if point_cloud.shape[0] <= num_points:
            # pad with zeros
            point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], 6), dtype=np.float32)], axis=0)
            return point_cloud
        
        if method == 'uniform':
            # uniform sampling
            sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
            point_cloud = point_cloud[sampled_indices]
        elif method == 'fps':
            # fast point cloud sampling using torch3d
            point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
            num_points = torch.tensor([num_points]).cuda()
            # remember to only use coord to sample
            _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points)
            point_cloud = point_cloud.squeeze(0).cpu().numpy()
            point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
        else:
            raise NotImplementedError(f"point cloud sampling method {method} not implemented")

        return point_cloud
    
    def _place_scene(self, joint_noise, object_pos, cam_setting):
        domain, task = self.cfg_task.task.task_name.split('_', 1)
        self.sim.data.qpos[-len(self.init_pose):] =  self.init_pose + joint_noise
        if domain=='airplay':
            if task=='lift' or task=='reach':
                self.sim.set_freejoint_pos('object_anchor', object_pos, np.zeros(4))
            elif task=='button':
                self.sim.set_body_pos('buttonbox', object_pos)
            elif task=='laptop':
                init_angle = np.random.uniform(low=-0.15, high=-0.05)
                self.sim.set_body_pos('laptop', object_pos)
                self.sim.set_joint_pos('laptop_joint', init_angle)
            elif task=='picknplace':
                bottom_low=(0.00, 0.10, 0.75),
                bottom_high=(0.1, 0.2, 0.75),
                bottom_pos = np.random.uniform(low=bottom_low, high=bottom_high)
                self.sim.set_freejoint_pos('bowl_anchor', bottom_pos)
                self.sim.set_freejoint_pos('object_anchor', object_pos)
        elif domain=='ur5':
            if task=='drawer':
                self.sim.set_body_pos('drawer_base', object_pos)
            elif task=='reach_dex':
                self.sim.set_freejoint_pos('object_anchor', object_pos)
            elif task=='button_dex':
                self.sim.set_body_pos('buttonbox', object_pos)
                
        
        cam_id = self.sim.model.name2id('track_cam', 'camera')
        initial_point = np.array(cam_setting['initial_point'])
        circle_center = np.array(cam_setting['circle_center'])
        min_val = np.array(cam_setting['min_val'])
        max_val = np.array(cam_setting['max_val'])

        initial_offset = initial_point - circle_center
        initial_radius = np.linalg.norm(initial_offset)
        initial_dir = initial_offset / initial_radius

        azimuth = np.arctan2(initial_dir[1], initial_dir[0]) 
        elevation = np.arcsin(initial_dir[2])            

        delta_azimuth = np.radians(np.random.uniform(min_val[0], max_val[0]))  
        delta_elevation = np.radians(np.random.uniform(min_val[1], max_val[1]))  
        radius_scale = np.random.uniform(min_val[2], max_val[2])

        new_azimuth = azimuth + delta_azimuth
        new_elevation = np.clip(elevation + delta_elevation, -np.pi/2 + 0.1, np.pi/2 - 0.1)

        new_radius = initial_radius * radius_scale
        new_dir = np.array([
            np.cos(new_elevation) * np.cos(new_azimuth),
            np.cos(new_elevation) * np.sin(new_azimuth),
            np.sin(new_elevation)
        ])
        new_pos = circle_center + new_dir * new_radius


        self.sim.model.cam_pos[cam_id] = new_pos


        self.sim.forward()
    
    
    def _save_scene(self, scene_id: int, data: dict):
        scene_dir = os.path.join(self.output_dir, f"scene_{scene_id:04d}")
        os.makedirs(scene_dir, exist_ok=True)
        
        np.savez_compressed(
            os.path.join(scene_dir, "data.npz"),
            views=np.array(data["views"], dtype=object), 
            object_pos=data["object_pos"],
            joint_noise=data["joint_noise"]
        )
        
    
    def _save_metadata(self):
        with open(os.path.join(self.output_dir, "metadata.json"), 'w') as f:
            json.dump(self.metadata, f, indent=2)

@hydra.main(config_path='../cfgs', config_name='camera_aug_config_pc')
# @hydra.main(config_path='cfgs', config_name='camera_aug_config_backup')
def main(cfg):
    output_dir=os.path.join("TODO/ManiVID-3D-Anonymous/ManiVID_3D/viewnet/data", cfg.task.task_name)
    output_dir=os.path.join(output_dir, f"{cfg.max_points}")

    train_dir=os.path.join(output_dir, "train")
    test_dir=os.path.join(output_dir, "test")

    collector_train = PointCloudDataCollector(
        cfg_task=cfg,
        output_dir=train_dir
    )
    collector_test = PointCloudDataCollector(
        cfg_task=cfg,
        output_dir=test_dir
    )

    collector_train.collect_data(num_scenes=10, num_view=10)
    collector_test.collect_data(num_scenes=5, num_view=5)


if __name__ == '__main__':
    main()

    