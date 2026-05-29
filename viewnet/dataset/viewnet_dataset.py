import os
import json
import numpy as np
import torch
from tqdm import tqdm
import dm_control as dmc
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

class ViewNetDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, max_points: int = 1024):
        """
        点云数据集加载器
        
        参数:
            data_dir: 数据目录路径
            transform: 可选的数据变换
            max_points: 点云最大点数 (不足的补零，超出的下采样)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_points = max_points
        
        # 加载元数据
        with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)
        
        # 准备样本索引列表 (scene_id, view_id)
        self.samples = []
        for scene_id in self.metadata["scenes"]:
            scene_dir = os.path.join(data_dir, f"scene_{scene_id:04d}")
            data = np.load(os.path.join(scene_dir, "data.npz"), allow_pickle=True)
            num_views = len(data["views"])
            for view_id in range(num_views):
                self.samples.append((scene_id, view_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        scene_id, view_id = self.samples[idx]
        scene_dir = os.path.join(self.data_dir, f"scene_{scene_id:04d}")
        
        # 加载场景数据
        data = np.load(os.path.join(scene_dir, "data.npz"), allow_pickle=True)
        view_data = data["views"][view_id]
        
        # 获取点云数据
        gt_points_track = view_data["gt_cloud_track"]
        gt_points_fix = view_data["gt_cloud_fix"]
        
        # 获取原始点云 (相机坐标系)
        org_points = view_data["org_cloud"]
        
        # 获取场景参数
        object_pos = data["object_pos"]
        joint_noise = data["joint_noise"]
        
        # 转换为torch张量
        gt_points_track = torch.from_numpy(gt_points_track).float()
        gt_points_fix = torch.from_numpy(gt_points_fix).float()
        org_points = torch.from_numpy(org_points).float()
        object_pos = torch.from_numpy(object_pos).float()
        joint_noise = torch.from_numpy(joint_noise).float()
        
        # 应用变换 (如果有)
        if self.transform:
            points = self.transform(points)
            org_points = self.transform(org_points)
        
        return {
            "org_points": org_points,  # 相机坐标系点云
            "gt_points":{ # 世界坐标系点云
                "gt_cloud_track": gt_points_track,  
                "gt_cloud_fix": gt_points_fix,
            }
        }
    
def viewnet_dataloader(data_dir: str, batch_size: int = 32, shuffle: bool = True, 
                   num_workers: int = 4, max_points: int = 1024):
    """
    创建点云数据加载器
    
    参数:
        data_dir: 数据目录路径
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        max_points: 点云最大点数
    """
    dataset = ViewNetDataset(data_dir, max_points=max_points)
    
    # 自定义collate_fn处理变长点云
    def collate_fn(batch):
        # 初始化结果字典
        collated_batch = {
            "org_points": [],
            "gt_points": {
                "gt_cloud_track": [],
                "gt_cloud_fix": []
            }
        }
        
        # 收集所有样本
        for sample in batch:
            collated_batch["org_points"].append(sample["org_points"])
            collated_batch["gt_points"]["gt_cloud_track"].append(sample["gt_points"]["gt_cloud_track"])
            collated_batch["gt_points"]["gt_cloud_fix"].append(sample["gt_points"]["gt_cloud_fix"])
        
        # 堆叠张量
        collated_batch["org_points"] = torch.stack(collated_batch["org_points"])
        
        # 处理嵌套的gt_points
        for key in collated_batch["gt_points"]:
            collated_batch["gt_points"][key] = torch.stack(collated_batch["gt_points"][key])
        
        return collated_batch
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader