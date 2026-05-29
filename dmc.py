# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
from dm_control.utils import rewards
import re
from envs.tasks import *
from envs.control import MocapCtrlWrapper, JointCtrlWrapper, JointIKCtrlWrapper, DofWrapper, BinaryGripperWrapper, DualJointCtrlWrapper, OriginalJointCtrlWrapper, ArmJointCtrlWrapper, DualOpenPickJointCtrlWrapper, OldJointCtrlWrapper
from point_cloud_generator import PointCloudGenerator
import torch
import copy
import utils
import cv2
import torch.nn.functional as F
import pytorch3d.ops as torch3d_ops
import open3d as o3d
from algos.viewnet import ViewNet

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)
        

class StateExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    state: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)
        


class AugExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    aug_observation: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)



class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)



class StateFrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        
        state_num = 0
        for key in env.observation_spec().keys():
            state_num += env.observation_spec()[key].shape[0]
        print('state_num:', state_num)
        self._obs_spec = specs.BoundedArray(shape=np.array([num_frames * state_num]),
                                            dtype=np.float32,
                                            minimum=-np.inf,
                                            maximum=np.inf,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)


    def _extract_state(self, time_step):
        
        for i, key in enumerate(time_step.observation.keys()):
            if i == 0:
                state = time_step.observation[key]
            else:
                state = np.concatenate([state, time_step.observation[key]], axis=0)
        return np.float32(state.copy())


    def reset(self):
        time_step = self._env.reset()
        state = self._extract_state(time_step)
        for _ in range(self._num_frames):
            self._frames.append(state)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        state = self._extract_state(time_step)
        self._frames.append(state)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env, randomize):
        self._env = env
        self._randomize = randomize
        
    def transform(self, action):
        scale =  np.array([6.28319,  6.28319, 3.1415 , 6.28319, 6.28319, 6.28319, 127.5])
        orig_minimum = np.array([-6.28319, -6.28319, -3.1415 , -6.28319, -6.28319, -6.28319,0.])
        minimum = np.array([-1.])
        new_action = orig_minimum + scale * (action - minimum)
        return new_action.astype(self._env.action_spec.dtype, copy=False)

    def reset(self):
        if self._randomize:
            self._env.randomize()
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class StateExtendedTimeStepWrapper(ExtendedTimeStepWrapper):
    
    def __init__(self, env, randomize):
        super().__init__(env, randomize)
    
    def _augment_time_step(self, time_step, action=None, state=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return StateExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0,
                                state=state)
    
    def reset(self):
        if self._randomize:
            self._env.randomize()
        time_step = self._env.reset()
        state = np.concatenate([self._env._state_frames], axis=0)
        # import ipdb;ipdb.set_trace()
        return self._augment_time_step(time_step, state=state)
        
    def step(self, action):
        time_step = self._env.step(action)
        state = np.concatenate([self._env._state_frames], axis=0)
        return self._augment_time_step(time_step, action, state)


class CameraViewWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, cam_id, height=84, width=84, depth=False):
        self._env = env

        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)

        self._height = height
        self._width = width
        self._cam_id = cam_id
        self._depth = depth
        self.channel = 3
        
        extra_channel = 1 if self._depth else 0

        self._observation_spec = specs.BoundedArray(
            shape=np.array([self.channel * num_frames + extra_channel, height, width]),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        
    def _get_pixels(self, **render_kwargs):
        pixels = self._env.physics.render(**render_kwargs)
        return pixels.transpose(2, 0, 1).copy()
    
    def _add_depth_noise(self, depth, depth_dependent_noise=True, gaussion_noise_scale=0.01, depth_noise_scale=0.05):
        gaussion_noise = np.random.normal(0, gaussion_noise_scale, depth.shape)
        
        if depth_dependent_noise:
            depth_scale = depth_noise_scale * np.abs(depth)
            depth_noise = np.random.normal(0, depth_scale, depth.shape)
            noisy_depth = depth + gaussion_noise + depth_noise
        else:
            noisy_depth = depth + gaussion_noise
        
        noisy_depth = cv2.GaussianBlur(noisy_depth, (7, 7), 1)

        return noisy_depth

    def _get_depth(self, **render_kwargs):
        depth = self._env.physics.render(**render_kwargs)
        depth = self._add_depth_noise(depth)
        depth_max = 2
        depth[depth >= depth_max] = depth_max
        # depth[depth >= depth_max] = depth.max()
        depth = 255 * (depth - depth.min()) / (depth.max() - depth.min())
        depth = np.clip(depth, 0, 255).astype(np.uint8)
        return depth[None].copy()
    
    def _get_obs(self):
        obs = self._get_pixels(height=self._height, width=self._width, camera_id=self._cam_id)
        return obs

    def _transform_obs(self, time_step):
        if isinstance(time_step.observation, np.ndarray):
            obs = np.concatenate([time_step.observation, *self._frames], axis=0)
        else:
            obs = np.concatenate(list(self._frames), axis=0)
        if self._depth:
            depth_obs = self._get_depth(height=self._height, width=self._width, camera_id=self._cam_id, depth=True)
            obs = np.concatenate([obs, depth_obs], axis=0)
        return time_step._replace(observation=obs)
    
    def reset(self):
        time_step = self._env.reset()
        obs = self._get_obs()
        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._transform_obs(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._get_obs()
        self._frames.append(obs)
        return self._transform_obs(time_step)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    
class StateCameraViewWrapper(CameraViewWrapper):
    def __init__(self, env, num_frames, cam_id, height=84, width=84, depth=False):
        super().__init__(env, num_frames, cam_id, height, width, depth)
        self._state_frames = deque([], maxlen=num_frames)
        self._state_spec = specs.BoundedArray(
            shape=np.array([num_frames, env.state_num[0]]),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name='state'
        )
    
        
    def _get_state(self):
        state = self._env.get_ctrl_qpos()
        return state


    def reset(self):
        time_step = self._env.reset()
        obs = self._get_obs()
        state = self._get_state()
        for _ in range(self._num_frames):
            self._frames.append(obs)
            self._state_frames.append(state)
        return self._transform_obs(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._get_obs()
        state = self._get_state()
        self._frames.append(obs)
        self._state_frames.append(state)
        return self._transform_obs(time_step)

    def state_spec(self):
        return self._state_spec

class MultiDepthCameraViewWrapper(CameraViewWrapper):
    
    def __init__(self, env, num_frames, cam_id, height=84, width=84, depth=False):
        super().__init__(env, num_frames, cam_id, height, width, depth)
        self._observation_spec = specs.BoundedArray(
            shape=np.array([3 * num_frames + 3, height, width]),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        
    def _transform_obs(self, time_step):
        if isinstance(time_step.observation, np.ndarray):
            obs = np.concatenate([time_step.observation, *self._frames], axis=0)
        else:
            obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)
    
    def _get_obs(self):
        obs = self._get_pixels(height=self._height, width=self._width, camera_id=self._cam_id)
        if self._depth:
            depth_obs = self._get_depth(height=self._height, width=self._width, camera_id=self._cam_id, depth=True)
            obs = np.concatenate([obs, depth_obs], axis=0)
        return obs
    


class PointCloudViewWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, cam_id, height=84, width=84, max_points=1024, add_rgb=False, downsample_method = 'uniform', crop_dist=None, device='cpu', eval=False):
        """
        点云数据处理环境包装器
        
        参数:
            env: 原始环境
            num_frames: 堆叠的帧数
            max_points: 每帧最大点数(超过将下采样)
            add_rgb: 是否包含RGB颜色信息
        """
        self._env = env
        self._num_frames = num_frames
        self._max_points = max_points
        self._add_rgb = add_rgb
        self._cam_id = cam_id
        self._height = height
        self._width = width
        self._downsample_method = downsample_method
        self.pc_generator = PointCloudGenerator(sim=self._env.physics,
                                                cam_id=cam_id,
                                                img_height=height, 
                                                img_width=width)
        self.crop_dist = crop_dist
        self.eval = eval
        self.pre_scale = None
        self.pre_translation = None
        self.sim3 = None
        self.viewnet_path ='TODO'
        self.device = device
        
        # 计算每个点的特征维度
        self._point_feature_dim = 3  # xyz坐标
        if add_rgb:
            self._point_feature_dim += 3
        
        # 初始化帧队列
        self._frames = deque([], maxlen=num_frames)
        
        # 定义观察规范
        self._observation_spec = specs.BoundedArray(
            shape=(num_frames*max_points, self._point_feature_dim),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name='observation'
        )
    
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

    def _get_depth(self, **render_kwargs):
        depth = self._env.physics.render(**render_kwargs)
        depth = self._add_depth_noise(depth)
        depth_max = 2
        depth[depth >= depth_max] = depth_max
        # depth[depth >= depth_max] = depth.max()
        depth = 255 * (depth - depth.min()) / (depth.max() - depth.min())
        depth = np.clip(depth, 0, 255).astype(np.uint8)
        return depth[None].copy()
    
    def _get_point_cloud(self, crop_dist=None):
        """从环境中获取点云数据"""
        points_w, points_c, _ = self.pc_generator.generateCroppedPointCloud(crop_dist=crop_dist)

        # 确保是numpy数组
        points_w = np.array(points_w, dtype=np.float32)
        points_c = np.array(points_c, dtype=np.float32)
        
        return points_w, points_c
    
    def _normalize_points_fix(self, points, max_val, min_val):
        # 确保max_val和min_val是numpy数组
        max_val = np.asarray(max_val)
        min_val = np.asarray(min_val)
        
        center = (min_val + max_val) / 2

        # 只归一化xyz坐标（前3列）
        # 防止除以零，添加一个小量
        range_val = np.max(max_val - min_val)
        if range_val == 0:
            range_val = 1e-8  # 防止除以零
        
        # 归一化xyz坐标
        normalized_xyz = (points[:, :3] - center) / (range_val / 2.0)

        # 检查哪些点在 [-1, 1] 范围内
        valid_mask = np.all((normalized_xyz >= -1.2) & (normalized_xyz <= 1.2), axis=1)
        
        # 保留有效点
        normalized_pc = np.hstack([normalized_xyz, points[:, 3:]])[valid_mask]


        return normalized_pc
    
    def _normalize_points_limits(self, points):
        # 确保points是numpy数组
        points = np.asarray(points)
    
        # 计算点云xyz坐标的实际最大最小值
        min_val = np.min(points[:, :3], axis=0)  # 各维度最小值 [3,]
        max_val = np.max(points[:, :3], axis=0)  # 各维度最大值 [3,]
        # low_percentile=2
        # high_percentile=98

        # min_val = np.percentile(points[:, :3], low_percentile, axis=0)
        # max_val = np.percentile(points[:, :3], high_percentile, axis=0)
        
        center = (min_val + max_val) / 2

        # 只归一化xyz坐标（前3列）
        # 防止除以零，添加一个小量
        range_val = np.max(max_val - min_val)
        if range_val == 0:
            range_val = 1e-8  # 防止除以零
        
        # print(range_val)
        # 归一化xyz坐标
        normalized_xyz = (points[:, :3] - center) / (range_val / 2.0)

        # 检查哪些点在 [-1, 1] 范围内
        valid_mask = np.all((normalized_xyz >= -1.2) & (normalized_xyz <= 1.2), axis=1)
        
        # 保留有效点
        normalized_pc = np.hstack([normalized_xyz, points[:, 3:]])[valid_mask]


        return normalized_pc
    
    def _normalize_points_viewnet(self, points, max_val, min_val, reset):

        # 确保points是numpy数组
        points = np.asarray(points)
        if reset:
            # 计算点云xyz坐标的质心（均值）
            centroid = np.mean(points[:, :3], axis=0)  # 各维度均值 [3,]

            # 确保max_val和min_val是torch张量并具有正确的形状
            max_val = np.asarray(max_val)
            min_val = np.asarray(min_val)
            
            # 计算中心点和范围
            range_val = np.max(max_val - min_val)
            if range_val == 0:
                range_val = 1e-8  # 防止除以零
            scale = range_val/2
        else:
            centroid = self.pre_translation
            scale = self.pre_scale
        
        # 归一化处理
        normalized_xyz = (points[:, :3] - centroid) / scale
        normalized_pc = np.hstack([normalized_xyz, points[:, 3:]])

        return normalized_pc, centroid, scale
    
    def _process_point_cloud(self, points):
        """处理点云数据"""
        # 归一化到[-1, 1]
        max_val = [0.7, 0.1, 1.5] #for airplay
        min_val = [0, -0.1, 0.7] #for airplay
        
        # points = self._normalize_points_fix(points, max_val, min_val)
        points = self._normalize_points_limits(points)
        # 下采样到固定大小
        points = self._downsample_points(points, self._max_points, self._downsample_method)
        if not self._add_rgb:
            points = points[..., :3]
        # print(self._add_rgb)
        # print(points.shape)
        return points
    
    def _process_point_cloud_eval(self, points, reset=False):
        """处理点云数据"""
        # 归一化到[-1, 1]
        max_val = [0.7, 0.1, 1.5] #for airplay
        min_val = [0, -0.1, 0.7] #for airplay
        
        if reset:
            points, centroid, scale = self._normalize_points_viewnet(points, max_val, min_val, reset)
            self.pre_translation = centroid
            self.pre_scale = scale
        else:
            points = self._normalize_points_viewnet(points, max_val, min_val, reset)

        # 下采样到固定大小
        points = self._downsample_points(points, self._max_points, self._downsample_method)
        if not self._add_rgb:
            points = points[..., :3]
        # print(self._add_rgb)
        # print(points.shape)
        return points
    
    def _transform_obs(self, time_step):
        """转换观察数据"""
        
        if isinstance(time_step.observation, np.ndarray):
            obs = np.concatenate([time_step.observation, *self._frames], axis=0)
        else:
            obs = np.concatenate(list(self._frames), axis=0)

        return time_step._replace(observation=obs)
    
    def reset(self):
        """重置环境"""
        time_step = self._env.reset()
        points_w, points_c = self._get_point_cloud(crop_dist=self.crop_dist)
        if not self.eval: 
            points = points_w
            processed_points = self._process_point_cloud(points)
        else:
            points = points_c
            processed_points = self._process_point_cloud_eval(points, reset=True)

            if not hasattr(self, 'viewnet_model'):
                model = ViewNet(use_rgb=self._add_rgb).to(self.device)
                checkpoint = torch.load(self.viewnet_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.viewnet_model = model

            input_points = torch.from_numpy(processed_points).float().to(self.device)
            if len(input_points.shape) == 2:
                input_points = input_points.unsqueeze(0)  # 添加batch维度
            # 预测Sim3变换
            with torch.no_grad():
                sim3, _ = self.viewnet_model(input_points)
            self.sim3 = sim3.cpu().numpy()[0]  # 去掉batch维度
            # 提取旋转、缩放和平移部分
            R = self.sim3[:3, :3]  # 旋转和缩放
            t = self.sim3[:3, 3]    # 平移
            
            # 变换点云（只变换xyz坐标）
            points_xyz = points[:, :3]
            transformed_xyz = (R @ points_xyz.T).T + t
            points[:, :3] = transformed_xyz
            
            # 重新处理变换后的点云
            processed_points = self._process_point_cloud(points)


        # 检查特征维度是否匹配
        expected_dim = 3
        if self._add_rgb:
            expected_dim += 3
            
        if processed_points.shape[1] != expected_dim:
            raise ValueError(f"点云特征维度不匹配。期望:{expected_dim}, 实际:{processed_points.shape[1]}")

        
        # 初始化帧队列
        for _ in range(self._num_frames):
            self._frames.append(processed_points)
        
        return self._transform_obs(time_step)
    
    def step(self, action):
        """执行动作"""
        time_step = self._env.step(action)
        points_w, points_c = self._get_point_cloud(crop_dist=self.crop_dist)
        if not self.eval: 
            points = points_w
            processed_points = self._process_point_cloud(points)
        else:
            points = points_c
            processed_points = self._process_point_cloud_eval(points, reset=True)
            # 提取旋转、缩放和平移部分
            R = self.sim3[:3, :3]  # 旋转和缩放
            t = self.sim3[:3, 3]    # 平移
            
            # 变换点云（只变换xyz坐标）
            points_xyz = points[:, :3]
            transformed_xyz = (R @ points_xyz.T).T + t
            points[:, :3] = transformed_xyz
            
            # 重新处理变换后的点云
            processed_points = self._process_point_cloud(points)
        # pcd = o3d.geometry.PointCloud()
        # # 设置点坐标
        # pcd.points = o3d.utility.Vector3dVector(processed_points[:, :3])

        # 检查特征维度是否匹配
        expected_dim = 3
        if self._add_rgb:
            expected_dim += 3
            
        if processed_points.shape[1] != expected_dim:
            raise ValueError(f"点云特征维度不匹配。期望:{expected_dim}, 实际:{processed_points.shape[1]}")
        
        # 更新帧队列
        self._frames.append(processed_points)
        
        return self._transform_obs(time_step)
    
    def observation_spec(self):
        """观察规范"""
        return self._observation_spec
    
    def action_spec(self):
        """动作规范"""
        return self._env.action_spec()
    
    def __getattr__(self, name):
        """转发未实现的属性访问到原始环境"""
        return getattr(self._env, name)

    

class ActionClipWrapper(dm_env.Environment):
    def __init__(self, env, minimum, maximum):
        self._env = env
        
        wrapped_action_spec = self._env.action_spec()
        self._action_min = np.array([new_min or orig_min for new_min, orig_min in zip(minimum, wrapped_action_spec.minimum)])
        self._action_max = np.array([new_max or orig_max for new_max, orig_max in zip(maximum, wrapped_action_spec.maximum)])
        


    def step(self, action):
        # TODO: Clip action or Set action space ?
        action = np.clip(action, self._action_min, self._action_max)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed, 
         img_size=84, randomize=False, use_embedding=False, use_aug=True, two_cam=False, use_depth=False, control='joint', use_state=False, 
         use_pc=False, max_points=1024, add_rgb=False, downsample_method='uniform', crop_dist=None, device='cpu', eval=False):
    
    if re.match('^anymal', name):
        name_list = name.split('_')
        domain = name_list[0] + '_' + name_list[1]
        task = name_list[2]
    else:
        domain, task = name.split('_', 1)
        # overwrite cup to ball_in_cup
        domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    elif domain in ['ur5', 'franka', 'xarm', 'airplay']:
        # print("Available environments:", [k for k in globals().keys() if not k.startswith('_')])
        env = globals()[name]()
        pixels_key = 'pixels'

        JOINT_NUM = {
            'franka': 7,
            'ur5': 6,
            'xarm': 6,
            'airplay': 6,
        }
        DOF_SETTING = {
            'franka': [None, None, 0., None, 0., None, 0.],
            'ur5': [None, None, None, 0., 0., 0.],
            'xarm': [None, None, None, 0., 0., 0.],
            'airplay': [None, None, None, 0., None, 0.],
        }
        if control == 'mocap':
            arm_dof = 6
            arm_dof_setting = [None, None, None, 0., 0., 0.]
            if domain == 'franka':
                env = MocapCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), tcp_min=np.array([0, -0.45, 0]), tcp_max=np.array([1.0, 0.45, 0.6]))
            else:
                env = MocapCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
        elif control == 'joint':
            arm_dof = JOINT_NUM[domain]
            arm_dof_setting = DOF_SETTING[domain]
            if 'dual' in task:
                env = DualOpenPickJointCtrlWrapper(env, action_min=-np.ones(14), action_max=np.ones(14))
                # env = DualJointCtrlWrapper(env, action_min=-np.ones(14), action_max=np.ones(14), moving_average=1, action_scale=0.025)
            elif task == 'bowl_dex':
                env = JointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), moving_average=0.4, action_scale=0.04)
            elif task == 'close_dex':
                env = ArmJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
            # elif task == 'dual_dex':
            #     arm_dof = 7
            #     arm_dof_setting = [None, None, 0., None, 0., None, 0.]
            #     dual_arm_dof = 7
            #     env = DualJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), moving_average=0.4, dual_action_min=-np.ones(dual_arm_dof), dual_action_max=np.ones(dual_arm_dof))
            #elif domain == 'airplay' and task == 'lift':
            #    env = OldJointCtrlWrapper(env, action_min=np.array([-1, -1, -0.087, -2.96, -1.74, -3.14]), action_max=np.array([1, 0.17, 3.14, 2.96, 1.74, 3.14]))
            else:
                # env = OriginalJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
                env = OldJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
        
        if domain == 'xarm' and task == 'bowl_dex':
            env = DofWrapper(env, arm_dof_setting + [0, None, 0.4, 0.6] * 3 + [1.8, 0.2, -0.4, 0.5])
        elif domain == 'xarm' and task == 'close_dex':
            env = DofWrapper(env, [0., None, None, 0., None, 0.] + [0, 0.1, 0.3, 0.7] * 3 + [0.2, 0, 0, 0.2])
        elif task == 'lift_dex_cube' or task == 'bowl_dex':
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
        elif task == 'lift_dex_cup':
            arm_dof_setting[5] = 0.
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
        elif task == 'pour_dex':
            env = DofWrapper(env, [None, None, 0., None, None, None, None] + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, -0.1, 0.6])  
            env = ActionClipWrapper(env, [None] * 9, [None] * 6 + [0.9, 1.0, 1.0])
        elif task == 'bowl_stack_dex':
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
            env = ActionClipWrapper(env, [None] * 9, [None] * 6 + [0.5, 0.6, 0.5])
        elif task == 'button_dex':
            env = DofWrapper(env, arm_dof_setting + [0., 0.3, 0.3, 0.3] * 3 + [0.5, 0.3, 0.3, 0.3])
        elif 'lift_dex' in task:
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
            # env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.2, 0.2, None, 0.6])
        elif task == 'dual_open_pick':
            env = DofWrapper(env, [0., None, 0., None, 0., None, 0.] + [0., None, 0., None, 0., 0., 0.] + [0., 0.1, 0.8, 1.0] * 3 + [0., 0.6, 0, 0.2] + [None])
        elif 'dual' in task:
            # env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6] + arm_dof_setting + [None])
            env = DofWrapper(env, [0., None, 0., None, 0., 0., 0.] + [0., None, 0., None, 0., None, 0.] + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6] + [0.])
        elif 'dex' in task:
           env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.3] *  3 + [0.3, 0.2, None, 0.3])
        elif task == 'drawer':
            env = DofWrapper(env, arm_dof_setting + [127])
        elif domain == 'airplay' and task=='button':
            env = DofWrapper(env, arm_dof_setting + [0])
        elif domain == 'airplay' and task=='laptop':
            env = DofWrapper(env, [None, None, None, 0., 0., 0.] + [1])
        else:
            env = DofWrapper(env, arm_dof_setting + [None])
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks

    
    if (domain, task) in suite.ALL_TASKS or domain in ['ur5', 'franka', 'xarm', 'airplay']:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)

        if use_pc:
            env = PointCloudViewWrapper(
                    env, num_frames=frame_stack, cam_id=camera_id, height=img_size, width=img_size, max_points=max_points, add_rgb=add_rgb, downsample_method=downsample_method, crop_dist=crop_dist, device=device, eval=eval)
        else:
            if use_state:
                env = StateCameraViewWrapper(
                    env, num_frames=frame_stack, cam_id=camera_id, height=img_size, width=img_size, depth=use_depth)
            else:
                env = CameraViewWrapper(
                    env, num_frames=frame_stack, cam_id=camera_id, height=img_size, width=img_size, depth=use_depth)
    # stack several frames
    # env = FrameStackWrapper(env, frame_stack, pixels_key)

    if two_cam:
        if use_pc:
            env = PointCloudViewWrapper(
                    env, num_frames=frame_stack, cam_id='track_cam', height=img_size, width=img_size, max_points=max_points, add_rgb=add_rgb, downsample_method=downsample_method, crop_dist=crop_dist, device = device, eval=eval)
        else:
            if use_state:
                env = StateCameraViewWrapper(
                    env, num_frames=frame_stack, cam_id='track_cam', height=img_size, width=img_size, depth=use_depth)
            else:
                env = CameraViewWrapper(
                    env, num_frames=frame_stack, cam_id='track_cam', height=img_size, width=img_size, depth=use_depth)
    
    if use_state:
        env = StateExtendedTimeStepWrapper(env, randomize)
    else:
        env = ExtendedTimeStepWrapper(env, randomize)
    
    return env





def state_make(name, frame_stack, action_repeat, seed, randomize=False, control='joint'):
    if re.match('^anymal', name):
        name_list = name.split('_')
        domain = name_list[0] + '_' + name_list[1]
        task = name_list[2]
    else:
        domain, task = name.split('_', 1)
        # overwrite cup to ball_in_cup
        domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    elif domain in ['ur5', 'franka', 'xarm', 'airplay']:
        env = globals()[name]()
        pixels_key = 'pixels'

        JOINT_NUM = {
            'franka': 7,
            'ur5': 6,
            'xarm': 6,
            'airplay': 6,
        }
        DOF_SETTING = {
            'franka': [None, None, 0., None, 0., None, 0.],
            'ur5': [None, None, None, 0., 0., 0.],
            'xarm': [None, None, None, 0., 0., 0.],
            'airplay': [None, None, None, 0., None, 0.],
        }
        if control == 'mocap':
            arm_dof = 6
            arm_dof_setting = [None, None, None, 0., 0., 0.]
            if domain == 'franka':
                env = MocapCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), tcp_min=np.array([0, -0.45, 0]), tcp_max=np.array([1.0, 0.45, 0.6]))
            else:
                env = MocapCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
        elif control == 'joint':
            arm_dof = JOINT_NUM[domain]
            arm_dof_setting = DOF_SETTING[domain]
            if 'dual' in task:
                env = DualJointCtrlWrapper(env, action_min=-np.ones(14), action_max=np.ones(14), moving_average=0.4)
            elif task == 'bowl_dex':
                env = JointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), moving_average=0.4)
            elif task == 'close_dex':
                env = ArmJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
            # elif task == 'dual_dex':
            #     arm_dof = 7
            #     arm_dof_setting = [None, None, 0., None, 0., None, 0.]
            #     dual_arm_dof = 7
            #     env = DualJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), moving_average=0.4, dual_action_min=-np.ones(dual_arm_dof), dual_action_max=np.ones(dual_arm_dof))
            else:
                env = OriginalJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
        
        if domain == 'xarm' and task == 'bowl_dex':
            env = DofWrapper(env, arm_dof_setting + [0, None, 0.4, 0.6] * 3 + [1.8, 0.2, -0.4, 0.5])
        elif domain == 'xarm' and task == 'close_dex':
            env = DofWrapper(env, [0., None, None, 0., None, 0.] + [0, 0.1, 0.3, 0.7] * 3 + [0.2, 0, 0, 0.2])
        elif task == 'lift_dex_cube' or task == 'bowl_dex':
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
        elif task == 'lift_dex_cup':
            arm_dof_setting[5] = 0.
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
        elif task == 'pour_dex':
            env = DofWrapper(env, [None, None, 0., None, None, None, None] + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, -0.1, 0.6])  
            env = ActionClipWrapper(env, [None] * 9, [None] * 6 + [0.9, 1.0, 1.0])
        elif task == 'bowl_stack_dex':
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
            env = ActionClipWrapper(env, [None] * 9, [None] * 6 + [0.5, 0.6, 0.5])
        elif task == 'button_dex':
            env = DofWrapper(env, arm_dof_setting + [0., 0.3, 0.3, 0.3] * 3 + [0.5, 0.3, 0.3, 0.3])
        elif 'lift_dex' in task:
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
            # env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.2, 0.2, None, 0.6])
        elif task == 'dual_open_pick':
            env = DofWrapper(env, [0., None, 0., None, 0., None, 0.] + [0., None, 0., None, 0., 0., 0.] + [0., 0.1, 0.8, 1.0] * 3 + [0., 0.6, 0, 0.2] + [None])
        elif 'dual' in task:
            # env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6] + arm_dof_setting + [None])
            env = DofWrapper(env, [0., None, 0., None, 0., 0., 0.] + [0., None, 0., None, 0., None, 0.] + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6] + [0.])
        elif 'dex' in task:
           env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.3] *  3 + [0.3, 0.2, None, 0.3])
        elif task == 'drawer':
            env = DofWrapper(env, arm_dof_setting + [127])
        else:
            env = DofWrapper(env, arm_dof_setting + [None])
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # stack several frames
    env = StateFrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env, randomize)
    
    return env