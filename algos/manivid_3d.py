# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet
import utils
import random
from collections import deque
# from algos.stn import TransformNet_STN_PerImage, TransformNet_STN1, PerspectiveSTNPerImage, PerspectiveSTN
from algos.stn_3d import SIM3STNPerImage, SIM3STN
from utils import random_overlay, random_mask_freq_v2, color_jitter, random_point_aug, normalize_pc_points

def chamfer_dist(pc1, pc2):
    # pc1: (B, N, 3), pc2: (B, M, 3)
    dist = torch.cdist(pc1, pc2)  # (B, N, M)
    min_dist_pc1_to_pc2 = torch.min(dist, dim=2)[0]  # (B, N)
    min_dist_pc2_to_pc1 = torch.min(dist, dim=1)[0]  # (B, M)
    
    return min_dist_pc1_to_pc2.mean() + min_dist_pc2_to_pc1.mean()

def chamfer_dist_lowmem(pc1, pc2, chunk_size=128):
    B, N, _ = pc1.shape
    M = pc2.shape[1]
    dist_pc1_to_pc2 = []
    
    # 分块计算 pc1 -> pc2
    for i in range(0, N, chunk_size):
        chunk = pc1[:, i:i+chunk_size]
        dist_chunk = torch.cdist(chunk, pc2)  # [B, chunk_size, M]
        min_dist = torch.min(dist_chunk, dim=2)[0]  # [B, chunk_size]
        dist_pc1_to_pc2.append(min_dist)
    
    dist_pc1_to_pc2 = torch.cat(dist_pc1_to_pc2, dim=1)  # [B, N]
    
    # 同理计算 pc2 -> pc1
    dist_pc2_to_pc1 = []
    for j in range(0, M, chunk_size):
        chunk = pc2[:, j:j+chunk_size]
        dist_chunk = torch.cdist(pc1, chunk)  # [B, N, chunk_size]
        min_dist = torch.min(dist_chunk, dim=1)[0]  # [B, chunk_size]
        dist_pc2_to_pc1.append(min_dist)
    
    dist_pc2_to_pc1 = torch.cat(dist_pc2_to_pc1, dim=1)  # [B, M]
    
    return dist_pc1_to_pc2.mean() + dist_pc2_to_pc1.mean()

def compare_models(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.equal(param1, param2):
            print(f"参数不同: {name1} vs {name2}")
            return False
    return True

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
    

class RandomPCJitterAug(nn.Module):
    
    def __init__(self, sigma=0.001, clip=0.003):
        super().__init__()
        self.sigma = sigma
        self.clip = clip
    
    def forward(self, x):
        
        device = x.device
        B, N, _ = x.shape

        noise = torch.clamp(
        torch.randn((B, N, 3), device=device) * self.sigma,
        min=-self.clip, 
        max=self.clip
        )
        
        jittered_points = x.clone()
        jittered_points[..., :3] += noise  # 只对xyz坐标添加噪声
        
        return jittered_points
    
class RandomPCRotationAug(nn.Module):
    def __init__(self, 
                 sigma=0.001, 
                 clip=0.003,
                 max_rotation_angle=5.0,  # 最大旋转角度(度)
                 max_translation=0.01,    # 最大平移距离
                 apply_noise=False,
                 apply_rotation=True,
                 apply_translation=True):
        super().__init__()
        self.sigma = sigma
        self.clip = clip
        self.max_rotation_angle = max_rotation_angle
        self.max_translation = max_translation
        self.apply_noise = apply_noise
        self.apply_rotation = apply_rotation
        self.apply_translation = apply_translation
        
    def forward(self, x):
        device = x.device
        B, N, _ = x.shape
        
        aug_points = x.clone()
        
        # 1. 添加随机旋转
        if self.apply_rotation:
            # 生成随机旋转角度(绕x,y,z轴)
            angles = torch.rand((B, 3), device=device) * 2 * self.max_rotation_angle - self.max_rotation_angle  # [-max, max]
            
            # 为每个batch创建旋转矩阵
            for i in range(B):
                rotation = Rotation.from_euler('xyz', angles[i].cpu().numpy(), degrees=True)
                rot_matrix = torch.tensor(rotation.as_matrix(), dtype=torch.float32, device=device)
                aug_points[i] = torch.matmul(aug_points[i], rot_matrix.T)
        
        # 2. 添加随机平移
        if self.apply_translation:
            translation = torch.rand((B, 3), device=device) * 2 * self.max_translation - self.max_translation  # [-max, max]
            aug_points[..., :3] += translation.unsqueeze(1)  # 广播到所有点
        
        # 3. 添加高斯噪声
        if self.apply_noise:
            noise = torch.clamp(
                torch.randn((B, N, 3), device=device) * self.sigma,
                min=-self.clip, 
                max=self.clip
            )
            aug_points[..., :3] += noise
        
        return aug_points
    



class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        # cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        # cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3])
        )

        self.mlp = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
        )

        # self.mlp = nn.Sequential(
        #     nn.Linear(in_channels, block_channel[0]),
        #     nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
        #     nn.ReLU(),
        #     nn.Linear(block_channel[0], block_channel[1]),
        #     nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
        #     nn.ReLU(),
        #     nn.Linear(block_channel[1], block_channel[2]),
        #     nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
        #     nn.ReLU(),
        #     nn.Linear(block_channel[2], block_channel[3]),
        # )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
        input: (Batch, Num, 3)
        output: (Batch, out_channels)
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        # cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        # cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, "PointNetEncoderXYZ only supports 3 channels, but got {in_channels}"
       
        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        self.layer3_rel = nn.Sequential(
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        # self.mlp = nn.Sequential(
        #     self.layer1,
        #     self.layer2,
        #     self.layer3
        # )

        # self.mlp = nn.Sequential(
        #     nn.Linear(in_channels, block_channel[0]),
        #     nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
        #     nn.ReLU(),
        #     nn.Linear(block_channel[0], block_channel[1]),
        #     nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
        #     nn.ReLU(),
        #     nn.Linear(block_channel[1], block_channel[2]),
        #     nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
        #     nn.ReLU(),
        # )
        
        

        self.final_projection_inv = nn.Sequential(
            nn.Linear(block_channel[-1], out_channels),
            nn.LayerNorm(out_channels)
        )

        self.final_projection_rel = nn.Sequential(
            nn.Linear(block_channel[-1], out_channels),
            nn.LayerNorm(out_channels)
        )

        # VIS_WITH_GRAD_CAM = False
        # if VIS_WITH_GRAD_CAM:
        #     self.gradient = None
        #     self.feature = None
        #     self.input_pointcloud = None
        #     self.mlp[0].register_forward_hook(self.save_input)
        #     self.mlp[6].register_forward_hook(self.save_feature)
        #     self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.layer2(self.layer1(x))
        x_inv = self.layer3(x)
        x_inv = torch.max(x_inv, 1)[0]
        x_inv = self.final_projection_inv(x_inv)

        x_rel = self.layer3_rel(x)
        x_rel = torch.max(x_rel, 1)[0]
        x_rel = self.final_projection_rel(x_rel)
        return x_inv, x_rel
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class Encoder3D(nn.Module):
    def __init__(
            self, 
            pointcloud_encoder_cfg,
            obs_shape,  # (num_points, in_channels)
            use_pc_color=False):
        super().__init__()

        in_channels = obs_shape[1]
        num_points = obs_shape[0]
        self.repr_dim = pointcloud_encoder_cfg.out_channels
        self.use_pc_color = use_pc_color
        self.in_channels = in_channels
        self.num_points = num_points


        # pointcloud_encoder_cfg.out_channels = self.repr_dim 
        if use_pc_color:
            pointcloud_encoder_cfg.in_channels = 6
            assert in_channels == 6
            self.model = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg).cuda()
        else:
            pointcloud_encoder_cfg.in_channels = 3
            assert in_channels == 3
            self.model = PointNetEncoderXYZ(**pointcloud_encoder_cfg).cuda()
        

        # Construct STN
        # self.stn3d = SIM3STN(num_points).cuda()

        # self.apply(utils.weight_init)

    

    def forward(self, x, layer_feat=False, return_sim3=False):
        
        # x_norm = normalize_pc_points(x, self.use_pc_color)

        # print('before stn:')
        # print(torch.max(x_norm))
        # print(torch.min(x_norm))

        layers = []
        # STN3D
        ###################################
        # if self.use_pc_color:
        #     xyz = x[..., :3]
        #     color = x[..., 3:]

        #     xyz, T_sim3 = self.stn3d(xyz, True)
        #     x = torch.cat([xyz, color], dim=-1)
        # else:
        #     x, T_sim3 = self.stn3d(x, True)
        ########################################
        T_sim3 = torch.eye(4)

        # print('after stn:')
        # print(torch.max(x))
        # print(torch.min(x))
        # print(T_sim3.mean(dim=0))
        # x = normalize_pc_points(x, self.use_pc_color)
        
        layers.append(x)
        
        x = self.model.layer1(x)
        feat_layer = torch.max(x, 1)[0]
        layers.append(feat_layer)

        x = self.model.layer2(x)
        feat_layer = torch.max(x, 1)[0]
        layers.append(feat_layer)

        # disentangle layer
        x_inv = self.model.layer3(x)
        x_inv = torch.max(x_inv, 1)[0]
        layers.append(x_inv)

        x_rel = self.model.layer3_rel(x)
        x_rel = torch.max(x_rel, 1)[0]

        feat_inv = self.model.final_projection_inv(x_inv)
        feat_rel = self.model.final_projection_rel(x_rel)

        if return_sim3:
            if layer_feat:
                return feat_inv, feat_rel, layers, T_sim3
            else:
                return feat_inv, feat_rel, T_sim3
        else:
            if layer_feat:
                return feat_inv, feat_rel, layers
            else:
                return feat_inv, feat_rel


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist
    

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class Auxiliary(nn.Module):
    def __init__(self, repr_dim, feature_dim, temp=0.1):
        super().__init__()

        self.temp = temp
        
        self.projector = nn.Linear(repr_dim, feature_dim)

        self.apply(utils.weight_init)

    def contrastive_loss(self, q, k):
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.temp
        labels = torch.arange(logits.shape[0], dtype=torch.long).cuda()
        contrastive_loss = nn.CrossEntropyLoss()(logits, labels)
        return contrastive_loss

    def forward(self, q, k):
        q = self.projector(q)
        k = self.projector(k)
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        loss = (self.contrastive_loss(q, k) + self.contrastive_loss(k, q)) / 2
        return loss

class Auxiliary_disen(nn.Module):
    def __init__(self, repr_dim, feature_dim, temp=0.1, ortho_weight=1.0):
        super().__init__()
        self.temp = temp
        self.ortho_weight = ortho_weight 
        
        self.projector_rel = nn.Linear(repr_dim, feature_dim)

    def contrastive_loss(self, q, k_pos, k_neg):


        pos_logits = torch.einsum('nc,nc->n', [q, k_pos]).unsqueeze(-1) / self.temp

        neg_logits = torch.einsum('nc,mc->nm', [q, k_neg]) / self.temp

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda() 
        return nn.CrossEntropyLoss()(logits, labels)

    def orthogonality_loss(self, feat1, feat2):
        """正交损失：计算两个特征间的正交性"""
        feat1_norm = nn.functional.normalize(feat1, dim=1)
        feat2_norm = nn.functional.normalize(feat2, dim=1)
        return torch.mean(torch.abs(torch.sum(feat1_norm * feat2_norm, dim=1)))

    def forward(self, feat_invariant1, feat_specific1, feat_invariant2, feat_specific2):
        """
        输入两对特征(来自两个不同视角)
        - feat_invariant*: 视角无关特征
        - feat_specific*: 视角相关特征
        """
        # 特征投影
        proj_rel1 = nn.functional.normalize(self.projector_rel(feat_specific1), dim=1)
        proj_rel2 = nn.functional.normalize(self.projector_rel(feat_specific2), dim=1)

        pos_pairs = proj_rel1.roll(shifts=1, dims=0)

        spec_loss = self.contrastive_loss(
            q=proj_rel1,          # 查询：固定视角
            k_pos=pos_pairs,      # 正样本：同一固定视角的其他状态
            k_neg=proj_rel2       # 负样本：同一状态的所有随机视角特征
        )

        # 正交损失（保持不变）
        ortho_loss = (self.orthogonality_loss(feat_invariant1, feat_specific1) +
                     self.orthogonality_loss(feat_invariant2, feat_specific2)) / 2

        return spec_loss + self.ortho_weight * ortho_loss

    
class SigmoidLR:
    def __init__(self, optimizer, lr_max=1, lr_min=0, sigmoid_slope=0.015, sigmoid_center=500):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.sigmoid_slope = sigmoid_slope
        self.sigmoid_center = sigmoid_center
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)

    def lr_lambda(self, epoch):
        lr = (self.lr_max - self.lr_min) / (1 + np.exp(-self.sigmoid_slope * (epoch - self.sigmoid_center))) + self.lr_min
        return lr

    def step(self):
        self.scheduler.step()

class ManiAgent3D:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, use_wandb,
                 temp, aux_coef, aux_l2_coef, aux_tcc_coef, aux_latency, lr_stn, use_pc_color, pointcloud_encoder_cfg):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb or use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.aux_coef = aux_coef
        self.aux_l2_coef = aux_l2_coef
        self.aux_tcc_coef = aux_tcc_coef
        self.aux_latency = aux_latency
        self.use_pc_color = use_pc_color
        self.aux_min_scaling = 0.1
        self.aux_schedule_steps = 100000


        # models
        self.encoder = Encoder3D(pointcloud_encoder_cfg, obs_shape, use_pc_color).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.auxiliary = Auxiliary(self.encoder.repr_dim, feature_dim=256, temp=temp).to(device)
        self.auxiliary_disen = Auxiliary_disen(self.encoder.repr_dim, feature_dim=256, temp=temp).to(device)

        # optimizers
        self.encoder_no_stn_opt = torch.optim.Adam(self.encoder.model.parameters(), lr=lr)
        # self.stn_opt = torch.optim.Adam(self.encoder.stn3d.parameters(), lr=lr_stn)
        self.encoder_opt = torch.optim.Adam(self.encoder.model.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.encoder_no_stn_aux_opt = torch.optim.Adam(self.encoder.model.parameters(), lr=lr)
        # self.aux_opt_scheduler = SigmoidLR(self.encoder_no_stn_aux_opt)
        # self.stn_opt_scheduler = SigmoidLR(self.stn_opt)

        # data augmentation
        self.aug = RandomPCRotationAug()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.float()
        # print(obs.dtype)
        obs, _ = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]
    
    def read_q(self, obs, step):
        stddev = utils.schedule(self.stddev_schedule, step)
        obs = torch.as_tensor(obs, device=self.device)
        obs, _ = self.encoder(obs.unsqueeze(0))
        dist = self.actor(obs, stddev)
        action = dist.mean
        q = self.critic(obs, action)
        
        return q


    def update_critic(self, obs, action, reward, discount, next_obs, step, aug_obs, aug_move_obs):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        aug_Q1, aug_Q2 = self.critic(aug_obs, action)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)
        
        
        # critic_loss = 0.5 * (critic_loss + aug_loss) + 0.3 * aug_move_loss
        if step > self.aux_latency:
            aug_move_Q1, aug_move_Q2 = self.critic(aug_move_obs, action)
            aug_move_loss = F.mse_loss(aug_move_Q1, target_Q) + F.mse_loss(aug_move_Q2, target_Q)
            critic_loss = 0.5 * critic_loss + 0.25 * (aug_loss + aug_move_loss)
            # critic_loss = 0.5 * (critic_loss + aug_move_loss)
            # critic_loss = 0.5 * (critic_loss + aug_loss) + 0.3 * aug_move_loss
        else:
            critic_loss = 0.5 * (critic_loss + aug_loss)
        # critic_loss = 0.5 * (critic_loss + aug_move_loss)

        # l2_loss_aug = F.mse_loss(obs, aug_obs) * 0.1

        if self.use_tb:
            # metrics['critic_target_q'] = target_Q.mean().item()
            # metrics['critic_q1'] = Q1.mean().item()
            # metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_no_stn_opt.zero_grad(set_to_none=True)
        # self.stn_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        # (critic_loss + l2_loss_aug).backward()
        critic_loss.backward()
        self.critic_opt.step()
        # self.stn_opt.step()
        self.encoder_no_stn_opt.step()
        
        # if self.use_tb:
        #     grad = self.encoder.model.conv1.weight.grad
        #     metrics['grad_critic_mean'] = grad.mean().item() if grad is not None else 0
        #     metrics['grad_critic_max'] = grad.max().item() if grad is not None else 0
        #     metrics['grad_critic_min'] = grad.min().item() if grad is not None else 0
            # metrics['aux_l2_loss_aug'] = l2_loss_aug.item()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            # metrics['actor_logprob'] = log_prob.mean().item()
            # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_auxiliary(self, step, fix_obs, move_obs, trajs):
        metrics = dict()

        
        # update auxiliary task
        def calc_aux():
            fix_view_feat_inv, fix_view_feat_rel, fix_layers, sim3 = self.encoder(fix_obs, layer_feat=True, return_sim3=True)
            move_view_feat_inv, move_view_feat_rel, move_layers = self.encoder(move_obs, layer_feat=True)
            
            contrastive_loss = self.auxiliary(fix_view_feat_inv, move_view_feat_inv)
            disen_loss = self.auxiliary_disen(fix_view_feat_inv, fix_view_feat_rel, move_view_feat_inv, move_view_feat_rel)
            
            l2_loss = F.mse_loss(fix_view_feat_inv, move_view_feat_inv)
            
            # only use layer2 & layer3 feat
            fix_layers = fix_layers[-2:]
            move_layers = move_layers[-2:]

            l2_loss_layers = 0
            for fix_layer, move_layer in zip(fix_layers, move_layers):
                l2_loss_layers += F.mse_loss(fix_layer, move_layer)
            l2_loss_layers /= len(fix_layers)
    
            # print("contrastive_loss:", contrastive_loss.item())
            # print("l2_loss:", l2_loss.item())
            # print("l2_loss_layers:", l2_loss_layers.item())

            aux_loss = contrastive_loss * self.aux_coef + \
                    l2_loss * self.aux_l2_coef + l2_loss_layers * self.aux_l2_coef

                
            if self.use_tb:
                metrics['aux_contrastive_loss'] = contrastive_loss.item()
                metrics['aux_l2_loss'] = l2_loss.item()
                metrics['aux_l2_loss_layers'] = l2_loss_layers.item()
                metrics['aux_disen_loss'] = disen_loss.item()
                # metrics['sim3_02'] = sim3.mean(dim=0)[0][0].item()
                # metrics['aux_lr'] = self.encoder_no_stn_aux_opt.param_groups[0]['lr']
                    
            return aux_loss, disen_loss
        
        self.encoder_no_stn_aux_opt.zero_grad(set_to_none=True)
        # self.stn_opt.zero_grad(set_to_none=True)
        
        if step > self.aux_latency:
            
            aux_scaling = max(self.aux_min_scaling, min(1.0, (step - self.aux_latency) / self.aux_schedule_steps))
            aux_loss_1, disen_loss = calc_aux()

            aux_loss = aux_loss_1 + disen_loss*aux_scaling
 
            aux_loss.backward()
            # nn.utils.clip_grad_norm_(self.encoder.parameters(), 25, error_if_nonfinite=False)
            # self.stn_opt.step()
            self.encoder_no_stn_aux_opt.step()
        else:
            # with torch.no_grad(), utils.eval_mode(self.encoder):
            #     aux_loss = calc_aux()
            metrics['aux_contrastive_loss'] = 0
            metrics['aux_l2_loss'] = 0
            metrics['aux_l2_loss_layers'] = 0
            metrics['aux_disen_loss'] = 0
            metrics['theta1_02'] = 0
            metrics['aux_lr'] = self.encoder_no_stn_aux_opt.param_groups[0]['lr']
            
        
        # if self.use_tb:
        #     grad = self.encoder.model.conv1.weight.grad
        #     metrics['grad_aux_mean'] = grad.mean().item() if grad is not None else 0
        #     metrics['grad_aux_max'] = grad.max().item() if grad is not None else 0
        #     metrics['grad_aux_min'] = grad.min().item() if grad is not None else 0

        return metrics

    def update(self, replay_iter, trajs, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # auxiliary
        l = obs.shape[1] // 2
        fix_obs=obs.float()[:, :l]
        move_obs=obs.float()[:, l:]
        fix_next_obs = next_obs.float()[:, :l]
        
        # augment
        obs = self.aug(fix_obs)
        original_obs = obs.clone()
        next_obs = self.aug(fix_next_obs)
        # import ipdb;ipdb.set_trace()
        original_move_obs = move_obs.clone()

        # TODO: not elegant
        # strong augmentation + SRM
        # if l % 3 == 0:
        #     aug_obs = random_mask_freq_v2(random_overlay(original_obs))
        #     if step > self.aux_latency:
        #         aug_move_obs = random_mask_freq_v2(random_overlay(original_move_obs))
        #         aug_move_obs = self.encoder(aug_move_obs) 
        #     else:
        #         aug_move_obs = None
        # else:
        #     aug_obs = random_mask_freq_v2(random_overlay(original_obs[:, :l-1]))
        #     # print("", aug_obs.shape, )
        #     aug_obs = torch.cat([aug_obs, original_obs[:, l-1:l]], dim=1)
        #     if step > self.aux_latency:
        #         aug_move_obs = random_mask_freq_v2(random_overlay(original_move_obs[:, :l-1]))
        #         aug_move_obs = torch.cat([aug_move_obs, original_move_obs[:, l-1:l]], dim=1)
        #         aug_move_obs = self.encoder(aug_move_obs)       
        #     else:
        #         aug_move_obs = None
        
        # strong augmentation

        aug_obs = random_point_aug(original_obs)
        if self.use_pc_color:
            aug_obs = color_jitter(original_obs)
        if step > self.aux_latency:
            aug_move_obs = random_point_aug(original_move_obs)
            if self.use_pc_color:
                aug_move_obs = color_jitter(original_move_obs)
            aug_move_obs, _ = self.encoder(aug_move_obs)
        else:
            aug_move_obs = None


        aug_obs, _ = self.encoder(aug_obs)
        # encode
        obs, _ = self.encoder(obs)
        with torch.no_grad():
            next_obs, _ = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, aug_obs, aug_move_obs))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update auxiliary task
        metrics.update(self.update_auxiliary(step, fix_obs, move_obs, trajs))

        # print("layer1的梯度范数:", self.encoder.model.layer1[0].weight.grad.norm())
        # print("layer2的梯度范数:", self.encoder.model.layer2[0].weight.grad.norm())
        # print("layer3的梯度范数:", self.encoder.model.layer3[0].weight.grad.norm())
        
        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    
