import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import kornia.geometry as kg
import kornia.geometry.transform as kt
import math


class SIM3STNPerImage(nn.Module):
    def __init__(self, num_points):
        super().__init__()

        self.num_points = num_points
        
        # PointNet-like 特征提取
        self.localization = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.AdaptiveMaxPool1d(1)  # Global feature
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # [tx, ty, tz, rx, ry, rz, s]
        )
        
        # 初始化为恒等变换
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        )

    def forward(self, x):
        # x: (B, 3, N)
        B = x.size(0)
        xs = self.localization(x).view(B, -1)  # (B, 256)
        params = self.fc_loc(xs)  # (B, 7)
        
        # 解析参数
        t = params[:, :3]
        r = params[:, 3:6]
        s = torch.exp(params[:, 6])  # s > 0
        
        # 构造 SIM(3) 矩阵
        R = kg.axis_angle_to_rotation_matrix(r)  # (B, 3, 3)
        T = torch.eye(4).repeat(B, 1, 1).to(x.device)
        T[:, :3, :3] = s.view(-1, 1, 1) * R
        T[:, :3, 3] = t
        
        # 变换点云
        x_hom = torch.cat([x, torch.ones(B, 1, self.num_points).to(x.device)], dim=1)
        x_transformed = torch.bmm(T, x_hom)[:, :3, :]  # (B, 3, N)
        
        return x_transformed

class SIM3STN(nn.Module):
    def __init__(self, num_points):
        super().__init__()

        self.num_points = num_points
        
        # PointNet-like 特征提取
        self.localization = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.AdaptiveMaxPool1d(1)  # Global feature
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # [tx, ty, tz, rx, ry, rz, s]
        )
        
        # 初始化为恒等变换
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([0, 0, 1e-6, 1e-6, 1e-6, 0, 0], dtype=torch.float)
        )

    def forward(self, x_org, return_sim3=False):
        # x: (B, N, 3)
        # ToDo: transform to (B, 3, N)
        x = x_org.transpose(1, 2).contiguous()
        
        B = x.size(0)
        xs = self.localization(x).view(B, -1)  # (B, 256)
        params = self.fc_loc(xs)  # (B, 7)
        # print(params)
        params[:, 3:6]
        # 解析参数
        t = torch.tanh(params[:, :3]) # [-1,1]
        r = torch.tanh(params[:, 3:6]) * math.pi # [-π, π]
        s = torch.sigmoid(params[:, 6]) + 0.5  # [0.5, 1.5]
        
        # 构造 SIM(3) 矩阵
        # R = kg.axis_angle_to_rotation_matrix(r)  # (B, 3, 3)
        # # R = self.stabilize_rotation(R)    # 强制正交
        # T = torch.eye(4).repeat(B, 1, 1).to(x.device) + 1e-6
        # T[:, :3, :3] = s.view(-1, 1, 1) * R
        # T[:, :3, 3] = t
        R = kg.axis_angle_to_rotation_matrix(r)  # (B, 3, 3)
        scale_R = s.view(-1, 1, 1) * R          # 缩放旋转矩阵
        t = t.unsqueeze(-1)                     # (B, 3, 1)
        
        # 构造 T 矩阵
        top = torch.cat([scale_R, t], dim=-1)    # (B, 3, 4)
        bottom = torch.tensor([0, 0, 0, 1], device=x.device).view(1, 1, 4).expand(B, 1, 4)
        T = torch.cat([top, bottom], dim=1)      # (B, 4, 4)
        
        # 变换点云
        x_hom = torch.cat([x, torch.ones(B, 1, self.num_points).to(x.device)], dim=1)
        x_transformed = torch.bmm(T, x_hom)[:, :3, :]  # (B, 3, N)

        # ToDo: transform to (B, N, 3)
        x_out = x_transformed.transpose(1, 2).contiguous()
        
        if return_sim3:
            return x_out, T
        else:    
            return x_out

