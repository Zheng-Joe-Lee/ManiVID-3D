import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnetpp_utils import PointNetSetAbstraction


class ViewNet(nn.Module):
    def __init__(self, use_rgb=False):
        super(ViewNet, self).__init__()
        in_channel = 6 if use_rgb else 3
        self.use_rgb = use_rgb
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 8)  # 预测四元数[4] + 平移[3] + 缩放[1]

    def forward(self, xyz):
        """
        xyz: input points position data, [B, N, C]
        """
        B, _, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1) # transfoem to [B, C, N]
        if self.use_rgb:
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        quat = F.normalize(x[:, :4], dim=1)  # 单位四元数
        trans = torch.tanh(x[:, 4:7]) * 0.5 #平移限制在[-0.5, 0.5]范围内
        scale = torch.sigmoid(x[:, 7:]) * 0.2 + 0.9  # 缩放限制在[0.9, 1.1]

        # 构建SIM3矩阵
        sim3 = self._build_sim3_matrix(quat, trans, scale)

        return sim3, l3_points

    def _build_sim3_matrix(self, quat, trans, scale):
        """根据预测参数构建SIM(3)矩阵"""
        B = quat.size(0)
        device = quat.device
        
        # 四元数转旋转矩阵
        R = self._quaternion_to_matrix(quat)
        
        # 构建4x4变换矩阵
        sim3 = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        sim3[:, :3, :3] = R * scale.view(B, 1, 1)  # 缩放旋转
        sim3[:, :3, 3] = trans
        
        return sim3

    def _quaternion_to_matrix(self, quat):
        """四元数转旋转矩阵"""
        x, y, z, w = quat.unbind(-1)
        
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return torch.stack([
            1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)
        ], dim=-1).view(-1, 3, 3)



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        
    def forward(self, pred_sim3, target_cloud_fix, target_cloud_track, source_cloud):
        """
        计算Chamfer Distance损失
        Args:
            pred_sim3: 预测的SIM(3)矩阵 [B,4,4]
            target_cloud: 目标点云 [B,N,3]
            source_cloud: 源点云 [B,N,3]
        """
        # 应用变换到源点云
        transformed_cloud = torch.bmm(
            pred_sim3[:, :3, :3], source_cloud.transpose(1, 2)) + pred_sim3[:, :3, 3].unsqueeze(-1)
        
        # 计算Chamfer Distance
        dist_src_tgt = torch.cdist(transformed_cloud.transpose(1, 2), target_cloud_fix).min(dim=2)[0].mean()
        dist_tgt_src = torch.cdist(target_cloud_fix, transformed_cloud.transpose(1, 2)).min(dim=2)[0].mean()

        mse = F.mse_loss(transformed_cloud.transpose(1, 2), target_cloud_track)
        
        return 0.5*(dist_src_tgt + dist_tgt_src)/2 + 0.5*mse