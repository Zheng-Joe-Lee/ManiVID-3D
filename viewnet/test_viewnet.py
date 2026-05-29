import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algos.viewnet import ViewNet, get_loss
from dataset.viewnet_dataset import viewnet_dataloader
from utils import normalize_pc_points_fix, normalize_pc_points_centroid
from config_viewnet import config_viewnet

def test():
    
    checkpoint_path ='TODO'

    max_points = config_viewnet['max_points']
    data_dir = os.path.join(config_viewnet['data_dir'], config_viewnet['task_name'])
    data_dir = os.path.join(data_dir, f"{max_points}")

    os.environ['CUDA_VISIBLE_DEVICES'] = config_viewnet['gpu_id']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ViewNet(use_rgb=config_viewnet['use_rgb']).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loader = viewnet_dataloader(
        data_dir=os.path.join(data_dir, 'test'),
        batch_size=config_viewnet['batch_size'],
        shuffle=False,
        num_workers=config_viewnet['num_workers'],
        max_points=config_viewnet['max_points']
    )
    
    chamfer_distances = []
    mse_errors = []
    
    with torch.no_grad():
        for batch in test_loader:

            org_points = batch['org_points'].to(device)
            gt_track = batch['gt_points']['gt_cloud_track'].to(device)
            gt_fix = batch['gt_points']['gt_cloud_fix'].to(device)
            
            org_points = normalize_pc_points_centroid(org_points, config_viewnet['use_rgb'], config_viewnet['max_val'], config_viewnet['min_val'])
            gt_track = normalize_pc_points_fix(gt_track, config_viewnet['use_rgb'], config_viewnet['max_val'], config_viewnet['min_val'])
            gt_fix = normalize_pc_points_fix(gt_fix, config_viewnet['use_rgb'], config_viewnet['max_val'], config_viewnet['min_val'])

            pred_sim3, _ = model(org_points)
            
            transformed = torch.bmm(pred_sim3[:, :3, :3], org_points.transpose(1, 2)) + pred_sim3[:, :3, 3].unsqueeze(-1)
            transformed = transformed.transpose(1, 2)
            
            dist_src_tgt = torch.cdist(transformed, gt_fix).min(dim=2)[0].mean()
            dist_tgt_src = torch.cdist(gt_fix, transformed).min(dim=2)[0].mean()
            chamfer = 0.5 * (dist_src_tgt + dist_tgt_src)
            
            mse = torch.mean((transformed - gt_track) ** 2)
            
            chamfer_distances.append(chamfer.item())
            mse_errors.append(mse.item())
    
    avg_chamfer = np.mean(chamfer_distances)
    avg_mse = np.mean(mse_errors)
    print(f'Test Results - Avg Chamfer Distance: {avg_chamfer:.6f}, Avg MSE: {avg_mse:.6f}')
    
    visualize_example(model, test_loader, device)

def visualize_example(model, dataloader, device, num_examples=10):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from colorsys import hls_to_rgb
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_examples:
                break
                
            org = batch['org_points'][0].to(device)
            gt = batch['gt_points']['gt_cloud_fix'][0].to(device)
            
            org = normalize_pc_points_centroid(org.unsqueeze(0), config_viewnet['use_rgb'], config_viewnet['max_val'], config_viewnet['min_val'])[0]
            gt = normalize_pc_points_fix(gt.unsqueeze(0), config_viewnet['use_rgb'], config_viewnet['max_val'], config_viewnet['min_val'])[0]

            sim3, _ = model(org.unsqueeze(0))
            transformed = (torch.mm(sim3[0, :3, :3], org.T) + sim3[0, :3, 3].unsqueeze(-1)).T
            
            org_np = org.cpu().numpy()
            gt_np = gt.cpu().numpy()
            pred_np = transformed.cpu().numpy()
            
            fig = plt.figure(figsize=(15, 5), facecolor='none', dpi=100)
            titles = ['Input (Camera Coords)', 'Ground Truth (World)', 'Predicted (World)']

            base_color = np.array([0.53, 0.81, 1]) 
            base_hue = 0.6
            
            for j, (title, pts) in enumerate(zip(titles, [org_np, gt_np, pred_np])):
                ax = fig.add_subplot(1, 3, j+1, projection='3d')
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
                ax.set_title(title)
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([-1, 1])
            
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    test()