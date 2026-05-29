import os
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algos.viewnet import ViewNet, get_loss
from dataset.viewnet_dataset import viewnet_dataloader
from utils import normalize_pc_points_fix, normalize_pc_points_centroid
from scipy.spatial.transform import Rotation
from config_viewnet import config_viewnet

def pc_aug(points):
    device = points.device
    B, N, _ = points.shape
    sigma=0.001
    clip=0.003
    max_rotation_angle=5.0  
    max_translation=0.01
    
    aug_points = points.clone()
    
    angles = torch.rand((B, 3), device=device) * 2 * max_rotation_angle - max_rotation_angle 
    
    for i in range(B):
        rotation = Rotation.from_euler('xyz', angles[i].cpu().numpy(), degrees=True)
        rot_matrix = torch.tensor(rotation.as_matrix(), dtype=torch.float32, device=device)
        aug_points[i] = torch.matmul(aug_points[i], rot_matrix.T)

    translation = torch.rand((B, 3), device=device) * 2 * max_translation - max_translation
    aug_points[..., :3] += translation.unsqueeze(1) 
    
    noise = torch.clamp(
        torch.randn((B, N, 3), device=device) * sigma,
        min= -clip, 
        max= clip
    )
    aug_points[..., :3] += noise
    
    return aug_points

def train():

    os.environ['CUDA_VISIBLE_DEVICES'] = config_viewnet['gpu_id']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    max_points = config_viewnet['max_points']
    task = config_viewnet['task_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config_viewnet['log_dir'], f'train_{timestamp}_{task}_{max_points}')
                           
    checkpoint_dir = os.path.join(log_dir, 'ckpts')
    info_dir = os.path.join(log_dir, 'info')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    
    model = ViewNet(use_rgb=config_viewnet['use_rgb']).to(device)
    criterion = get_loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config_viewnet['lr'], weight_decay=config_viewnet['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    data_dir = os.path.join(config_viewnet['data_dir'], config_viewnet['task_name'])
    data_dir = os.path.join(data_dir, f"{max_points}")

    train_loader = viewnet_dataloader(
        data_dir=os.path.join(data_dir, 'train'),
        batch_size=config_viewnet['batch_size'],
        shuffle=True,
        num_workers=config_viewnet['num_workers'],
        max_points=config_viewnet['max_points']
    )

    test_loader = viewnet_dataloader(
        data_dir=os.path.join(data_dir, 'test'),
        batch_size=config_viewnet['test_batch_size'],
        shuffle=False,
        num_workers=config_viewnet['num_workers'],
        max_points=config_viewnet['max_points']
    )

    
    writer = SummaryWriter(info_dir, f'train_{timestamp}')
    
    best_loss_eval = float('inf')
    best_loss_train = float('inf')
    for epoch in range(config_viewnet['epochs']):
        model.train()
        running_loss = 0.0
        running_loss_print = 0.0
        
        for i, batch in enumerate(train_loader):
            org_points = batch['org_points'].to(device)  # [B, N, 3]
            gt_track = batch['gt_points']['gt_cloud_track'].to(device)
            gt_fix = batch['gt_points']['gt_cloud_fix'].to(device)

            org_points = pc_aug(org_points)
            org_points = normalize_pc_points_centroid(org_points, config_viewnet['use_rgb'], config_viewnet['max_val'], config_viewnet['min_val'])
            gt_track = normalize_pc_points_fix(gt_track, config_viewnet['use_rgb'], config_viewnet['max_val'], config_viewnet['min_val'])
            gt_fix = normalize_pc_points_fix(gt_fix, config_viewnet['use_rgb'], config_viewnet['max_val'], config_viewnet['min_val'])

            optimizer.zero_grad()

            pred_sim3, _ = model(org_points)

            loss = criterion(pred_sim3, gt_fix, gt_track, org_points)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_print += loss.item()

            if i % 100 == 99:
                avg_loss = running_loss_print / 100
                print(f'Epoch [{epoch+1}/{config_viewnet["epochs"]}], Batch [{i+1}/{len(train_loader)}], Loss: {avg_loss:.6f}')
                writer.add_scalar('batch_loss', avg_loss, epoch * len(train_loader) + i)
                running_loss_print = 0.0

        scheduler.step()
 
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('epoch_loss', epoch_loss, epoch)

        if epoch_loss < best_loss_train:
            best_loss_train = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss_train,
            }, os.path.join(checkpoint_dir, 'best_model_train.pth'))
        
        if (epoch + 1) % config_viewnet['eval_interval'] == 0 or epoch == config_viewnet['epochs'] - 1:
            test_metrics = evaluate(model, test_loader, device)
            writer.add_scalar('test_chamfer', test_metrics['chamfer'], epoch)
            writer.add_scalar('test_mse', test_metrics['mse'], epoch)
            print(f"Epoch [{epoch+1}] Test Metrics - Chamfer: {test_metrics['chamfer']:.6f}, MSE: {test_metrics['mse']:.6f}")

            if test_metrics['chamfer'] < best_loss_eval:
                best_loss_eval = test_metrics['chamfer']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss_eval,
                }, os.path.join(checkpoint_dir, 'best_model.pth'))
                print(f"New best model saved with Chamfer: {best_loss_eval:.6f}")

        if epoch % 10 == 9:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print(f'Epoch [{epoch+1}/{config_viewnet["epochs"]}] completed. Avg Loss: {epoch_loss:.6f}, Best Loss: {best_loss_train:.6f}')
    
    writer.close()
    print('Training finished')

def evaluate(model, dataloader, device):
    model.eval()
    chamfer_distances = []
    mse_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            org_points = batch['org_points'].to(device)
            gt_track = batch['gt_points']['gt_cloud_track'].to(device)
            gt_fix = batch['gt_points']['gt_cloud_fix'].to(device)
            org_points = normalize_pc_points_centroid(org_points, config_viewnet['use_rgb'], config_viewnet['max_val'], config_viewnet['min_val'])
            gt_track = normalize_pc_points_fix(gt_track, config_viewnet['use_rgb'], config_viewnet['max_val'], config_viewnet['min_val'])
            gt_fix = normalize_pc_points_fix(gt_fix, config_viewnet['use_rgb'], config_viewnet['max_val'], config_viewnet['min_val'])
            
            pred_sim3, _ = model(org_points)
            transformed = torch.bmm(pred_sim3[:, :3, :3], org_points.transpose(1, 2)) + pred_sim3[:, :3, 3].unsqueeze(-1)
            transformed = transformed.transpose(1, 2)
            
            # Chamfer Distance
            dist_src_tgt = torch.cdist(transformed, gt_fix).min(dim=2)[0].mean()
            dist_tgt_src = torch.cdist(gt_fix, transformed).min(dim=2)[0].mean()
            chamfer = 0.5 * (dist_src_tgt + dist_tgt_src)
            
            # MSE
            mse = torch.mean((transformed - gt_track) ** 2)
            
            chamfer_distances.append(chamfer.item())
            mse_errors.append(mse.item())
    
    model.train()
    return {
        'chamfer': sum(chamfer_distances) / len(chamfer_distances),
        'mse': sum(mse_errors) / len(mse_errors)
    }

if __name__ == '__main__':
    train()