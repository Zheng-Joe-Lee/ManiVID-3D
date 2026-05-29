import os

workspace_dir = './'

config_viewnet = {
    'task_name': 'airplay_lift',
    'max_val': [0.7, 0.1, 1.5],
    'min_val': [0, -0.1, 0.7],
    'data_dir': os.path.join(workspace_dir, 'data'),
    'log_dir': os.path.join(workspace_dir, 'logs'),
    'batch_size': 32,
    'test_batch_size': 16,
    'num_workers': 4,
    'max_points': 1024,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'epochs': 200,
    'eval_interval': 5,
    'use_rgb': False,
    'gpu_id': '0'
}