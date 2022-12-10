import os
import torch

def save_checkpoint(saver_cfg, cpk, file_name='checkpoint.pt'):
    root = saver_cfg.root 
    os.makedirs(root, exist_ok=True)
    name = saver_cfg.name
    save_dir = os.path.join(root, name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,file_name)
    torch.save(cpk, save_path)
    print(f"Training checkpoint has been saved at {save_path}")

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):    # 存在预训练模型
        checkpoint = torch.load(checkpoint_path)
        print(f"Load checkpoint from {checkpoint_path}")
        return checkpoint
    else:
        print(f"Checkpoint_path {checkpoint_path} not found!")
        return None