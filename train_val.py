import sys
import os
import torch 
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"]='0, 1'

# 数据集提供的密度图
from data.dataloader import build_dataloader
from networks.safe import build_network
from utils.optim_helper import get_optimizer, get_scheduler
from utils.checkpoint_helper import load_checkpoint, save_checkpoint
import wandb

# from utils.show_helper import show_density, show_img_boxes_points
import argparse
import yaml
from easydict import EasyDict

# 单批次
def train_epoch(t_loader, net, optimizer, epoch, gpu_id, factor=255.0):
    # 训练模型，确定优化参数
    net.train()
    net.backbone.eval()
    for p in net.backbone.parameters():
        p.requires_grad = False
    
    # 测度，评价指标
    mae = 0
    mse = 0 
    train_loss = 0
    num_sample_sum = 0
    for index, sample in enumerate(t_loader):
        images, densitys, boxes, points, st_sizes, file_name = sample 
        images = images.to(gpu_id)    # b,c,h,w
        boxes = boxes.to(gpu_id)      # b,k,4
        densitys = densitys.to(gpu_id)  # b,h,w
        
        num_sample = images.shape[0]
        num_sample_sum += num_sample
        
        # gt = [len(point) for point in points]
        gt = len(points[0])
        densitys = densitys.unsqueeze(1)*factor  # 1, 1, 512, 512

        # 前向+反向
        optimizer.zero_grad()
        outputs = net(images, boxes.squeeze(0))
        loss = torch.nn.MSELoss()(outputs, densitys)
        loss.backward()
        optimizer.step()

        # 结果
        outputs = outputs/factor
        pre = outputs.sum().item()
        err = pre-gt
        train_loss += loss.item()
        print("[TRAIN]|Epoch:{}|Iter:{}/{}|GT:{:5.1f}|Pred:{:5.1f}|Error:{:5.1f}|Loss:{}".format(
                        epoch, index, len(t_loader), gt, pre, err, loss.item()))

        mae += abs(gt-pre)
        mse += ((gt-pre)*(gt-pre))
    train_loss /= num_sample_sum
    mae /= num_sample_sum
    mse /= num_sample_sum
    rmse = mse ** 0.5
    print("[TRAIN]|Epoch:{}|MAE:{:5.3f}|RMSE:{:5.3f}|Train_Loss:{}".format(
        epoch, mae, rmse, train_loss
    ))
    return train_loss, mae, rmse
              
def val(v_loader, net, gpu_id, factor=255.0):
    mae = 0
    mse = 0 
    val_loss = 0
    num_sample_sum = 0
    net.eval()
    with torch.no_grad():
        for index, sample in enumerate(v_loader):
            images, densitys, boxes, points, st_sizes, file_name = sample 
            images = images.to(gpu_id)    # b,c,h,w
            boxes = boxes.to(gpu_id)      # b,k,4
            densitys = densitys.to(gpu_id)  # b,h,w
            
            num_sample = images.shape[0]
            num_sample_sum += num_sample

            gt = len(points[0])
            densitys = densitys.unsqueeze(1)*factor  # 1, 1, 512, 512

            # 前向传播
            outputs = net(images, boxes[0])
            loss = torch.nn.MSELoss()(outputs, densitys)

            # 结果
            outputs = outputs/factor
            pre = outputs.sum().item()
            err = pre-gt
            val_loss += loss.item()    
    
            print("[VAL]|Iter:{}/{}|GT:{:5.1f}|Pred:{:5.1f}|Error:{:5.1f}|Loss:{}".format(
                        index, len(v_loader), gt, pre, err, loss.item()))
            mae += abs(gt-pre)
            mse += ((gt-pre)*(gt-pre))
        val_loss /= num_sample_sum
        mae /= num_sample_sum
        mse /= num_sample_sum
        rmse = mse ** 0.5
        print("[VAL]MAE:{:5.3f}|RMSE:{:5.3f}|Val_Loss:{}".format(
            mae, rmse, val_loss
        ))
        return val_loss, mae, rmse

def load_obj(cfg):
    # 数据
    loader = build_dataloader(cfg.dataset, distributed=False)
    # 模型
    net = build_network(cfg.net)
    net.train()
    net.backbone.eval()
    # 训练参数
    for p in net.backbone.parameters():
        p.requires_grad = False
        # parameters not include backbone
    parameters = [
        p for n, p in net.named_parameters() if "backbone" not in n
    ]
    # 优化器
    optimizer = get_optimizer(parameters, cfg.trainer.optimizer)
    # 学习率变更
    lr_scheduler = get_scheduler(optimizer, cfg.trainer.lr_scheduler)
    return loader, net, optimizer, lr_scheduler

# def save_checkpoint(saver_cfg, cpk, file_name='checkpoint.pt'):
#     root = saver_cfg.root
#     os.makedirs(root, exist_ok=True)
#     name = saver_cfg.name
#     save_dir = os.path.join(root, name)
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir,file_name)
#     torch.save(cpk, save_path)
#     print(f"Training checkpoint has been saved at {save_path}")

parser = argparse.ArgumentParser(description="class-agnostic counting")
parser.add_argument( "-c", "--config", type=str, default="/home/zg/counting/notebook.yaml", help="Path of config")

def main():
    args = parser.parse_args(args=[])
    with open(args.config) as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    wandb.init(project="counting", entity="hzg")
    wandb.config = { "learning_rate": cfg.trainer.optimizer.kwargs.lr, "epochs": cfg.trainer.max_epochs,}

    # 1. 数据、模型、优化器
    loader, net, optimizer, lr_scheduler = load_obj(cfg)

    # 2. 参数
    max_epochs = cfg.trainer.max_epochs
    begin_epoch = cfg.trainer.begin_epoch
    best_mae = np.inf
    best_rmse = np.inf
    save_every = cfg.trainer.save_every

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    net.to(device)

    # 3. 预训练，断点续训
    # if os.path.exists(cfg.saver['load_path']):    # 存在预训练模型
    #     checkpoint = torch.load(cfg.saver['load_path'])
    #     net.load_state_dict(checkpoint["state_dict"])
    #     begin_epoch = checkpoint["epoch"]+1
    #     best_mae = checkpoint["best_mae"]
    #     best_rmse = checkpoint["best_rmse"]

    cpk = load_checkpoint(cfg.saver['load_path'])
    if cpk!=None:
        net.load_state_dict(cpk["state_dict"])
        begin_epoch = cpk["epoch"]+1
        best_mae = cpk["best_mae"]
        best_rmse = cpk["best_rmse"]
        # logger.info(f"Epoch {epoch} | Training checkpoint saved at {save_path}")

    for epoch in range(begin_epoch, max_epochs):
        print('| epoch: {} | learning_rate : {} |'.format(epoch, lr_scheduler.get_last_lr()[0]))
        train_loss, train_mae, train_rmse = train_epoch(loader[0], net, optimizer, epoch, device)
        val_loss, val_mae, val_rmse = val(loader[1], net, device)
        if epoch%save_every==0:
            cpk = {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "best_mae": best_mae, 
                    "best_rmse": best_rmse,
            }
            save_checkpoint(cfg.saver, cpk, file_name='checkpoint.pt')
        if val_mae<best_mae:
            best_mae = val_mae
            best_rmse = val_rmse
            cpk = {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "best_mae": best_mae, 
                    "best_rmse": best_rmse,
            }
            save_checkpoint(cfg.saver, cpk, file_name='best_val.pt')
        lr_scheduler.step()

        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, commit=False)
        wandb.log({"train_mae": train_mae, "val_mae": val_mae}, commit=False)
        wandb.log({"train_rmse": train_rmse, "val_rmse": val_rmse})

if __name__ == '__main__':
    main()
