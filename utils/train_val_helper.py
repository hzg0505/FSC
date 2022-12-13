import torch

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
