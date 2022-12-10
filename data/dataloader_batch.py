import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
# from data.dataset import FSCDataset
from data.fsc.augment import *

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


# 2. dataset
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
# from augment import *

class BaseDataset(Dataset):
    """
    A dataset should implement
        1. __len__ to get size of the dataset, Required
        2. __getitem__ to get a single data, Required

    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

class FSCDataset(BaseDataset):
    def __init__(
        self,
        img_dir,
        density_dir,
        meta_file,
        shot,
        transform_fn=None,
        transform_train=None
    ):
        self.img_dir = img_dir
        self.density_dir = density_dir
        self.meta_file = meta_file
        self.shot = shot
        self.transform_fn = transform_fn
        self.transform_train = transform_train

        # construct metas
        if isinstance(meta_file, str):
            meta_file = [meta_file]
        self.metas = []
        for _meta_file in meta_file:
            with open(_meta_file, "r+") as f_r:
                for line in f_r:
                    meta = json.loads(line)
                    self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        meta = self.metas[index]
        # read img
        img_name = meta["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        # read density
        density_name = meta["density"]
        density_path = os.path.join(self.density_dir, density_name)
        density = np.load(density_path)
        # get boxes, h, w
        boxes = meta["boxes"]
        if self.shot:
            boxes = boxes[: self.shot]
        # points
        points = meta["points"]
        
        # list int64 => numpy float32
        boxes = np.array(boxes, dtype=np.float32)
        points = np.array(points, dtype=np.float32)
        
        # transform
        if self.transform_train:
            image, density, boxes, points = self.transform_train(
                image, density, boxes, points
            )
        if self.transform_fn:
            image, density, boxes, points = self.transform_fn(
                image, density, boxes, points
            )
        size = [height, width]
        return image, density, boxes, points, size, img_name


## 公共部分
class FSCTransforms(object):
    def __init__(self, size=(512, 512), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.size = size
        self.mean = mean
        self.std = std
        self.augment = Compose([
            Resize(self.size),    # resize ：image、density，boxes， points 相对位置不用变 
            ConvertFromInts(),    # 将 uint64 -> float32
            Normalization(self.mean, self.std),   # /255 -mean/std
            ToTensor(), # 训练，img:c,h,w, 所有数据变为 tensor 格式 
        ])

    def __call__(self, img, density, boxes, points):
        return self.augment(img, density, boxes, points)

## 训练集
class FSCTrainTrans(object):
    def __init__(self):
        self.augment = Compose([
            RandomHFilp(),      # 水平翻转
            # RandomVFilp(),      # 竖直翻转
            ConvertFromInts(),
            PhotometricDistort(),     # 光学变换
        ])

    def __call__(self, img, density, boxes, points):
        return self.augment(img, density, boxes, points)

# 数据加 batch
def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    densitys = torch.stack(transposed_batch[1], 0)
    
    # boxes = torch.stack(transposed_batch[2], 0) # transposed_batch[2]  
    boxes = transposed_batch[2]
    points = transposed_batch[3]  # the number of points is not fixed, keep it as a list of tensor
    st_sizes = torch.FloatTensor(transposed_batch[4])
    file_name = transposed_batch[5]
    return images, densitys, boxes, points, st_sizes, file_name 

def data_loader(cfg, phase, distributed):
    img_dir = cfg["img_dir"]
    density_dir = cfg["density_dir"]
    shot = cfg["shot"]
    size = cfg["size"]
    mean = cfg["mean"]
    std = cfg["std"]

    if phase=='train':
        dataset = FSCDataset(
            img_dir,
            density_dir,
            cfg["train"]["meta_file"],
            shot,
            transform_fn=FSCTransforms(size=size,mean=mean,std=std),
            transform_train=FSCTrainTrans()
        )
    elif phase=='val':
        dataset = FSCDataset(
            img_dir,
            density_dir,
            cfg["val"]["meta_file"],
            shot,
            transform_fn=FSCTransforms(),
            transform_train=None
        )
    elif phase=='test':
        dataset = FSCDataset(
            img_dir,
            density_dir,
            cfg["test"]["meta_file"],
            shot,
            transform_fn=FSCTransforms(),
            transform_train=None
        )
    else:
        raise ValueError("phase must among [train, val, test]!")
    
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    
    data_loader = DataLoader(
                    dataset,
                    collate_fn=(train_collate ), #if phase=='train' else default_collate),
                    batch_size=(cfg["batch_size"]),
                    num_workers=cfg["workers"],
                    pin_memory=(True if phase=='train' else False),
                    sampler=sampler,
    )                      
    return data_loader
        
def build_dataloader(cfg_dataset, distributed=False):
    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = data_loader(cfg_dataset, phase="train", distributed=distributed)

    val_loader = None
    if cfg_dataset.get("val", None):
        val_loader = data_loader(cfg_dataset, phase="val", distributed=distributed)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = data_loader(cfg_dataset, phase="test", distributed=distributed)
    
    return train_loader, val_loader, test_loader

# loader = build_dataloader(cfg.dataset, distributed=False)
# for index, sample in enumerate(loader[1]):   # 3: train, val, test
#     # 数据基本信息
#     images, densitys, boxes, points, st_sizes, file_name = sample  
#     print(images.shape)
#     batchsize = images.shape[0]
#     print(batchsize)
#     for i in range(batchsize):
#         print("len_points:{}".format(len(points[i])))
#         print("density_sum:{}".format(densitys[i].sum()))
#         show_img_boxes_points(images[i].unsqueeze(0), boxes[i], points[i])
#         show_density(densitys[i])
#     if index==2:
#         break;